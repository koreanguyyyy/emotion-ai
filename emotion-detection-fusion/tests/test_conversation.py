# tests/test_conversation.py
"""
대화 모드 검증 (2026-11-03): 개입 뒤 이어지는 대화, 잡음 거르기, 끝내기, 웹 채팅 통로.
실행: python tests/test_conversation.py   (카메라·마이크·서버 불필요 — 에이전트 서버와 음성은 가짜)
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ai import conversation as cv
from ai.conversation import Conversation, for_speech, is_noise
from ai.live_server import build_live_payload, start_live_server

passed = 0


def check(cond, msg):
    global passed
    assert cond, "실패: " + msg
    passed += 1
    print("  ✓", msg)


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def make(replies):
    sent, spoken = [], []
    it = iter(replies)
    def send(p):
        sent.append(p)
        return next(it)
    clock = Clock()
    c = Conversation(send=send, speak=spoken.append, get_context=lambda: {"face_emotion": "슬픔", "stress_index": 6.8},
                     user_id=7, idle_sec=90, clock=clock)
    return c, sent, spoken, clock


print("[1] 개입 → 대답 → 이어지는 대화")
c, sent, spoken, clock = make([
    {"reply": "많이 힘드셨겠어요. 어떤 일이 있었어요?", "ended": False, "recommendations": []},
    {"reply": "그럼 'Clair de Lune' 어떠세요?", "ended": False,
     "recommendations": [{"category": "MUSIC", "title": "Clair de Lune", "artist": "Debussy", "url": "https://open.spotify.com/track/x"}]},
])
c.start("오늘 많이 지쳐 보여요. 'Weightless' 들어 보실래요? https://open.spotify.com/track/abc", intervention_id=42)
check(c.is_active() and c.session_id == "iv-42", "개입 ID로 대화 세션 시작")
check(spoken[0].endswith("들어 보실래요?") and "http" not in spoken[0], "첫 메시지를 읽을 때 URL은 소리 내지 않음")
r = c.submit("회사 일이 너무 많아서요", "voice")
check(r["reply"].startswith("많이 힘드셨겠어요"), "음성 대답에 에이전트가 답함")
p = sent[0]
check(p["session_id"] == "iv-42" and p["intervention_id"] == 42 and p["user_id"] == 7, "세션·개입·사용자 ID 전달")
check(p["opening"].startswith("오늘 많이") and p["context"]["face_emotion"] == "슬픔", "첫 메시지와 지금 상태 전달")
r = c.submit("다른 노래 없어?", "voice")
check(len(spoken) == 3 and c.is_active(), "두 번째 턴도 읽고 대화 계속")
snap = c.snapshot()
check([m["role"] for m in snap["messages"]] == ["agent", "user", "agent", "user", "agent"], "대화 기록 순서")
check(snap["messages"][-1]["links"][0]["title"] == "Clair de Lune", "새 추천 링크가 대화 기록에 붙음")

print("[2] 잡음·Whisper 환각 거르기")
check(is_noise("시청해 주셔서 감사합니다.") and is_noise("음") and not is_noise("고마워요"), "환각·너무 짧은 말 판별")
n = len(sent)
check(c.submit("시청해 주셔서 감사합니다.", "voice") is None and len(sent) == n, "대화 중 환각 문장은 보내지 않음")

print("[3] 끝내기")
c2, sent2, spoken2, _ = make([{"reply": "좋아요, 푹 쉬세요!", "ended": True, "recommendations": []}])
c2.start("괜찮으세요?", 5)
c2.submit("고마워 이제 괜찮아", "voice")
check(not c2.is_active() and c2.snapshot()["endReason"] == "agent", "에이전트가 마무리하면 대화 종료")
check(c2.submit("또 말하기", "voice") is None, "종료 뒤 음성은 대화로 가지 않음")
c3, _, _, clock3 = make([])
c3.start("괜찮으세요?", 6)
clock3.t += 91
check(not c3.is_active() and c3.snapshot()["endReason"] == "idle", "90초 동안 말이 없으면 종료")
c3.start("다시", 7); c3.end("user")
check(not c3.is_active() and c3.snapshot()["endReason"] == "user", "웹의 '대화 끝내기'")

print("[4] 서버 실패·웹 채팅")
c4, sent4, spoken4, _ = make([None, {"reply": "안녕하세요!", "ended": False, "recommendations": []}])
c4.start("괜찮으세요?", 8)
r = c4.submit("음 그냥요", "voice")
check("다시 말해" in r["reply"] and c4.is_active(), "에이전트 서버가 실패해도 대화는 유지")
c5, sent5, _, _ = make([{"reply": "반가워요.", "ended": False, "recommendations": []}])
check(c5.submit("안녕", "voice") is None, "대화 중이 아니면 음성은 무시 (평소 혼잣말이 대화로 가지 않게)")
c5.submit("안녕", "web")
check(c5.is_active() and sent5[0]["session_id"].startswith("free-") and sent5[0]["intervention_id"] is None,
      "웹 채팅은 대화가 없으면 새로 시작")

print("[5] on_voice_text 연결")
c6, sent6, _, _ = make([{"reply": "네.", "ended": False, "recommendations": []}])
cv.configure(c6)
check(cv.on_voice_text("안녕하세요") is False, "대화 중이 아니면 False (스트레스 버퍼로만)")
c6.start("어떠세요?", 9)
check(cv.on_voice_text("시청해주셔서 감사합니다") is True and not sent6, "대화 중 잡음은 삼킴")
cv.on_voice_text("조금 나아졌어요")
for _ in range(50):
    if sent6 and not c6.busy: break
    time.sleep(0.02)
check(sent6 and sent6[0]["message"] == "조금 나아졌어요", "대화 중 음성은 비동기로 에이전트에게 감")

print("[6] 실시간 통로 /chat")
c7, sent7, _, _ = make([{"reply": "그랬군요.", "ended": False, "recommendations": []}] * 3)
srv = start_live_server(18766, lambda: build_live_payload({}, None, c7.snapshot()), post_handlers={
    "/chat": lambda b: (c7.submit_async(b["message"], "web"), {"ok": True})[1],
    "/chat/end": lambda b: (c7.end("user"), {"ok": True})[1],
})
def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:18766{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=3) as r:
        return r.status, json.loads(r.read()), dict(r.headers)
code, out, hdr = post("/chat", {"message": "오늘 좀 힘들었어"})
check(code == 200 and out["ok"] and "POST" in hdr["Access-Control-Allow-Methods"], "POST /chat 200 + CORS")
for _ in range(50):
    st = json.loads(urllib.request.urlopen("http://127.0.0.1:18766/state", timeout=3).read())
    if len(st["chat"]["messages"]) >= 3: break
    time.sleep(0.02)
check(st["chat"]["active"] and st["chat"]["messages"][-1]["text"] == "그랬군요.", "/state의 chat에 답이 나타남")
try:
    post("/chat", "x"); bad = False
except urllib.error.HTTPError as e:
    bad = e.code == 400
check(bad, "잘못된 요청은 400")
post("/chat/end", {})
st = json.loads(urllib.request.urlopen("http://127.0.0.1:18766/state", timeout=3).read())
check(not st["chat"]["active"], "POST /chat/end로 종료")
srv.shutdown()

print("[7] 음성용 문장 다듬기")
check(for_speech("**좋아요** 'A' https://x.y/z 들어 보세요") == "좋아요 'A' 들어 보세요", "URL·마크다운 기호 제거")

print("[8] 스트리밍 답 + 문장 단위 음성 (2026-11-04)")
import threading as _th
gate = _th.Event(); seen_mid = {}
def stream(p):
    yield {"type": "delta", "text": "그 상황이면 정말 "}
    yield {"type": "delta", "text": "속상하셨겠어요. 어떤 점이 "}
    gate.wait(2)                                   # 이 사이에 화면 상태를 확인
    yield {"type": "delta", "text": "제일 힘드셨어요?"}
    yield {"type": "done", "reply": "그 상황이면 정말 속상하셨겠어요. 어떤 점이 제일 힘드셨어요?", "ended": False,
           "recommendations": []}
parts, waits = [], []
c8 = Conversation(send=lambda p: (_ for _ in ()).throw(AssertionError("한 번에 받기를 쓰면 안 됨")),
                  speak=lambda t: (_ for _ in ()).throw(AssertionError("통째 읽기를 쓰면 안 됨")),
                  get_context=lambda: {}, idle_sec=90, send_stream=stream,
                  speak_part=parts.append, speak_wait=lambda: waits.append(1))
c8.start("괜찮으세요?", 50, speak=False)
th = _th.Thread(target=c8.submit, args=("회사에서 혼났어", "voice")); th.start()
for _ in range(100):
    snap = c8.snapshot()
    if snap["messages"][-1]["role"] == "agent" and "어떤 점이" in snap["messages"][-1]["text"]:
        break
    time.sleep(0.01)
mid = snap["messages"][-1]
check(mid.get("streaming") is True and mid["text"].endswith("어떤 점이 ") and snap["busy"], "답이 오는 중에 화면에 글자가 늘어남")
check(parts == ["그 상황이면 정말 속상하셨겠어요."], "첫 문장이 완성되자마자 읽기 시작")
gate.set(); th.join(3)
last = c8.snapshot()["messages"][-1]
check(last["streaming"] is False and last["text"].endswith("힘드셨어요?") and parts[-1] == "어떤 점이 제일 힘드셨어요?"
      and waits, "끝나면 나머지 문장 읽고 다 읽을 때까지 기다림")
check(not c8.busy and c8.is_active(), "턴이 끝나면 다시 들을 준비")

def broken(p):
    raise ConnectionError("옛 서버 (404)")
    yield
c9, sent9, spoken9, _ = make([{"reply": "네, 듣고 있어요.", "ended": False, "recommendations": []}])
c9.send_stream = broken
c9.start("괜찮으세요?", 51, speak=False)
r = c9.submit("음 그냥요", "voice")
check(r["reply"] == "네, 듣고 있어요." and sent9 and c9.snapshot()["messages"][-1]["text"] == "네, 듣고 있어요."
      and "" not in [m["text"] for m in c9.snapshot()["messages"]], "스트리밍 실패 → 한 번에 받기, 빈 말풍선 없음")

def cut(p):
    yield {"type": "delta", "text": "잠깐만요, 그러니까"}
    raise ConnectionError("끊김")
c10 = Conversation(send=lambda p: None, speak=lambda t: None, get_context=lambda: {}, idle_sec=90, send_stream=cut)
c10.start("괜찮으세요?", 52, speak=False)
r = c10.submit("그래서요", "web")
check(r["reply"] == "잠깐만요, 그러니까" and not c10.busy, "중간에 끊기면 받은 데까지 보여 주고 마무리")

print("[9] 지난 대화 기억: 기억 키·최근 발화·끝날 때 알림 (2026-11-04)")
ends, keys = [], {"k": "user-7"}
c11, sent11, _, clock11 = make([{"reply": "그러셨군요.", "ended": False, "recommendations": []}] * 3)
c11.get_memory_key = lambda: keys["k"]
c11.get_speech_history = lambda: [{"time": "01:02:03", "text": "아 발표 망했다"}]
c11.on_end = ends.append
c11.start("괜찮으세요?", 60, speak=False)
c11.end("user")
time.sleep(0.05)
check(ends == [], "대화가 오가지 않았으면 끝나도 요약 요청 안 함")
c11.start("괜찮으세요?", 61, speak=False)
c11.submit("발표 때문에 속상해", "voice")
p = sent11[-1]
check(p["memory_key"] == "user-7" and p["speech_history"][0]["text"] == "아 발표 망했다", "기억 키와 최근 발화를 함께 보냄")
c11.end("user"); time.sleep(0.05)
check(ends == ["iv-61"], "웹 '끝내기' → 서버에 대화 종료 알림(요약)")
c11.start("괜찮으세요?", 62, speak=False); c11.submit("응", "web")
clock11.t += 100; c11.check_idle(); time.sleep(0.05)
check(ends[-1] == "iv-62", "무응답 종료도 알림")
c11.start("괜찮으세요?", 63, speak=False); c11.submit("그냥요", "web")
c11.reset(); time.sleep(0.05)
check(ends[-1] == "iv-63" and c11.snapshot()["messages"] == [], "새 관람객: 앞사람 대화를 알리고 화면을 비움")

from ai.stt_buffer import add_stt_text, clear_stt, get_speech_history
clear_stt(); add_stt_text("첫 번째 말"); add_stt_text("두 번째 말")
h = get_speech_history()
check([x["text"] for x in h] == ["첫 번째 말", "두 번째 말"] and len(h[0]["time"]) == 8, "최근 발화 기록 (시각 포함, 읽어도 지워지지 않음)")
check(len(get_speech_history()) == 2 and get_speech_history(sec=-1) == [], "기록 기간 밖은 제외")
clear_stt()

print(f"\n통과 {passed}개")
