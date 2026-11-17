# tests/test_chat.py
"""
이어지는 대화(/chat) 검증 (2026-11-03): 대화 기억, 추천 도구 재호출, 추천 저장, 끝내기.
실행: python tests/test_chat.py   (가짜 LLM·가짜 추천 도구 — API 키·네트워크 불필요)
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agent.chat import ChatManager, parse_tool_items, pick_mentioned

passed = 0


def check(cond, msg):
    global passed
    assert cond, "실패: " + msg
    passed += 1
    print("  ✓", msg)


@tool
def recommend_music(face_emotion: str) -> str:
    """가짜 음악 추천"""
    return ("[행복] 감정 기반 추천 음악 (검색: happy pop):\n"
            "- Good Day / IU\n  https://open.spotify.com/track/aaa\n"
            "- Shake It Off / Taylor Swift\n  https://open.spotify.com/track/bbb")


class FakeLLM:
    """정해 둔 답을 차례로 돌려주고, 받은 메시지를 기록한다."""
    def __init__(self, answers):
        self.answers, self.seen = list(answers), []
    def invoke(self, msgs):
        self.seen.append(list(msgs))
        return self.answers.pop(0)


print("[1] 도구 출력 해석")
items = parse_tool_items("recommend_music", recommend_music.invoke({"face_emotion": "행복"}))
check([i["title"] for i in items] == ["Good Day", "Shake It Off"] and items[0]["artist"] == "IU"
      and items[1]["url"].endswith("bbb") and items[0]["category"] == "MUSIC", "제목·아티스트·URL·분류")
check(pick_mentioned(items, "'Shake It Off' 어때요?")[0]["title"] == "Shake It Off", "답에 나온 곡만 고름")
check(pick_mentioned(items, "신나는 곡 골라 봤어요")[0]["title"] == "Good Day", "제목이 안 나오면 첫 곡")

print("[2] 대화 기억 + 도구 재호출 + 저장")
saved = []
log = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
tools_llm = FakeLLM([
    AIMessage(content="회사 일이 많으셨군요. 어떤 부분이 제일 힘드셨어요?"),
    AIMessage(content="", tool_calls=[{"name": "recommend_music", "args": {"face_emotion": "행복"}, "id": "c1"}]),
    AIMessage(content="기분 전환엔 IU의 'Good Day' 어떠세요? https://open.spotify.com/track/aaa"),
    AIMessage(content="오늘 이야기 나눠 줘서 고마워요. 푹 쉬세요! <END>"),
])
m = ChatManager(FakeLLM([]), tools_llm, [recommend_music], save_rec=lambda **kw: saved.append(kw), log_path=log)
ctx = {"face_emotion": "슬픔", "stress_index": 7.2, "heart_rate": 95, "resting_heart_rate": 72}
r1 = m.chat("iv-42", 1, "회사 일이 너무 많아요", intervention_id=42,
            opening="많이 지쳐 보여요. 'Weightless' 들어 보실래요?", context=ctx)
check(r1["reply"].startswith("회사 일이") and not r1["ended"] and r1["recommendations"] == [], "첫 대답: 공감 + 질문")
first = tools_llm.seen[0]
check(isinstance(first[0], SystemMessage) and "슬픔" in first[0].content and "95bpm" in first[0].content,
      "지금 감지된 상태가 시스템 프롬프트에 들어감")
check("Weightless" in first[0].content, "첫 메시지에서 추천한 곡을 '이미 추천' 목록에 넣음")
check([type(x).__name__ for x in first[1:]] == ["AIMessage", "HumanMessage"], "첫 메시지(에이전트) + 사용자 말 순서")

r2 = m.chat("iv-42", 1, "신나는 노래 없어?", intervention_id=42, context=ctx)
check(r2["recommendations"][0]["title"] == "Good Day", "추천 요청 → 도구 호출 → 새 추천")
check("http" not in r2["reply"], "음성으로 읽을 답에서 URL 제거")
second_round = tools_llm.seen[2]
check(any(isinstance(x, ToolMessage) and x.tool_call_id == "c1" for x in second_round), "도구 결과를 LLM에 다시 넘김")
check(len(tools_llm.seen[1]) == 1 + 4, "두 번째 턴에 앞선 대화(4개)를 기억함")
check(saved and saved[0]["intervention_id"] == 42 and saved[0]["external_url"].endswith("aaa")
      and saved[0]["reason"] == "대화 중 추천", "새 추천을 개입 42의 추천으로 저장")

r3 = m.chat("iv-42", 1, "고마워 이제 괜찮아", intervention_id=42, context=ctx)
check(r3["ended"] and "<END>" not in r3["reply"], "작별 인사 → ended=true, 표시는 지움")
h = m.history("iv-42")
check([x["role"] for x in h["messages"]] == ["agent", "user", "agent", "user", "agent", "user", "agent"] and h["ended"],
      "대화 기록 7개, 끝난 상태")
rows = [json.loads(l) for l in open(log, encoding="utf-8")]
check(len(rows) == 7 and rows[0]["kind"] == "opening" and rows[4]["tools"] == ["recommend_music"], "대화 로그 jsonl")

print("[3] 실패·예외")
m2 = ChatManager(FakeLLM([]), FakeLLM([]), [recommend_music], log_path=None)   # 답이 없어 예외
r = m2.chat("free-1", 1, "안녕", context=None)
check("다시 말해" in r["reply"] and not r["ended"], "LLM 오류여도 대화는 이어짐")
check(m2.chat("free-1", 1, "  ")["reply"] == "", "빈 말은 무시")
loop = AIMessage(content="", tool_calls=[{"name": "recommend_music", "args": {"face_emotion": "무감정"}, "id": "x"}])
m3 = ChatManager(FakeLLM([AIMessage(content="'Good Day' 들어 보세요")]), FakeLLM([loop, loop]),
                 [recommend_music], log_path=None)
r = m3.chat("s3", 1, "노래")
check(r["reply"].startswith("'Good Day'"), "도구를 계속 부르면 도구 없이 한 번 더 답을 받음")
check(m.history("nope") is None, "없는 세션은 None")

print("[4] 추천 끔 → 상담형 (2026-11-04)")
class NoTools:
    def invoke(self, msgs):
        raise AssertionError("추천을 끄면 도구 LLM을 부르면 안 된다")
plain = FakeLLM([AIMessage(content="그 상황이면 정말 속상하셨겠어요. 어떤 점이 제일 힘드셨어요?")])
mc = ChatManager(plain, NoTools(), [recommend_music], log_path=None, use_tools=False)
r = mc.chat("c1", 1, "회사에서 혼났어", opening="많이 지쳐 보이세요. 'Weightless' 들어 보실래요?")
check(r["recommendations"] == [] and r["reply"].startswith("그 상황이면"), "도구 없이 공감형 답")
sysmsg = plain.seen[0][0].content
check("추천은 하지 마세요" in sysmsg and "추천 도구를 부르세요" not in sysmsg and "이미 추천한 항목" not in sysmsg,
      "상담 프롬프트 사용 (추천 지시 없음)")

print("[5] 스트리밍 (2026-11-04)")
from agent.chat import _safe_prefix_len
check(_safe_prefix_len("잘 쉬세요 <EN") == len("잘 쉬세요") and _safe_prefix_len("안녕<END>") == 2
      and _safe_prefix_len("안녕하세요") == 5, "<END>나 그 앞부분은 내보내지 않음")


class StreamLLM:
    def __init__(self, pieces):
        self.pieces = pieces
    def stream(self, msgs):
        for p in self.pieces:
            yield AIMessage(content=p)
    def invoke(self, msgs):
        raise AssertionError("스트리밍에서는 invoke를 쓰지 않음")


ms = ChatManager(StreamLLM(["그 상황이면 ", "정말 속상", "하셨겠어요. ", "어떤 점이 ", "힘드셨어요?"]), NoTools(), [],
                 log_path=None, use_tools=False)
ev = list(ms.chat_stream("st1", 1, "혼났어", opening="괜찮으세요?"))
deltas = "".join(e["text"] for e in ev if e["type"] == "delta")
check(len([e for e in ev if e["type"] == "delta"]) >= 4 and ev[-1]["type"] == "done", "조각 여러 개 + 마지막 done")
check(deltas == ev[-1]["reply"] == "그 상황이면 정말 속상하셨겠어요. 어떤 점이 힘드셨어요?", "조각을 합치면 최종 답과 같음")
check([m["role"] for m in ms.history("st1")["messages"]] == ["agent", "user", "agent"], "스트리밍 답도 대화 기록에 남음")
me = ChatManager(StreamLLM(["오늘 이야기 나눠 줘서 고마워요. 푹 쉬세요! <", "EN", "D>"]), NoTools(), [],
                 log_path=None, use_tools=False)
ev = list(me.chat_stream("st2", 1, "고마워 이제 괜찮아"))
deltas = "".join(e["text"] for e in ev if e["type"] == "delta")
check("<" not in deltas and ev[-1]["ended"] and deltas == ev[-1]["reply"], "<END>가 나뉘어 와도 화면에 안 보이고 ended=true")


class BrokenStream:
    def stream(self, msgs):
        raise RuntimeError("연결 끊김")
        yield
ev = list(ChatManager(BrokenStream(), NoTools(), [], log_path=None, use_tools=False).chat_stream("st3", 1, "안녕"))
check(ev[-1]["type"] == "done" and "다시 말해" in ev[-1]["reply"], "스트리밍 실패해도 done으로 마무리")

print(f"\n통과 {passed}개")
