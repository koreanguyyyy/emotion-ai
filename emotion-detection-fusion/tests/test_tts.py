# tests/test_tts.py
"""
음성 출력 검증 (2026-11-04): 문장 나누기, 스트리밍 문장 버퍼, 합성·재생 줄(순서·말하는 중 표시), 실패 시 say.
실행: python tests/test_tts.py   (소리를 내지 않는다 — 합성·재생을 가짜로 바꿔 시험)
"""
import os, sys, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import detection.tts as T

passed = 0
def check(cond, msg):
    global passed
    assert cond, "실패: " + msg
    passed += 1; print("  ✓", msg)

print("[1] 문장 나누기")
s = T.split_sentences("많이 힘드셨겠어요. 어떤 일이 있었는지 이야기해 주실래요? 네. 천천히요!")
check(s == ["많이 힘드셨겠어요.", "어떤 일이 있었는지 이야기해 주실래요?", "네. 천천히요!"], f"문장 단위 + 짧은 조각은 붙임 {s}")
check(T.split_sentences("") == [] and T.split_sentences("한 문장") == ["한 문장"], "빈 글·한 문장")

print("[2] 스트리밍 문장 버퍼")
b = T.SentenceBuffer(); got = []
for piece in ["그 상황이면 정말 ", "속상하셨겠어요. 어떤", " 점이 제일 힘드", "셨어요? 네. 천천", "히 말해 주세요."]:
    got += b.feed(piece)
got.append(b.flush())
check(got == ["그 상황이면 정말 속상하셨겠어요.", "어떤 점이 제일 힘드셨어요?", "네. 천천히 말해 주세요."], f"조각 → 완성된 문장 {got}")

print("[3] 합성·재생 줄")
log = []
def synth(t):
    time.sleep(0.02); log.append(("synth", t)); return ("file", t)
def play(item):
    log.append(("play-start", item[1], T.SPEAKING.is_set())); time.sleep(0.05); log.append(("play-end", item[1]))
sp = T.Speaker(synth, play)
t0 = time.time()
for x in ["하나 문장입니다.", "둘 문장입니다.", "셋 문장입니다."]:
    sp.put(x)
check(T.SPEAKING.is_set(), "넣자마자 '말하는 중' 표시 (마이크 입력 무시)")
sp.wait(5)
plays = [e[1] for e in log if e[0] == "play-start"]
check(plays == ["하나 문장입니다.", "둘 문장입니다.", "셋 문장입니다."], "넣은 순서대로 재생")
i_syn2 = log.index(("synth", "둘 문장입니다.")); i_end1 = log.index(("play-end", "하나 문장입니다."))
check(i_syn2 < i_end1, "첫 문장을 재생하는 동안 다음 문장을 미리 합성")
check(not T.SPEAKING.is_set() and T.was_speaking_since(t0), "다 읽으면 표시 해제, 방금 말한 것은 기억 (잔향 무시)")

print("[4] 신경망 음성 실패 → say")
T._engine, T._fail_count = "edge", 0
orig = T._synth_edge
T._synth_edge = lambda text, path: (_ for _ in ()).throw(OSError("network"))
r = [T.synth_audio("안녕하세요") for _ in range(3)]
check(r[0] == ("say", "안녕하세요") and T._engine == "say", "실패하면 say로 읽고, 3번 연속이면 say로 전환")
T._synth_edge = orig; T._engine = "edge"; T._fail_count = 0
def fake_edge(text, path):
    open(path, "wb").write(b"x" * 500)
T._synth_edge = fake_edge
k, path = T.synth_audio("안녕하세요")
check(k == "file" and os.path.exists(path), "성공하면 mp3 파일")
os.remove(path)

print(f"\n모든 검증 통과: {passed}개")
