# tests/test_offline.py
"""
카메라·마이크·서버 없이 돌리는 검증 (합성 데이터).
실행: emotion-detection-fusion 폴더에서  python tests/test_offline.py
"""
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np

# mediapipe가 없는 환경에서도 rppg/fusion을 불러올 수 있게 (roi.py만 mediapipe 필요)
try:
    import mediapipe  # noqa: F401
except ImportError:
    sys.modules["mediapipe"] = types.ModuleType("mediapipe")

from ai.emotion_label import dominant_label, label_from_blendshapes
from ai.stress_index import BaselineHR, StressConfig, StressIndexer

FPS = 15.0
passed = 0


def check(cond, msg):
    global passed
    assert cond, "실패: " + msg
    passed += 1
    print("  ✓", msg)


def feed(indexer, seconds, fn, t0=0.0, detected=True, label="무감정"):
    """fn(t) → (정서가, 표정 각성도, 신뢰도) 또는 (..., 심박 상승 bpm 또는 None)"""
    n = int(seconds * FPS)
    for i in range(n):
        t = t0 + i / FPS
        out = fn(t)
        v, a, c = out[:3]
        hr = out[3] if len(out) > 3 else None
        indexer.add(t, detected, v, a, c, label, hr)
    return t0 + n / FPS


print("== 1. 스트레스 지수 (부정 0.35 · 심박 상승 0.35 · 표정 각성 0.10 · 기복 0.20)")
rng = np.random.default_rng(0)
cases = {}
for name, fn, label in [
    ("평온 (중립 표정, 평상시 심박)", lambda t: (rng.normal(0, .02), rng.normal(0, .02), .1, rng.normal(0, 2)), "무감정"),
    ("심박만 상승 (+20bpm, 표정 없음)", lambda t: (0.0, 0.0, 0.0, 20.0), "무감정"),
    ("부정 표정만 (심박 평상시)", lambda t: (-0.4, 0.0, 0.8, 0.0), "분노"),
    ("부정 표정 + 심박 +20", lambda t: (-0.4, 0.0, 0.8, 20.0), "분노"),
    ("미소 + 심박 +20 (신남·운동 직후)", lambda t: (0.8, 0.3, 1.0, 20.0), "행복"),
    ("표정이 3초마다 뒤집힘", lambda t: ((0.3 if int(t // 3) % 2 else -0.3), 0.0, 0.6, 0.0), "무감정"),
    ("부정 표정만, 심박 모름 (재정규화)", lambda t: (-0.4, 0.0, 0.8), "분노"),
    ("부정 + 표정 각성, 심박 모름", lambda t: (-0.4, 0.5, 1.0), "공포"),
]:
    ix = StressIndexer(StressConfig())
    feed(ix, 70, fn, label=label)
    r = ix.compute()
    cases[name] = r
    hu = "모름" if r.hr_up is None else f"{r.hr_up:.2f}"
    print(f"  {name:30} 지수 {r.index:4.1f}  (부정 {r.neg:.2f} 심박 {hu} 각성 {r.arousal:.2f} 기복 {r.instab:.2f}) {r.label}")
c = list(cases.values())
check(c[0].index < 1.5, "평온하면 지수가 낮다")
check(abs(c[1].index - 3.5) < 0.2, "심박 상승만으로는 3.5점 → 개입 기준(6) 미만")
check(abs(c[2].index - 3.5) < 0.2, "부정 표정만으로도 3.5점")
check(c[3].index >= 6.9, "부정 표정과 심박 상승이 함께면 7점 → 개입 대상 (표정 신뢰도와 상관없이 심박이 그대로 반영)")
check(c[4].index < 6, "웃으면서 심박이 오르면 개입하지 않음")
check(c[5].instab > 0.9 and c[5].neg < 0.2, "표정이 자주 뒤집히면 평균은 중립이어도 기복 점수가 오른다")
check(c[6].hr_up is None and 5.0 < c[6].index < 6, "심박을 모르면 재정규화: 부정 표정만으로 약 5.4점 (개입 기준 미만)")
check(c[7].index >= 6, "심박을 몰라도 부정 표정 + 표정 각성이 함께면 개입")

ix = StressIndexer(StressConfig())
feed(ix, 10, lambda t: (-0.4, 0.5, 0.8, 20.0), detected=False)
check(ix.compute() is None, "얼굴이 안 잡힌 구간은 판단을 보류한다")

print("== 2. 평상시 심박 기준선")
b = BaselineHR(calib_sec=60)
t = 0.0
for k in range(59):
    b.add(t, 70 + (k % 5) - 2); t += 1
check(not b.ready, "보정 60초 전에는 준비 안 됨")
b.add(60.0, 71)
check(b.ready and abs(b.value - 70) <= 1, f"60초 뒤 중앙값으로 기준선 확정 ({b.value:.0f} bpm)")
check(BaselineHR(fixed_bpm=65).ready, "RESTING_HR를 주면 보정 없이 바로 사용")
b2 = BaselineHR(calib_sec=60, timeout_sec=120)   # 심박이 한 번도 안 잡히는 경우
check(b2.status(119) == "calibrating" and b2.status(120) == "unavailable",
      "120초 안에 보정이 안 되면 unavailable → 표정만으로 동작")
for k in range(70):
    b2.add(130 + k, 72)
check(b2.status(200) == "ready", "시간 초과 뒤에도 심박이 잡히면 보정이 끝나 융합으로 복귀")

from ai.stress_index import is_octave_jump, majority_octave_median
check(majority_octave_median([52, 54, 104, 106, 108]) == 106,
      "보정 표본이 절반·두 배 두 무리로 갈리면 높은 무리 (실제 기록: 진짜 104, 낮은 무리 54)")
check(majority_octave_median([70, 71, 72, 140]) == 71, "두 배 값이 30% 미만이면 무시")
check(majority_octave_median([60, 62, 61, 65]) == 61.5, "한 무리면 그대로 중앙값")
check(is_octave_jump(52, 104) and is_octave_jump(208, 104) and not is_octave_jump(90, 104) and not is_octave_jump(120, 104),
      "보정 후에는 평상시의 절반·두 배 근처만 버리고, 90·120 같은 실제 변화는 받아들임")
b3 = BaselineHR(calib_sec=60)
for k, bpm in enumerate([108] * 12 + [54] * 18 + [111] * 4):   # 4차 실행 첫 60여 초와 같은 구성
    b3.add(k * 2.0, bpm)
check(b3.value is not None and b3.value >= 100, f"4차 실행과 같은 표본 구성에서 평상시 심박 {b3.value:.0f} (이전 방식은 54)")

print("== 2-1. 표정 신뢰도 (실제 웹캠 결과 반영)")
import importlib.util as _iu   # roi.py만 단독으로 불러옴 (패키지 전체는 scipy 필요)
_spec = _iu.spec_from_file_location("_roi_only", ROOT / "emotion_pipeline" / "roi.py")
_roi = _iu.module_from_spec(_spec); sys.modules["_roi_only"] = _roi; _spec.loader.exec_module(_roi)
blendshapes_to_affect = _roi.blendshapes_to_affect
_, _, conf_blink = blendshapes_to_affect({"eyeBlinkLeft": 0.45, "eyeBlinkRight": 0.45})
check(conf_blink == 0.0, f"무표정 + 눈꺼풀만 내려옴 → 신뢰도 {conf_blink:.2f} (수정 전에는 1.00)")
_, _, conf_smile = blendshapes_to_affect({"mouthSmileLeft": 0.5, "mouthSmileRight": 0.5, "eyeBlinkLeft": 0.45})
check(conf_smile == 1.0, "뚜렷한 미소는 여전히 신뢰도 높음")

print("== 3. 블렌드셰입 → 감정 라벨")
L = lambda **kw: label_from_blendshapes(kw)
check(L(mouthSmileLeft=.7, mouthSmileRight=.7) == "행복", "웃음 → 행복")
check(L(browDownLeft=.6, browDownRight=.6, mouthPressLeft=.3, mouthPressRight=.3) == "분노", "미간 찌푸림 → 분노")
check(L(mouthFrownLeft=.5, mouthFrownRight=.5, browInnerUp=.5) == "슬픔", "입꼬리 내림 + 눈썹 안쪽 올림 → 슬픔")
check(L(eyeWideLeft=.6, eyeWideRight=.6, browInnerUp=.6, mouthStretchLeft=.5, mouthStretchRight=.5) == "공포", "눈 크게 + 눈썹 올림 + 입 옆으로 → 공포")
check(L(eyeWideLeft=.6, eyeWideRight=.6, jawOpen=.7, browOuterUpLeft=.5, browOuterUpRight=.5) == "놀람", "눈 크게 + 입 벌림 → 놀람")
check(L(noseSneerLeft=.5, noseSneerRight=.5) == "혐오", "코 찡그림 → 혐오")
check(L(mouthSmileLeft=.05, browDownLeft=.05) == "무감정", "약한 움직임 → 무감정")
check(dominant_label(["무감정"] * 7 + ["분노"] * 3) == "분노", "구간의 1/4 이상 분노면 대표 라벨은 분노")

print("== 4. 실제 처리 fps에서 rPPG 심박 + 품질 기준")
try:
    import scipy  # noqa: F401
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("  (scipy 없음 → 건너뜀)")
if HAS_SCIPY:
    from emotion_pipeline.rppg import CHROMExtractor
    for fps in (15.0, 24.0):
        ex = CHROMExtractor(fs=fps, window_sec=20)
        t = np.arange(int(25 * fps)) / fps
        pulse = 0.004 * np.sin(2 * np.pi * (72 / 60) * t)
        est = None
        for k in range(len(t)):
            noise = rng.normal(0, 0.002, 3)
            est = ex.update((150 * (1 + 0.5 * pulse[k] + noise[0]), 110 * (1 + pulse[k] + noise[1]), 90 * (1 + 0.3 * pulse[k] + noise[2])))
        check(abs(est.bpm - 72) < 3 and est.quality >= 0.35, f"{fps:.0f}fps 정답 72 → 추정 {est.bpm:.1f} bpm, 품질 {est.quality:.2f}")
    passed_noise = total_noise = 0
    for seed in range(10):   # 맥박 없는 잡음 10회
        r2 = np.random.default_rng(seed)
        ex = CHROMExtractor(fs=15.0, window_sec=20)
        last = None
        for k in range(int(40 * 15)):
            est = ex.update(tuple(np.array([150, 110, 90]) * (1 + r2.normal(0, 0.002, 3))))
            if est is not None and est is not last:
                last = est
                total_noise += 1
                passed_noise += est.quality >= 0.35
    check(passed_noise / total_noise < 0.02,
          f"맥박 없는 잡음: 품질 기준(0.35) 통과 {passed_noise}/{total_noise}회 → 사실상 전부 버려짐")
    ex_wrong = CHROMExtractor(fs=30.0, window_sec=20)   # 실제 15fps인데 30으로 잘못 넣으면
    for k in range(int(45 * 15)):   # 30fps 기준 20초 버퍼(600프레임)를 채우려면 15fps로 40초 이상 필요
        tt = k / 15.0
        p = 0.004 * np.sin(2 * np.pi * 1.2 * tt)
        est = ex_wrong.update((150 * (1 + .5 * p), 110 * (1 + p), 90 * (1 + .3 * p)))
    print(f"  (참고) 실제 15fps를 30fps로 잘못 넣으면 72 bpm이 {est.bpm:.0f} bpm으로 나옴 → 실제 fps 측정이 필요한 이유")

print("== 5. 개입 흐름 (서버·백엔드·TTS는 가짜로 대체)")
os.environ["DETECTION_INTERVAL"] = "0"
import main as M
from ai.session_logger import SessionLogger
from state.emotion_state import stop_event, update_state

ix = StressIndexer(StressConfig())
feed(ix, 30, lambda t: (-0.4, 0.0, 0.8, 20.0), label="분노")
update_state(resting_hr=70.0, hr_bpm=91.0, hr_mode="fusion")
calls = {}

def fake_llm(**kw):
    calls["llm"] = kw
    json.dumps(kw)   # 서버로 보낼 수 있는 형식인지
    return {"stress_level": 8, "recommendation": "많이 긴장하신 것 같아요. 잠깐 쉬어 가요.",
            "intervention_id": 11, "recommendation_id": 22}

def fake_log(**kw):
    json.dumps(kw); calls["log"] = kw; return 5

sleeps = {"n": 0}
def fake_sleep(_):
    sleeps["n"] += 1
    if sleeps["n"] > 2:
        stop_event.set()

with tempfile.TemporaryDirectory() as d:
    slog = SessionLogger(d)
    cooldown_results = iter([(False, 0), (None, 0)])   # 1회차: 백엔드 정상, 2회차: 백엔드 장애
    with mock.patch.object(M, "check_cooldown", lambda: next(cooldown_results)), \
         mock.patch.object(M, "save_emotion_log", fake_log), \
         mock.patch.object(M, "request_llm", fake_llm), \
         mock.patch.object(M, "speak", lambda text: calls.setdefault("tts", text)), \
         mock.patch.object(M.time, "sleep", fake_sleep):
        M.run_stress_loop(ix, slog)
    slog.close()
    events = Path(slog.events_path).read_text(encoding="utf-8-sig")
stop_event.clear()
M.CONV.end("test")   # 개입 뒤 대화 모드에 들어가므로 다음 시험 전에 끝낸다

check(calls["llm"]["vad_stress"] >= 6.9 and calls["llm"]["face_emotion"] == "분노",
      f"서버 요청: vad_stress={calls['llm']['vad_stress']}, face_emotion={calls['llm']['face_emotion']}")
check(calls["log"]["vad_stress_score"] == calls["llm"]["vad_stress"], "백엔드 감정 로그도 같은 지수로 저장")
import time as _t; _t.sleep(0.3)   # TTS는 별도 스레드에서 실행됨
check(calls.get("tts", "").startswith("많이 긴장"), "개입 메시지를 TTS로 읽음")
check("intervention" in events, "개입 사건이 events CSV에 기록")
check("cooldown_skip" in events, "2회차: 백엔드 장애 시 로컬 쿨다운으로 재개입 방지")

update_state(resting_hr=None, hr_bpm=None, hr_mode="calibrating")
calls.clear(); sleeps["n"] = 0
with tempfile.TemporaryDirectory() as d:
    slog = SessionLogger(d)
    with mock.patch.object(M, "request_llm", fake_llm), mock.patch.object(M.time, "sleep", fake_sleep):
        M.run_stress_loop(ix, slog)
    slog.close()
stop_event.clear()
check("llm" not in calls, "평상시 심박 보정 전에는 지수가 높아도 개입하지 않음")

update_state(hr_mode="face_only")
calls.clear(); sleeps["n"] = 0
with tempfile.TemporaryDirectory() as d:
    slog = SessionLogger(d)
    with mock.patch.object(M, "check_cooldown", lambda: (False, 0)), \
         mock.patch.object(M, "save_emotion_log", fake_log), \
         mock.patch.object(M, "request_llm", fake_llm), \
         mock.patch.object(M, "speak", lambda text: None), \
         mock.patch.object(M.time, "sleep", fake_sleep):
        M.run_stress_loop(ix, slog)
    slog.close()
    events = Path(slog.events_path).read_text(encoding="utf-8-sig")
stop_event.clear()
M.CONV.end("test")   # 개입 뒤 대화 모드에 들어가므로 다음 시험 전에 끝낸다
check("llm" in calls and "[face_only]" in events, "심박 보정 실패(표정만) 모드에서도 에이전트가 동작하고, 모드가 기록됨")

# 대화 중에는 지수가 높고 쿨다운이 끝났어도 새로 말을 걸지 않는다 (2026-11-03)
M.CONV.end("test")
M.CONV.start("이야기 나눠요", 99, speak=False)
calls.clear(); sleeps["n"] = 0
with tempfile.TemporaryDirectory() as d:
    slog = SessionLogger(d)
    with mock.patch.object(M, "check_cooldown", lambda: (False, 0)), \
         mock.patch.object(M, "save_emotion_log", fake_log), \
         mock.patch.object(M, "request_llm", fake_llm), \
         mock.patch.object(M.time, "sleep", fake_sleep):
        M.run_stress_loop(ix, slog)
    slog.close()
    events = Path(slog.events_path).read_text(encoding="utf-8-sig")
stop_event.clear()
M.CONV.end("test")
check("llm" not in calls and "chat_skip" in events, "대화 중에는 새 개입을 하지 않음")

# 시연용 '지금 말 걸기': 지수가 기준보다 낮고 쿨다운 중이어도 force=True로 개입 (2026-11-04)
from state.emotion_state import manual_trigger
ix_low = StressIndexer(StressConfig())
for k in range(40):
    ix_low.add(k * 0.25, True, 0.2, 0.0, 0.3, "무감정", 0.0)   # 평온 → 지수 낮음
update_state(hr_mode="calibrating")                              # 보정 중이어도
calls.clear(); sleeps["n"] = 0
manual_trigger.set()
with tempfile.TemporaryDirectory() as d:
    slog = SessionLogger(d)
    with mock.patch.object(M, "check_cooldown", lambda: (True, 20)), \
         mock.patch.object(M, "save_emotion_log", fake_log), \
         mock.patch.object(M, "request_llm", fake_llm), \
         mock.patch.object(M, "speak", lambda text: None), \
         mock.patch.object(M.time, "sleep", fake_sleep):
        M.run_stress_loop(ix_low, slog)
    slog.close()
    events = Path(slog.events_path).read_text(encoding="utf-8-sig")
stop_event.clear()
M.CONV.end("test")
check(calls.get("llm", {}).get("force") is True and "manual_trigger" in events, "시연 요청: 지수 낮음·쿨다운 중이어도 force로 개입")
check(calls["llm"].get("memory_key") and isinstance(calls["llm"].get("speech_history"), list), "개입 요청에 기억 키·최근 발화 포함")
check(calls["log"]["extra"]["logType"] == "manual", "시연 요청 기록은 logType=manual로 구분")
check(not manual_trigger.is_set(), "시연 요청은 한 번 쓰면 지워짐")

# 전시용 '새 관람객' (2026-11-04)
from state.emotion_state import get_state, request_visitor_reset, visitor_reset
import ai.cooldown as _cd
M.CONV.start("괜찮으세요?", 7, speak=False)
manual_trigger.set(); _cd.update_intervention_time()
update_state(last_intervention_id=7, stt_text="아까 한 말")
request_visitor_reset()
st = get_state()
check(visitor_reset.is_set() and not M.CONV.is_active() and M.CONV.snapshot()["messages"] == [],
      "새 관람객: 카메라 루프에 기준선 재보정 요청 + 대화 비움")
check(not manual_trigger.is_set() and _cd.last_intervention_time == 0.0 and st["last_intervention_id"] is None
      and st["stt_text"] is None, "새 관람객: 시연 요청·로컬 쿨다운·발화·최근 개입 비움")
visitor_reset.clear()
ix_c = StressIndexer(StressConfig()); ix_c.add(0, True, -0.5, 0.2, 0.8, "분노", None); ix_c.clear()
check(ix_c.compute(1.0) is None, "지수 기록 비우기(clear)")
check(M.MEMORY["key"].startswith("user-") and "visitor" in M.MEMORY["key"], "새 관람객: 지난 대화 기억 공간도 새로")

# 평소 켜 두고 쓸 때: 대화 밖 발화를 주기적으로 서버에 보내 요약 (2026-11-04)
from ai.stt_buffer import add_stt_text, clear_stt, get_speech_since
clear_stt()
M.MEMORY["speech_sent"] = 0.0
for x in ["아 발표 자료 언제 다 하지", "팀장님이 또 바꾸래", "커피나 마셔야겠다"]:
    add_stt_text(x)
add_stt_text("고마워 이제 괜찮아", in_chat=True)
check([x["text"] for x in get_speech_since(0)] == ["아 발표 자료 언제 다 하지", "팀장님이 또 바꾸래", "커피나 마셔야겠다"],
      "대화 중에 한 말은 혼잣말 요약에서 뺌")
sent_mem = []
waits = iter([False, True])
with mock.patch.object(M, "STT_ENABLED", True), mock.patch.object(M, "SPEECH_SUMMARY_EVERY_SEC", 1), \
     mock.patch.object(M.stop_event, "wait", lambda s: next(waits)), \
     mock.patch.object(M, "request_speech_memory", lambda k, items: sent_mem.append((k, len(items))) or True):
    M.run_speech_memory()
check(sent_mem == [(M.MEMORY["key"], 3)] and M.MEMORY["speech_sent"] > 0, "주기마다 대화 밖 발화를 기억 공간 키와 함께 보냄")
check(get_speech_since(M.MEMORY["speech_sent"]) == [], "보낸 말은 다음 주기에 다시 보내지 않음")
clear_stt()

print(f"\n모든 검증 통과: {passed}개")
