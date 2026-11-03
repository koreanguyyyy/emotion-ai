# tests/test_face_affect.py
"""
개인별 무표정 기준선 + 새 정서가 계산(ai/face_affect.py) 검증.
실행: emotion-detection-fusion 폴더에서  python tests/test_face_affect.py

무표정 프로필은 2026-10-30 실제 웹캠 기록에서 무표정 구간에 자주 켜져 있던 값으로 만들었다
(눈 찡그림 0.45~0.5, 시선 아래 0.4, 눈썹 바깥 올림 0.4, 눈썹 안쪽 0.25, 입술 내밀기 0.3).
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
try:
    import mediapipe  # noqa: F401
except ImportError:
    sys.modules["mediapipe"] = types.ModuleType("mediapipe")

import numpy as np

from ai.face_affect import FaceBaseline, face_affect
from ai.stress_index import StressConfig, StressIndexer

_spec = importlib.util.spec_from_file_location("_roi_only", ROOT / "emotion_pipeline" / "roi.py")
_roi = importlib.util.module_from_spec(_spec); sys.modules["_roi_only"] = _roi; _spec.loader.exec_module(_roi)
old_affect = _roi.blendshapes_to_affect
_fspec = importlib.util.spec_from_file_location("_fusion_only", ROOT / "emotion_pipeline" / "fusion.py")
_fus = importlib.util.module_from_spec(_fspec); sys.modules["_fusion_only"] = _fus; _fspec.loader.exec_module(_fus)
adaptive_fuse, hr_to_arousal = _fus.adaptive_fuse, _fus.hr_to_arousal   # 패키지 전체는 scipy가 필요해서 단독으로 불러옴

passed = 0
rng = np.random.default_rng(0)
FPS = 30


def check(cond, msg):
    global passed
    assert cond, "실패: " + msg
    passed += 1
    print("  ✓", msg)


NEUTRAL = {
    "eyeSquintLeft": 0.50, "eyeSquintRight": 0.45, "eyeLookDownLeft": 0.42, "eyeLookDownRight": 0.40,
    "browOuterUpLeft": 0.40, "browOuterUpRight": 0.30, "browInnerUp": 0.25, "mouthPucker": 0.30,
    "eyeBlinkLeft": 0.20, "eyeBlinkRight": 0.20, "mouthSmileLeft": 0.03, "mouthSmileRight": 0.03,
    "mouthFrownLeft": 0.02, "mouthFrownRight": 0.02, "browDownLeft": 0.03, "browDownRight": 0.03,
    "jawOpen": 0.03, "mouthPressLeft": 0.08, "mouthPressRight": 0.08, "mouthShrugLower": 0.15,
}


def frame(extra=None, noise=0.03):
    bs = {k: float(np.clip(v + rng.normal(0, noise), 0, 1)) for k, v in NEUTRAL.items()}
    for k, v in (extra or {}).items():
        bs[k] = float(np.clip(v + rng.normal(0, noise), 0, 1))
    return bs


WORRIED = {"browInnerUp": 0.75}                                  # 실제 기록: 눈썹 안쪽 0.68
SAD = {"browInnerUp": 0.65, "mouthFrownLeft": 0.45, "mouthFrownRight": 0.45}
SUBTLE_SAD = {"browInnerUp": 0.45, "mouthFrownLeft": 0.20, "mouthFrownRight": 0.20, "mouthShrugLower": 0.65}  # 입꼬리는 살짝, 턱 올림
CHIN_ONLY = {"mouthShrugLower": 0.75}                           # 생각할 때처럼 턱만 올림
ANGRY = {"browDownLeft": 0.55, "browDownRight": 0.55, "mouthPressLeft": 0.45, "mouthPressRight": 0.45}
SMILE = {"mouthSmileLeft": 0.80, "mouthSmileRight": 0.80}        # 실제 기록: 미소 0.81


def mean_affect(extra, base, n=60):
    rs = [face_affect(frame(extra), base) for _ in range(n)]
    labels = [r.label for r in rs]
    return (float(np.mean([r.valence * r.confidence for r in rs])), float(np.mean([r.arousal for r in rs])),
            float(np.mean([r.confidence for r in rs])), max(set(labels), key=labels.count))


print("== 1. 기준선이 없을 때 (실제 실행에서 본 문제 재현)")
v, a, c, l = mean_affect(None, None)
print(f"  무표정 → 신뢰도 {c:.2f}, 정서가 {v:+.2f}, 라벨 {l}")
check(c >= 0.9, "기준선 없이는 무표정인데도 신뢰도가 꽉 참 (원래 켜져 있는 눈썹·눈 동작 때문)")
ov = np.mean([old_affect(frame(WORRIED))[0] * old_affect(frame(WORRIED))[2] for _ in range(60)])
print(f"  2학기 계산으로 눈썹 안쪽 올리기 → 정서가 {ov:+.3f} (실제 기록 최저 −0.04)")
check(abs(ov) < 0.06, "2학기 계산은 눈썹 안쪽 올리기를 정서가에 반영하지 못함")

print("\n== 2. 무표정 기준선 보정")
base = FaceBaseline(calib_sec=60)
for k in range(59 * FPS):
    base.add(k / FPS, frame())
check(not base.ready, "60초 전에는 보정 중")
for k in range(59 * FPS, 61 * FPS):
    base.add(k / FPS, frame())
check(base.ready and abs(base.values["eyeSquintLeft"] - 0.50) < 0.02, "60초 뒤 블렌드셰입별 중앙값으로 기준선 확정")
short = FaceBaseline(calib_sec=60, min_frames=150)
for k in range(100):
    short.add(k * 0.7, frame())                                  # 70초지만 프레임 100개뿐 (얼굴을 자주 놓침)
check(not short.ready, "얼굴이 잡힌 프레임이 150개 미만이면 보정을 끝내지 않음")

print("\n== 3. 기준선 보정 후 표정 읽기")
print("  표정               정서가(신뢰도 반영)  각성도  신뢰도  라벨")
res = {}
for name, extra in (("무표정", None), ("눈썹 안쪽 올리기", WORRIED), ("슬픈 표정", SAD), ("찡그림(분노)", ANGRY), ("미소", SMILE),
                    ("은근한 슬픔(입꼬리 살짝+턱)", SUBTLE_SAD), ("턱만 올림", CHIN_ONLY)):
    res[name] = mean_affect(extra, base)
    v, a, c, l = res[name]
    print(f"  {name:14} {v:+18.2f}  {a:+6.2f}  {c:6.2f}  {l}")
check(res["무표정"][2] < 0.35 and abs(res["무표정"][0]) < 0.05 and res["무표정"][3] == "무감정",
      "무표정: 신뢰도 낮음, 정서가 0 근처, 라벨 무감정")
check(res["눈썹 안쪽 올리기"][0] <= -0.3 and res["눈썹 안쪽 올리기"][3] == "슬픔",
      "눈썹 안쪽 올리기 → 정서가 −0.3 이하, 라벨 슬픔 (라벨과 정서가가 같은 방향)")
check(res["슬픈 표정"][0] <= -0.6, "입꼬리까지 내린 슬픈 표정 → 더 강한 부정")
check(res["찡그림(분노)"][0] <= -0.5 and res["찡그림(분노)"][3] == "분노", "미간 찌푸림 → 분노, 강한 부정")
check(res["미소"][0] >= 0.8 and res["미소"][3] == "행복", "미소 → 강한 긍정, 행복")
check(res["은근한 슬픔(입꼬리 살짝+턱)"][3] == "슬픔" and res["은근한 슬픔(입꼬리 살짝+턱)"][0] <= -0.25,
      "입꼬리를 살짝만 내려도 눈썹·턱 동작이 함께면 슬픔")
check(res["턱만 올림"][3] == "무감정", "턱만 올리는 동작(생각할 때)은 슬픔으로 세지 않음")
from ai.emotion_label import emotion_scores as _es
_old = 0.6 * 0.05 + 0.4 * 0.20   # 이전 식: 은근한 슬픔(기준선 대비 입꼬리 +0.05·눈썹 +0.20)
_new = _es({"mouthFrownLeft": .05, "mouthFrownRight": .05, "browInnerUp": .20, "mouthShrugLower": .50})["슬픔"]
check(_new > _old and _new >= 0.15 > _old, f"같은 은근한 슬픔: 이전 식 {_old:.2f}(기준 0.15 미만) → 새 식 {_new:.2f}")
dn = mean_affect({"eyeLookDownLeft": 0.8, "eyeLookDownRight": 0.8}, base)
check(abs(dn[1]) < 0.1 and dn[2] < 0.35, f"화면을 더 내려다봐도 각성도·신뢰도가 거의 그대로 ({dn[1]:+.2f}, {dn[2]:.2f})")

print("\n== 4. 스트레스 지수까지 (10초 유지, 평상시 심박 70)")


def index_for(extra, hr, hr_known=True):
    ix = StressIndexer(StressConfig())
    for k in range(12 * FPS):
        f = face_affect(frame(extra), base)
        # face_fusion.py와 같은 방식: 표정(신뢰도 반영 정서가, 표정 각성도)과 심박 상승을 따로 넣는다
        ix.add(k / FPS, True, f.confidence * f.valence, f.arousal, f.confidence, f.label,
               (hr - 70) if hr_known else None)
    return ix.compute().index


rows = [("무표정, 심박 평상시", None, 70, True), ("눈썹 안쪽 올리기, 심박 평상시", WORRIED, 70, True),
        ("눈썹 안쪽 올리기 + 심박 +20", WORRIED, 90, True), ("찡그림 + 심박 +20", ANGRY, 90, True),
        ("눈썹 안쪽 올리기, 심박 모름(표정만)", WORRIED, 0, False)]
idx = {}
for name, extra, hr, known in rows:
    idx[name] = index_for(extra, hr, known)
    print(f"  {name:28} 지수 {idx[name]:4.1f}")
check(idx["무표정, 심박 평상시"] < 1.0, "무표정이면 지수 낮음")
check(idx["눈썹 안쪽 올리기, 심박 평상시"] < 6, "표정만 걱정스러우면 개입 기준(6) 미만")
check(idx["눈썹 안쪽 올리기 + 심박 +20"] >= 6, "걱정스러운 표정 + 심박 상승이면 개입 기준 이상")
check(idx["찡그림 + 심박 +20"] >= 6, "찡그림 + 심박 상승도 개입 기준 이상 (변경 전 5.1)")
check(idx["눈썹 안쪽 올리기, 심박 모름(표정만)"] < 6, "심박을 모를 때 걱정스러운 표정만으로는 개입하지 않음")

print(f"\n모든 검증 통과: {passed}개")
