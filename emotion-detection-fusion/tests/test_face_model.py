# tests/test_face_model.py
"""
학습된 표정 모델 연결(detection/face_model.py) 검증.
실행: python tests/test_face_model.py
  - 모델 없이: 가짜 인식기로 출력 변환·얼굴 자르기·프레임 간격 로직 확인
  - hsemotion-onnx가 설치돼 있으면: 실제 모델을 불러와 출력 형식까지 확인 (처음엔 모델 16MB 다운로드)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np

from detection.face_model import FaceEmotionModel, KO_LABEL, crop_face, scores_to_affect

passed = 0


def check(cond, msg):
    global passed
    assert cond, "실패: " + msg
    passed += 1
    print("  ✓", msg)


IDX8 = {0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear", 4: "Happiness", 5: "Neutral", 6: "Sadness", 7: "Surprise"}

print("== 1. 출력 변환")
sad = np.array([.02, .01, .02, .05, .02, .18, .68, .02, -0.55, -0.20])          # 확률 8개 + 정서가·각성도
a = scores_to_affect(sad, IDX8, is_mtl=True)
check(a.label == "슬픔" and abs(a.valence + 0.55) < 1e-9 and abs(a.confidence - 0.82) < 1e-9 and abs(a.p_sad - .68) < 1e-9,
      f"슬픔 68% → 라벨 {a.label}, 정서가 {a.valence:+.2f}(모델 회귀값), 신뢰도 {a.confidence:.2f}(1 − 무표정)")
happy = scores_to_affect(np.array([.01, .01, .01, .01, .90, .05, .005, .005]), IDX8, is_mtl=False)
check(happy.label == "행복" and abs(happy.valence - (0.90 - 0.01)) < 1e-9 and happy.arousal == 0.0,
      "정서가·각성도 출력이 없는 모델이면 정서가 = 행복 − 가장 큰 부정 확률")
check(set(KO_LABEL.values()) <= {"무감정", "행복", "슬픔", "분노", "공포", "혐오", "놀람"},
      "모델 라벨이 에이전트 서버의 추천 도구 키(한국어 7종) 안에 있음")

print("== 2. 얼굴 자르기")
frame = np.zeros((480, 640, 3), dtype=np.uint8)
frame[100:300, 200:400] = (10, 20, 30)          # BGR
c = crop_face(frame, (200, 100, 400, 300))
check(c is not None and c.shape == (260, 260, 3) and tuple(c[130, 130]) == (30, 20, 10),
      "얼굴 상자에 15% 여유를 두고 자르고 BGR → RGB로 바꿈")
check(crop_face(frame, (100, 100, 110, 112)) is None and crop_face(frame, None) is None,
      "너무 작거나 없는 얼굴 상자는 건너뜀")


class FakeRec:
    idx_to_class = IDX8
    is_mtl = True

    def __init__(self): self.calls = 0

    def predict_emotions(self, img, logits=True):
        self.calls += 1
        return "Sadness", sad.copy()


print("== 3. 프레임 간격")
rec = FakeRec()
m = FaceEmotionModel(every=3, recognizer=rec)
outs = [m.update(frame, (200, 100, 400, 300)) for _ in range(9)]
check(rec.calls == 3 and all(o is not None and o.label == "슬픔" for o in outs),
      "3프레임마다 한 번 실행하고, 사이 프레임은 직전 결과를 씀 (9프레임에 3회)")
check(m.update(frame, None) is None, "얼굴을 놓치면 결과 없음")

try:
    import hsemotion_onnx  # noqa: F401
    HAS = True
except ImportError:
    HAS = False
print("== 4. 실제 모델" + ("" if HAS else " (hsemotion-onnx 미설치 → 건너뜀)"))
if HAS:
    real = FaceEmotionModel(every=1)
    out = real.update((np.random.rand(480, 640, 3) * 255).astype(np.uint8), (200, 100, 400, 300))
    check(out is not None and len(out.probs) == 8 and abs(sum(out.probs.values()) - 1) < 1e-3
          and -1 <= out.valence <= 1 and -1 <= out.arousal <= 1,
          f"실제 모델 출력: 확률 8개(합 1), 정서가·각성도 범위 정상 (임의 이미지 → {out.label_en})")

print(f"\n모든 검증 통과: {passed}개")
