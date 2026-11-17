# detection/face_model.py
"""
학습된 표정 모델(HSEmotion) — 규칙 기반 표정 계산(ai/face_affect.py)과 비교하거나 대신 쓰기 위한 옵션.

모델: HSEmotion `enet_b0_8_va_mtl` (EfficientNet-B0, AffectNet 학습, Apache-2.0)
  - 출력: 8가지 감정 확률(분노·경멸·혐오·공포·행복·무표정·슬픔·놀람) + 정서가·각성도 회귀값
  - 공개 정확도: AffectNet 8클래스 61.93%, 7클래스 64.94% (라이브러리 문서 기준)
  - 입력: 얼굴만 잘라낸 RGB 이미지. 얼굴 위치는 MediaPipe 결과(reading.boxes["face"])를 쓴다.

.env 설정
  FACE_MODEL=compare  규칙 기반으로 동작하고, 모델 결과는 세션 CSV에 나란히 기록만 (기본값)
  FACE_MODEL=use      모델의 정서가·각성도·라벨로 스트레스 지수를 계산
  FACE_MODEL=off      모델을 불러오지 않음
  FACE_MODEL_EVERY=3  몇 프레임마다 모델을 돌릴지 (30fps에서 3이면 초당 10회, 사이 프레임은 직전 결과 사용)

설치: pip install hsemotion-onnx  (없으면 경고 후 자동으로 off)

⚠️ 라이브러리 0.3의 모델 다운로드 코드는 `import urllib`만 하고 `urllib.request`를 쓰지 않아 실패할 수 있고,
   github.com/...?raw=true 주소가 일부 망에서 403이 난다. 그래서 처음 한 번은 여기서
   raw.githubusercontent.com 주소로 ~/.hsemotion/ 에 직접 받아 둔다 (약 16MB).
"""
from __future__ import annotations

import os
import urllib.request   # 라이브러리 버그 회피: 먼저 불러 두면 라이브러리 안의 urllib.request도 동작한다
from dataclasses import dataclass, field

import numpy as np

MODEL_NAME = "enet_b0_8_va_mtl"
MODEL_URL = ("https://raw.githubusercontent.com/HSE-asavchenko/face-emotion-recognition/main/"
             "models/affectnet_emotions/onnx/{name}.onnx")

KO_LABEL = {
    "Anger": "분노", "Contempt": "혐오", "Disgust": "혐오", "Fear": "공포",
    "Happiness": "행복", "Neutral": "무감정", "Sadness": "슬픔", "Surprise": "놀람",
}
NEGATIVE = ("Anger", "Contempt", "Disgust", "Fear", "Sadness")


@dataclass
class ModelAffect:
    label: str                 # 한국어 라벨 (에이전트 서버의 추천 도구 선택에 쓰는 것과 같은 7종)
    label_en: str
    valence: float             # −1~1 (모델의 정서가 회귀값. 없으면 행복 확률 − 가장 큰 부정 확률)
    arousal: float             # −1~1 (모델의 각성도 회귀값. 없으면 0)
    confidence: float          # 1 − 무표정 확률 (표정이 뚜렷한 정도)
    probs: dict = field(default_factory=dict)

    @property
    def p_sad(self) -> float:
        return float(self.probs.get("Sadness", 0.0))


def ensure_model_file(name: str = MODEL_NAME) -> str:
    cache = os.path.join(os.path.expanduser("~"), ".hsemotion")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, name + ".onnx")
    if not os.path.isfile(path) or os.path.getsize(path) < 1_000_000:
        print(f"[FaceModel] 표정 모델 {name} 내려받는 중 (약 16MB, 처음 한 번만)...")
        tmp = path + ".part"
        urllib.request.urlretrieve(MODEL_URL.format(name=name), tmp)
        os.replace(tmp, path)
    return path


def crop_face(frame_bgr: np.ndarray, box, margin: float = 0.15) -> np.ndarray | None:
    """MediaPipe 얼굴 상자(x1,y1,x2,y2 픽셀)를 여유를 두고 잘라 RGB로 돌려준다."""
    if not box:
        return None
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 20 or bh < 20:
        return None
    x1 = max(0, int(x1 - margin * bw)); x2 = min(w, int(x2 + margin * bw))
    y1 = max(0, int(y1 - margin * bh)); y2 = min(h, int(y2 + margin * bh))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return np.ascontiguousarray(crop[:, :, ::-1])


def scores_to_affect(scores, idx_to_class: dict, is_mtl: bool) -> ModelAffect:
    """라이브러리 출력(softmax 확률 [+ 정서가, 각성도]) → ModelAffect."""
    scores = np.asarray(scores, dtype=float)
    n = len(idx_to_class)
    probs = {idx_to_class[i]: float(scores[i]) for i in range(n)}
    best = max(probs, key=lambda k: probs[k])
    if is_mtl and len(scores) >= n + 2:
        valence = float(np.clip(scores[n], -1, 1))
        arousal = float(np.clip(scores[n + 1], -1, 1))
    else:
        valence = float(np.clip(probs.get("Happiness", 0.0) - max(probs.get(k, 0.0) for k in NEGATIVE), -1, 1))
        arousal = 0.0
    return ModelAffect(
        label=KO_LABEL.get(best, "무감정"), label_en=best, valence=valence, arousal=arousal,
        confidence=float(np.clip(1.0 - probs.get("Neutral", 0.0), 0, 1)), probs=probs,
    )


class FaceEmotionModel:
    """every 프레임마다 모델을 돌리고, 사이 프레임에는 직전 결과를 돌려준다."""

    def __init__(self, every: int = 3, model_name: str = MODEL_NAME, recognizer=None):
        self.every = max(1, every)
        self._n = 0
        self.last: ModelAffect | None = None
        if recognizer is None:
            ensure_model_file(model_name)
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
            recognizer = HSEmotionRecognizer(model_name=model_name)
        self._rec = recognizer

    def update(self, frame_bgr: np.ndarray, box) -> ModelAffect | None:
        self._n += 1
        if self._n % self.every != 1 and self.every > 1 and self.last is not None:
            return self.last
        face = crop_face(frame_bgr, box)
        if face is None:
            self.last = None
            return None
        _, scores = self._rec.predict_emotions(face, logits=False)
        self.last = scores_to_affect(scores, self._rec.idx_to_class, getattr(self._rec, "is_mtl", False))
        return self.last


def load_face_model(mode: str, every: int) -> FaceEmotionModel | None:
    """mode가 off가 아니면 모델을 불러온다. 설치·다운로드 실패 시 경고하고 None (규칙 기반으로 계속)."""
    if mode == "off":
        return None
    try:
        m = FaceEmotionModel(every=every)
        print(f"[FaceModel] 표정 모델 사용 ({'비교 기록만' if mode == 'compare' else '스트레스 계산에 사용'})")
        return m
    except ImportError:
        print("[FaceModel] hsemotion-onnx가 설치되지 않아 모델 없이 진행합니다 (pip install hsemotion-onnx)")
    except Exception as e:   # 네트워크 등
        print(f"[FaceModel] 표정 모델을 불러오지 못해 모델 없이 진행합니다: {e}")
    return None
