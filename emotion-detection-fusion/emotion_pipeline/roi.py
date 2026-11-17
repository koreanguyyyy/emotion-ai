"""
웹캠 프레임 → 얼굴 랜드마크(478점) + 블렌드셰입(52개 표정 특징) + rPPG용 ROI.

⚠️ 버전 주의 (이번에 직접 확인한 내용):
mediapipe 0.10.x부터는 많은 튜토리얼에 나오는 구식 API인
`mp.solutions.face_mesh.FaceMesh(...)` 가 더 이상 존재하지 않는다
(`mp.solutions` 자체가 없음). 반드시 아래처럼 새 Tasks API
(`mp.tasks.vision.FaceLandmarker`)를 써야 한다 — 이 파일은 설치된
mediapipe 0.10.32에서 실제로 클래스 시그니처를 조회해서 맞춘 것이다.

필요한 모델 파일(1회 다운로드, 로컬 PC에서):
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
    (공식 문서: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
    또는: pip install mediapipe 후 아래 명령으로 받을 수 있다.
        curl -L -o face_landmarker.task \\
          https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

⚠️ 이 파일은 클라우드 샌드박스에 카메라도 없고 위 모델 다운로드 경로도
방화벽에 막혀 있어 '실제 얼굴 영상'으로는 테스트하지 못했다. 아래 로직은
mediapipe 공식 Tasks API 문서/설치된 패키지의 실제 함수 시그니처를 그대로
따른 것이지만, 로컬 웹캠에서 돌려보기 전까지는 ROI 좌표 비율 등은 참고값으로
간주할 것 (test_synthetic.py의 한계 섹션 참고).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# mediapipe는 이 모듈(얼굴 랜드마크/블렌드셰입 추출)의 필수 의존성이다
# (requirements.txt에도 명시). 예전엔 try/except로 '소프트 임포트'해서 미설치
# 시 RuntimeError로 안내했는데, 그러면 pyright가 mp를 "possibly unbound/None"
# 으로 봐서 아래 FaceAnalyzer 전체에 거짓 타입 오류를 낸다(실제로 재현 확인함).
# requirements.txt대로 설치하면 항상 있는 모듈이라 일반 import로 바꿨다 —
# 혹시 설치 안 했으면 Python이 내는 기본 ModuleNotFoundError만으로도 충분히
# 명확하다 ("pip install -r requirements.txt" 안내는 README에 있음).
import mediapipe as mp


# ── 블렌드셰입 → 밸런스(valence)/각성(arousal) 매핑 ────────────────────
# mediapipe FaceLandmarker가 직접 계산해 주는 52개 ARKit 스타일 블렌드셰입
# 이름을 그대로 쓴다. 교수님이 말한 "감정의 특징을 인식해라"를 가장 직접적으로
# 구현하는 지점 — 임의의 랜드마크 거리 대신, 모델이 보장하는 이름 붙은 특징을 쓴다.
VALENCE_POS: dict[str, float] = {
    "mouthSmileLeft": 1.0, "mouthSmileRight": 1.0,
    "cheekSquintLeft": 0.3, "cheekSquintRight": 0.3,
}
VALENCE_NEG: dict[str, float] = {
    "mouthFrownLeft": 1.0, "mouthFrownRight": 1.0,
    "browDownLeft": 0.5, "browDownRight": 0.5,
    "noseSneerLeft": 0.6, "noseSneerRight": 0.6,
    "mouthLowerDownLeft": 0.25, "mouthLowerDownRight": 0.25,
    "mouthPressLeft": 0.2, "mouthPressRight": 0.2,
}
AROUSAL_POS: dict[str, float] = {
    "eyeWideLeft": 0.8, "eyeWideRight": 0.8,
    "jawOpen": 0.7,
    "browInnerUp": 0.5,
    "browOuterUpLeft": 0.4, "browOuterUpRight": 0.4,
    "mouthStretchLeft": 0.4, "mouthStretchRight": 0.4,
}
AROUSAL_NEG: dict[str, float] = {
    "eyeBlinkLeft": 0.3, "eyeBlinkRight": 0.3,
}

# 표정 신뢰도(뚜렷함) 계산에서 제외할 블렌드셰입 — 아래 blendshapes_to_affect 참고
CONFIDENCE_EXCLUDE = {"eyeBlinkLeft", "eyeBlinkRight"}

VALENCE_WEIGHTS = {**VALENCE_POS, **{k: -w for k, w in VALENCE_NEG.items()}}  # 하위호환용 조회 테이블
AROUSAL_WEIGHTS = {**AROUSAL_POS, **{k: -w for k, w in AROUSAL_NEG.items()}}

# 긍정/부정을 각자의 가중치 합으로 따로 정규화한다. 예: 웃음만 최대로 지었을 때
# (부정 블렌드셰입은 전부 0) valence가 1.0 가까이 나와야지, 분노 쪽 가중치까지
# 합친 분모로 나누면 어떤 단일 표정도 극단값에 도달하지 못하는 문제가 생긴다.
_VPOS_NORM = sum(VALENCE_POS.values())
_VNEG_NORM = sum(VALENCE_NEG.values())
_APOS_NORM = sum(AROUSAL_POS.values())
_ANEG_NORM = sum(AROUSAL_NEG.values())


@dataclass
class FaceReading:
    detected: bool
    forehead_rgb: tuple[float, float, float] | None = None
    cheek_rgb: tuple[float, float, float] | None = None
    valence: float = 0.0
    arousal: float = 0.0
    confidence: float = 0.0  # 표정이 얼마나 '뚜렷한지' — 작은 표정이면 자동으로 낮아짐
    blendshapes: dict[str, float] = field(default_factory=dict)
    # 화면 표시용 박스(픽셀 좌표 x1,y1,x2,y2): face / forehead / left_cheek / right_cheek
    boxes: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)


def blendshapes_to_affect(scores: dict[str, float]) -> tuple[float, float, float]:
    """블렌드셰입 점수 dict → (valence, arousal, confidence).

    confidence는 '표정이 중립에서 얼마나 벗어났는지'를 보는 것이라서,
    교수님이 지적한 '작은 표정'일 때 자연스럽게 낮게 나온다 — 별도 예외처리 불필요.
    """
    v_pos = sum(scores.get(k, 0.0) * w for k, w in VALENCE_POS.items()) / _VPOS_NORM
    v_neg = sum(scores.get(k, 0.0) * w for k, w in VALENCE_NEG.items()) / _VNEG_NORM
    a_pos = sum(scores.get(k, 0.0) * w for k, w in AROUSAL_POS.items()) / _APOS_NORM
    a_neg = sum(scores.get(k, 0.0) * w for k, w in AROUSAL_NEG.items()) / _ANEG_NORM

    valence = float(np.clip(v_pos - v_neg, -1.0, 1.0))
    arousal = float(np.clip(a_pos - a_neg, -1.0, 1.0))

    # 표정 강도(=신뢰도): 관여 블렌드셰입들의 활성 정도. 0.15 정도의 활성화를
    # '또렷한 표정'의 기준점으로 삼는다(ARKit 블렌드셰입 실전 경험상 흔한 임계값;
    # 실측 영상으로 보정 필요 — test_synthetic.py 결과 참고).
    # [fusion 버전 수정] 눈 깜빡임(eyeBlink)은 신뢰도 계산에서 뺀다.
    # 실제 웹캠 실행(2026-10-29)에서 무표정인데도 화면을 내려다보며 눈꺼풀이 살짝
    # 내려온 것만으로 신뢰도가 0.6~1.0까지 올라가, 융합에서 표정이 70%를 차지하고
    # 심박은 30%만 반영되는 문제가 있었다. 깜빡임은 표정의 '뚜렷함'이 아니다.
    involved = (set(VALENCE_WEIGHTS) | set(AROUSAL_WEIGHTS)) - CONFIDENCE_EXCLUDE
    intensity = max((scores.get(k, 0.0) for k in involved), default=0.0)
    confidence = float(np.clip(intensity / 0.35, 0.0, 1.0))

    return valence, arousal, confidence


def _roi_means_from_landmarks(frame_rgb: np.ndarray, landmarks) -> tuple[tuple, tuple, dict]:
    """정규화 랜드마크(0~1) 전체의 바운딩박스 비율로 이마/볼 ROI를 잡는다.

    특정 랜드마크 인덱스(예: '10번이 이마 중앙')에 의존하지 않고 바운딩박스
    비율만 쓰는 이유: 인덱스를 실제 얼굴 영상으로 검증하지 못한 상태라,
    틀린 인덱스를 쓰는 것보다 다소 거칠더라도 항상 맞는 방식을 택함.
    """
    h, w, _ = frame_rgb.shape
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bw, bh = max_x - min_x, max_y - min_y
    cx = (min_x + max_x) / 2

    def px(x_frac, y_frac):
        return int(x_frac * w), int(y_frac * h)

    # 이마: 얼굴 상단 5~22% 높이, 중앙 폭 36%
    fx1, fy1 = px(cx - 0.18 * bw, min_y + 0.05 * bh)
    fx2, fy2 = px(cx + 0.18 * bw, min_y + 0.22 * bh)
    forehead_box = (max(fx1, 0), max(fy1, 0), min(fx2, w), min(fy2, h))

    # 양쪽 볼: 얼굴 중하단부, 좌우 20%~40% 지점
    lx1, ly1 = px(min_x + 0.14 * bw, min_y + 0.55 * bh)
    lx2, ly2 = px(min_x + 0.34 * bw, min_y + 0.75 * bh)
    left_cheek_box = (max(lx1, 0), max(ly1, 0), min(lx2, w), min(ly2, h))

    rx1, ry1 = px(max_x - 0.34 * bw, min_y + 0.55 * bh)
    rx2, ry2 = px(max_x - 0.14 * bw, min_y + 0.75 * bh)
    right_cheek_box = (max(rx1, 0), max(ry1, 0), min(rx2, w), min(ry2, h))

    def mean_rgb(box):
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return (0.0, 0.0, 0.0)
        patch = frame_rgb[y1:y2, x1:x2]
        return tuple(float(c) for c in patch.reshape(-1, 3).mean(axis=0))

    forehead_rgb = mean_rgb(forehead_box)
    left_rgb = mean_rgb(left_cheek_box)
    right_rgb = mean_rgb(right_cheek_box)
    cheek_rgb = tuple((l + r) / 2 for l, r in zip(left_rgb, right_rgb))
    face_box = (max(int(min_x * w), 0), max(int(min_y * h), 0), min(int(max_x * w), w), min(int(max_y * h), h))
    boxes = {'face': face_box, 'forehead': forehead_box, 'left_cheek': left_cheek_box, 'right_cheek': right_cheek_box}
    return forehead_rgb, cheek_rgb, boxes


class FaceAnalyzer:
    """FaceLandmarker를 감싸서 프레임(BGR, OpenCV 기본)마다 FaceReading을 돌려준다."""

    def __init__(self, model_path: str, num_faces: int = 1):
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        options = FaceLandmarkerOptions(
            # macOS에서 기본값(GPU/Metal)이 'Service is unavailable'로 죽는 문제를 피하려고 CPU 고정
            base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=num_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._frame_idx = 0
        self._fs = 30.0  # detect_for_video용 타임스탬프 계산에 사용, 실제 fps로 덮어쓸 것

    def set_fps(self, fps: float) -> None:
        self._fs = fps

    def analyze_bgr(self, frame_bgr: np.ndarray) -> FaceReading:
        frame_rgb = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))

        timestamp_ms = int(self._frame_idx * (1000.0 / self._fs))
        self._frame_idx += 1

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return FaceReading(detected=False)

        landmarks = result.face_landmarks[0]
        forehead_rgb, cheek_rgb, boxes = _roi_means_from_landmarks(frame_rgb, landmarks)

        scores: dict[str, float] = {}
        if result.face_blendshapes:
            for cat in result.face_blendshapes[0]:
                scores[cat.category_name] = cat.score

        valence, arousal, confidence = blendshapes_to_affect(scores)

        return FaceReading(
            detected=True,
            forehead_rgb=forehead_rgb,
            cheek_rgb=cheek_rgb,
            valence=valence,
            arousal=arousal,
            confidence=confidence,
            blendshapes=scores,
            boxes=boxes,
        )

    def close(self) -> None:
        self._landmarker.close()
