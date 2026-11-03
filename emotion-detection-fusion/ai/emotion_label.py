# ai/emotion_label.py
"""
블렌드셰입 → 한국어 감정 라벨 (행복/슬픔/분노/공포/혐오/놀람/무감정).

에이전트 서버(emotion-ai-server)는 face_emotion 라벨로 추천 Tool(음악/장소/책)과
검색 키워드를 고른다. 1학기에는 DeepFace가 이 라벨을 줬는데, 이 버전은 DeepFace를
쓰지 않으므로 MediaPipe 블렌드셰입에서 라벨을 만든다.

⚠️ 이 라벨은 '추천 종류 선택용' 휴리스틱이다. 스트레스 지수 계산에는 쓰지 않는다
(스트레스는 ai/stress_index.py가 정서가·각성도·심박으로 따로 계산).
가중치와 기준값은 초기값이라 실제 영상으로 보정이 필요하다.
"""
from collections import Counter

NEUTRAL_LABEL = "무감정"
MIN_SCORE = 0.2   # 가장 강한 감정 점수가 이보다 작으면 무감정

EN_LABEL = {  # 화면 표시용 (cv2.putText는 한글을 못 그림)
    "행복": "happy", "슬픔": "sad", "분노": "angry", "공포": "fear",
    "혐오": "disgust", "놀람": "surprise", "무감정": "neutral",
}


def _avg(bs: dict, *names: str) -> float:
    return sum(bs.get(n, 0.0) for n in names) / len(names)


def emotion_scores(bs: dict) -> dict[str, float]:
    """블렌드셰입 점수(0~1) → 감정별 점수(0~1). FACS 표정 단위를 근사한 조합."""
    smile = _avg(bs, "mouthSmileLeft", "mouthSmileRight")
    frown = _avg(bs, "mouthFrownLeft", "mouthFrownRight")
    brow_down = _avg(bs, "browDownLeft", "browDownRight")
    brow_inner = bs.get("browInnerUp", 0.0)
    brow_outer = _avg(bs, "browOuterUpLeft", "browOuterUpRight")
    eye_wide = _avg(bs, "eyeWideLeft", "eyeWideRight")
    sneer = _avg(bs, "noseSneerLeft", "noseSneerRight")
    upper_lip = _avg(bs, "mouthUpperUpLeft", "mouthUpperUpRight")
    press = _avg(bs, "mouthPressLeft", "mouthPressRight")
    stretch = _avg(bs, "mouthStretchLeft", "mouthStretchRight")
    jaw = bs.get("jawOpen", 0.0)
    chin_raise = bs.get("mouthShrugLower", 0.0)   # 턱을 올려 아랫입술이 밀려 올라감 (FACS AU17)

    # 슬픔 (2026-10-30 보강): 입꼬리 내리기(AU15) 0.4 + 눈썹 안쪽 올리기(AU1) 0.4 + 턱 올리기(AU17) 0.2.
    # 실제 기록에서 턱 올리기가 0.6~0.84로 자주 나왔는데 이전 계산은 이 동작을 보지 않아서, 입꼬리를
    # 과장되게 내려야만 슬픔이 잡혔다. 턱 올리기는 생각할 때·입을 다물 때도 나오는 동작이라,
    # 입꼬리나 눈썹 동작이 함께 있을 때만 그 두 배까지 센다 (턱 올리기 단독으로는 슬픔이 아님).
    chin_part = min(chin_raise, 2.0 * (frown + brow_inner))

    return {
        "행복": smile,
        "슬픔": 0.4 * frown + 0.4 * brow_inner + 0.2 * chin_part,
        "분노": 0.7 * brow_down + 0.3 * press,
        "공포": (eye_wide + brow_inner + stretch) / 3,
        "혐오": 0.7 * sneer + 0.3 * upper_lip,
        "놀람": (eye_wide + jaw + brow_outer) / 3,
    }


def label_from_blendshapes(bs: dict, detected: bool = True) -> str:
    if not detected or not bs:
        return NEUTRAL_LABEL
    scores = emotion_scores(bs)
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] >= MIN_SCORE else NEUTRAL_LABEL


def dominant_label(labels: list[str]) -> str:
    """구간 안에서 가장 자주 나온 라벨. 무감정만 있으면 무감정."""
    if not labels:
        return NEUTRAL_LABEL
    non_neutral = [l for l in labels if l != NEUTRAL_LABEL]
    # 구간의 1/4 이상이 특정 감정이면 그 감정을 대표로 본다 (짧은 순간 표정도 반영)
    if non_neutral:
        label, count = Counter(non_neutral).most_common(1)[0]
        if count >= len(labels) / 4:
            return label
    return NEUTRAL_LABEL
