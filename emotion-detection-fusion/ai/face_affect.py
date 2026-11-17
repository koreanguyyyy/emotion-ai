# ai/face_affect.py
"""
개인별 무표정 기준선 + 표정 → 정서가·각성도·신뢰도·라벨 (fusion 버전).

2학기 roi.blendshapes_to_affect를 대신한다. 2026-10-30 실제 웹캠 실행에서 확인한 문제:
  1) 무표정인데도 눈 찡그림(eyeSquint 0.4~0.5), 시선 아래, 눈썹 올림 같은 값이 원래 켜져 있어서
     표정 신뢰도가 자주 1.0으로 찼다. 사람마다 '무표정'의 블렌드셰입 값이 다르다.
  2) 실제로 지은 부정 표정은 눈썹 안쪽 올리기(browInnerUp 0.68)였는데, 2학기 정서가 계산에는
     이 동작이 없어서 정서가가 최저 −0.04에 그쳤다. 반면 화면 라벨은 '슬픔'이라 라벨과 숫자가 어긋났다.

대응
  1) FaceBaseline: 처음 calib_sec(기본 60초) 동안 얼굴이 잡힌 프레임의 블렌드셰입 중앙값을
     '이 사람의 무표정'으로 기록하고, 이후에는 평소보다 '올라간 만큼'(delta)만 표정으로 읽는다.
  2) 정서가를 감정 라벨과 같은 계산(ai/emotion_label.emotion_scores)에서 만든다.
     정서가 = (행복 − 가장 강한 부정 감정[슬픔·분노·공포·혐오]) / VALENCE_SCALE
     → 눈썹 안쪽 올리기·입꼬리 내리기(슬픔), 미간 찌푸리기(분노) 등이 모두 정서가에 반영되고,
       라벨과 정서가가 같은 방향을 가리킨다.
⚠️ 기준값(VALENCE_SCALE, AROUSAL_SCALE, CONF_REF)은 초기값이다.
   입술 내밀기(mouthPucker)는 기본 감정 표정의 구성 요소가 아니라서(생각할 때·휘파람 등) 넣지 않았다.
"""
from __future__ import annotations

import os
import statistics
from dataclasses import dataclass

import numpy as np

from ai.emotion_label import MIN_SCORE, NEUTRAL_LABEL, emotion_scores

VALENCE_SCALE = 0.5   # 감정 점수 차이 0.5 → 정서가 ±1 (예: 눈썹 안쪽 +0.5 → 슬픔 0.2 → 정서가 −0.4)
AROUSAL_SCALE = 0.3
CONF_REF = 0.35       # 평소보다 0.35 이상 움직이면 '뚜렷한 표정' (2학기와 같은 기준점)
def _label_min() -> float:
    try:
        return float((os.getenv("FACE_LABEL_MIN") or "0.15").split("#")[0])
    except ValueError:
        return 0.15


LABEL_MIN = _label_min()   # 기준선 대비 감정 점수가 이 이상이면 라벨 표시(기본 0.15). .env FACE_LABEL_MIN으로 조절 — 낮을수록 작은 표정도 잡음
                      # (2학기 원래 값 0.2는 원값 기준이라, 정서가 −0.4인데 라벨은 무감정인 경우가 생겼다)

AROUSAL_POS = {
    "eyeWideLeft": 0.8, "eyeWideRight": 0.8, "jawOpen": 0.7, "browInnerUp": 0.5,
    "browOuterUpLeft": 0.4, "browOuterUpRight": 0.4, "mouthStretchLeft": 0.4, "mouthStretchRight": 0.4,
}
AROUSAL_NEG = {"eyeBlinkLeft": 0.3, "eyeBlinkRight": 0.3}
_APOS, _ANEG = sum(AROUSAL_POS.values()), sum(AROUSAL_NEG.values())

# 신뢰도(표정이 뚜렷한 정도)에 쓰는 블렌드셰입: 감정 점수와 각성도에 들어가는 동작. 눈 깜빡임·시선은 제외.
CONF_KEYS = {
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "eyeWideLeft", "eyeWideRight", "noseSneerLeft", "noseSneerRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "mouthPressLeft", "mouthPressRight",
    "mouthStretchLeft", "mouthStretchRight", "jawOpen",
}


@dataclass
class FaceAffect:
    valence: float      # −1~1 (아직 신뢰도를 곱하지 않은 값)
    arousal: float      # −1~1
    confidence: float   # 0~1
    label: str          # 한국어 감정 라벨
    top_delta: str      # 진단용: 평소보다 가장 많이 올라간 블렌드셰입 3개
    scores: dict | None = None   # 감정별 점수(평소 대비, 0 이상) — 웹 화면의 감정 막대용


class FaceBaseline:
    """개인별 무표정 기준선. calib_sec 동안 얼굴이 잡힌 프레임을 모아 블렌드셰입별 중앙값을 낸다."""

    def __init__(self, calib_sec: float = 60.0, min_frames: int = 150):
        self.calib_sec = calib_sec
        self.min_frames = min_frames
        self._samples: dict[str, list[float]] = {}
        self._n = 0
        self._start: float | None = None
        self.values: dict[str, float] | None = None

    @property
    def ready(self) -> bool:
        return self.values is not None

    def add(self, t: float, bs: dict) -> None:
        if self.ready or not bs:
            return
        if self._start is None:
            self._start = t
        for k, v in bs.items():
            self._samples.setdefault(k, []).append(float(v))
        self._n += 1
        if t - self._start >= self.calib_sec and self._n >= self.min_frames:
            self.values = {k: statistics.median(v) for k, v in self._samples.items()}
            self._samples = {}

    def progress(self, t: float) -> float:
        if self.ready:
            return 1.0
        return 0.0 if self._start is None else min(1.0, (t - self._start) / self.calib_sec)

    def delta(self, bs: dict) -> dict:
        """평소보다 올라간 만큼만 (내려간 것은 0). 기준선이 없으면 원래 값 그대로."""
        if not self.ready:
            return dict(bs)
        base = self.values or {}
        return {k: max(0.0, float(v) - base.get(k, 0.0)) for k, v in bs.items()}


def face_affect(bs: dict, baseline: FaceBaseline | None = None, detected: bool = True) -> FaceAffect:
    if not detected or not bs:
        return FaceAffect(0.0, 0.0, 0.0, NEUTRAL_LABEL, "")
    d = baseline.delta(bs) if baseline is not None else dict(bs)

    sc = emotion_scores(d)
    neg = max(sc["슬픔"], sc["분노"], sc["공포"], sc["혐오"])
    valence = float(np.clip((sc["행복"] - neg) / VALENCE_SCALE, -1.0, 1.0))

    a_pos = sum(d.get(k, 0.0) * w for k, w in AROUSAL_POS.items()) / _APOS
    a_neg = sum(d.get(k, 0.0) * w for k, w in AROUSAL_NEG.items()) / _ANEG
    arousal = float(np.clip((a_pos - a_neg) / AROUSAL_SCALE, -1.0, 1.0))

    intensity = max((d.get(k, 0.0) for k in CONF_KEYS), default=0.0)
    confidence = float(np.clip(intensity / CONF_REF, 0.0, 1.0))

    best = max(sc, key=lambda k: sc[k])
    threshold = LABEL_MIN if (baseline is not None and baseline.ready) else MIN_SCORE
    label = best if sc[best] >= threshold else NEUTRAL_LABEL

    top = sorted(((k, v) for k, v in d.items() if k != "_neutral"), key=lambda kv: kv[1], reverse=True)[:3]
    return FaceAffect(valence, arousal, confidence, label, " ".join(f"{k}:{v:.2f}" for k, v in top),
                      {k: max(0.0, float(v)) for k, v in sc.items()})
