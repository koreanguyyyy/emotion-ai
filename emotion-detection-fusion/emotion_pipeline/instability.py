"""
감정 기복(affective instability) 지표 — 교수님 원질문("감정 기복을 어떻게
확인하냐")에 대한 최종 출력단. 적용방안 문서 5절의 MSSD를 시계열 트래커로 구현.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def mssd(x: np.ndarray) -> float:
    """Mean Square Successive Difference — 연속값이 얼마나 자주/급격히 바뀌는지."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.mean(np.diff(x) ** 2))


def lag1_autocorr(x: np.ndarray) -> float:
    """1차 자기상관 — 감정이 얼마나 오래 지속되는지(관성). MSSD와 상보적."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or np.std(x) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


@dataclass
class InstabilityReport:
    n: int
    valence_mssd: float
    arousal_mssd: float
    valence_sd: float
    arousal_sd: float
    valence_autocorr: float


class InstabilityTracker:
    """(timestamp, valence, arousal) 스트림을 모아뒀다가 요청 시 기복 리포트를 낸다."""

    def __init__(self, maxlen: int | None = None):
        self._t: deque = deque(maxlen=maxlen)
        self._v: deque = deque(maxlen=maxlen)
        self._a: deque = deque(maxlen=maxlen)

    def add(self, timestamp: float, valence: float, arousal: float) -> None:
        self._t.append(timestamp)
        self._v.append(valence)
        self._a.append(arousal)

    def report(self) -> InstabilityReport:
        v, a = np.array(self._v), np.array(self._a)
        return InstabilityReport(
            n=len(v),
            valence_mssd=mssd(v),
            arousal_mssd=mssd(a),
            valence_sd=float(np.std(v)) if len(v) else 0.0,
            arousal_sd=float(np.std(a)) if len(a) else 0.0,
            valence_autocorr=lag1_autocorr(v),
        )

    def series(self) -> tuple[list, list, list]:
        return list(self._t), list(self._v), list(self._a)
