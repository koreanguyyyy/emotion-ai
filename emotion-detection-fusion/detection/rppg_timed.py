# detection/rppg_timed.py
"""
시각 기반 rPPG — 2학기 CHROMExtractor를 실제 웹캠 환경에 맞게 보강한 버전.

2026-10-29 실제 웹캠 실행에서 확인한 문제와 대응
  1) fps: 맥북 카메라는 "15fps"라고 알려주지만 실제로는 30fps로 프레임을 보낸다
     (10초 측정, 중복 프레임 0개). fps를 틀리게 넣으면 심박이 그 비율만큼 어긋난다.
     → 프레임마다 실제 시각(time.monotonic)을 받아, 일정 간격(TARGET_FS)으로
       선형 보간(리샘플링)한 뒤 계산한다. 카메라 표기값이나 처음 몇 초의 측정에
       기대지 않고, 처리 속도가 중간에 흔들려도 맞는 시간축을 쓴다.
  2) 신호가 약함 (품질 0.1~0.3): 이마만 쓰던 측정 영역을 이마 + 양 볼 평균으로 넓혀
     영역별로 독립적인 잡음을 줄인다.
  3) CHROM 하나만 사용 → POS(Wang et al., 2017)를 함께 계산하고, 품질이 높은 쪽을 쓴다.
     두 방식 중 최고값을 고르면 잡음도 점수가 올라가므로, 품질 기준은 이 선택 방식으로
     잡음만 넣었을 때의 통과율로 다시 정했다 (tests/test_rppg_timed.py).

이 모듈도 심박수(HR)까지만 책임진다. HRV는 웹캠으로 신뢰할 수 없어 계산하지 않는다.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from emotion_pipeline.rppg import bandpass, chrom_pulse, estimate_hr

TARGET_FS = 30.0   # 리샘플링 간격 (Hz)


def pos_pulse(R: np.ndarray, G: np.ndarray, B: np.ndarray, fs: float, win_sec: float = 1.6) -> np.ndarray:
    """POS — Plane-Orthogonal-to-Skin (Wang, den Brinker, Stuijk, de Haan, 2017).
    짧은 창(약 1.6초)마다 색을 정규화하고 피부색에 직교하는 평면으로 투영한 뒤
    겹쳐 더한다(overlap-add). 조명 세기 변화에 CHROM보다 강건하다고 보고된 방식."""
    C = np.vstack([R, G, B])                     # 3 x N
    n = C.shape[1]
    w = max(2, int(round(win_sec * fs)))
    H = np.zeros(n)
    P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    for start in range(0, n - w + 1):
        seg = C[:, start:start + w]
        mean = seg.mean(axis=1, keepdims=True)
        mean[mean == 0] = 1e-8
        Cn = seg / mean
        S = P @ Cn                               # 2 x w
        s1, s2 = S[0], S[1]
        h = s1 + (s1.std() / (s2.std() + 1e-8)) * s2
        H[start:start + w] += h - h.mean()
    return bandpass(H, fs)


@dataclass
class TimedHR:
    bpm: float                 # 채택된 심박 (nan 가능)
    quality: float             # 채택된 방식의 품질
    method: str                # "chrom" / "pos"
    chrom_bpm: float
    chrom_quality: float
    pos_bpm: float
    pos_quality: float
    fps_actual: float          # 이 윈도우의 실제 평균 프레임 속도
    # 진단용 (그래프 저장에 사용)
    t_uniform: np.ndarray = field(default=None, repr=False)        # type: ignore[assignment]
    rgb_uniform: np.ndarray = field(default=None, repr=False)      # type: ignore[assignment]
    chrom_signal: np.ndarray = field(default=None, repr=False)     # type: ignore[assignment]
    pos_signal: np.ndarray = field(default=None, repr=False)       # type: ignore[assignment]


def _combine(forehead_rgb, cheek_rgb):
    """이마와 양 볼(평균) 색을 같은 비중으로 합친다. 하나가 없으면 있는 쪽만 쓴다."""
    parts = [np.asarray(x, dtype=float) for x in (forehead_rgb, cheek_rgb)
             if x is not None and np.all(np.isfinite(x)) and np.sum(x) > 0]
    if not parts:
        return None
    return np.mean(parts, axis=0)


def resample_uniform(ts: np.ndarray, values: np.ndarray, fs: float, duration: float):
    """불규칙한 시각 ts의 값을 마지막 duration초 구간에서 1/fs 간격으로 선형 보간한다.
    values: N x 3"""
    t_end = ts[-1]
    t_start = t_end - duration
    t_u = np.arange(t_start, t_end, 1.0 / fs)
    out = np.column_stack([np.interp(t_u, ts, values[:, c]) for c in range(values.shape[1])])
    return t_u, out


class TimedRPPG:
    """프레임마다 update(시각, 이마RGB, 볼RGB)를 부르면, 약 step_sec마다 새 심박 추정을 돌려준다.

    window_sec: 심박 계산에 쓰는 최근 구간 길이 (기본 20초)
    step_sec  : 추정 주기
    min_fill  : 윈도우의 이 비율 이상이 쌓여야 계산 시작
    max_gap   : 프레임 사이 간격이 이보다 길면(얼굴 놓침 등) 버퍼를 비우고 다시 시작
    """

    def __init__(self, window_sec: float = 20.0, step_sec: float = 1.0, fs: float = TARGET_FS,
                 min_fill: float = 0.95, max_gap: float = 0.5):
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.fs = fs
        self.min_fill = min_fill
        self.max_gap = max_gap
        self._t: deque = deque()
        self._rgb: deque = deque()
        self._last_estimate_t = -1e9
        self.last: TimedHR | None = None

    def reset(self) -> None:
        self._t.clear(); self._rgb.clear()

    def buffered_sec(self) -> float:
        return (self._t[-1] - self._t[0]) if len(self._t) >= 2 else 0.0

    def update(self, t: float, forehead_rgb, cheek_rgb=None) -> TimedHR | None:
        """새 추정을 했으면 TimedHR, 아니면 None."""
        rgb = _combine(forehead_rgb, cheek_rgb)
        if rgb is None:
            return None
        if self._t and t - self._t[-1] > self.max_gap:
            self.reset()                       # 얼굴을 놓쳤다가 다시 잡은 경우: 끊긴 신호를 잇지 않는다
        if self._t and t <= self._t[-1]:
            return None                        # 시각이 거꾸로 가거나 같으면 무시
        self._t.append(t); self._rgb.append(rgb)
        while self._t and t - self._t[0] > self.window_sec + 1.0:
            self._t.popleft(); self._rgb.popleft()

        if self.buffered_sec() < self.window_sec * self.min_fill:
            return None
        if t - self._last_estimate_t < self.step_sec:
            return None
        self._last_estimate_t = t
        self.last = self._estimate()
        return self.last

    def _estimate(self) -> TimedHR:
        ts = np.fromiter(self._t, dtype=float)
        vals = np.array(self._rgb, dtype=float)
        duration = min(self.window_sec, ts[-1] - ts[0])
        n_in = int(np.sum(ts >= ts[-1] - duration))
        fps_actual = (n_in - 1) / duration if duration > 0 else float("nan")

        t_u, rgb_u = resample_uniform(ts, vals, self.fs, duration)
        R, G, B = rgb_u[:, 0], rgb_u[:, 1], rgb_u[:, 2]
        chrom = chrom_pulse(R, G, B, self.fs)
        pos = pos_pulse(R, G, B, self.fs)
        ec = estimate_hr(chrom, self.fs)
        ep = estimate_hr(pos, self.fs)
        best, method = (ec, "chrom") if ec.quality >= ep.quality else (ep, "pos")
        return TimedHR(
            bpm=best.bpm, quality=best.quality, method=method,
            chrom_bpm=ec.bpm, chrom_quality=ec.quality, pos_bpm=ep.bpm, pos_quality=ep.quality,
            fps_actual=fps_actual, t_uniform=t_u, rgb_uniform=rgb_u,
            chrom_signal=chrom, pos_signal=pos,
        )
