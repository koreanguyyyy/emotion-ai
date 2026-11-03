"""
rPPG(원격 광용적맥파) — CHROM 알고리즘 기반 스트리밍 맥박 추출기.

지난 데모(rppg_demo.py)에서 검증한 CHROM 수식을 '한 프레임씩 들어오는' 실시간
상황에 맞게 버퍼(deque) 기반 스트리밍 클래스로 재구성한 것.

중요한 제약 (적용방안 문서 6절 참고, 실제 문헌으로 확인한 내용):
- 심박수(HR) 추정에는 사용 가능 (오차 약 5bpm 수준).
- HRV(RMSSD 등)는 웹캠 rPPG로 신뢰할 수 없음 (실험실 조건에서도 r≈0.45) →
  이 모듈은 HR과 '심박수 추세'까지만 책임지고, HRV는 의도적으로 계산하지 않는다.
  안정시 HRV(rMSSD)는 접촉 센서로 별도 측정해야 한다.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass(sig: np.ndarray, fs: float, low: float = 0.7, high: float = 3.0, order: int = 3) -> np.ndarray:
    """사람 심박 가능 범위(42~180bpm = 0.7~3.0Hz)만 통과시켜 잡음 제거."""
    nyq = fs / 2
    low_n, high_n = low / nyq, high / nyq
    # 버퍼가 아주 짧으면(필터 차수 대비) filtfilt가 에러를 내므로 방어
    if len(sig) <= order * 3:
        return sig - sig.mean()
    # scipy에는 공식 타입 스텁(py.typed)이 없어서, Pylance/Pyright가 butter()의
    # '실제 구현 코드'를 읽어 반환 타입을 추론한다(useLibraryCodeForTypes). 구현
    # 내부에 output 값에 따라 2튜플/3튜플/배열을 각각 return하는 분기가 있는데,
    # 이게 literal 기반 @overload로 선언돼 있지 않아서 output="ba"를 호출부에
    # 넘겨도 추론 결과가 좁혀지지 않는다 — 즉 output="ba"만으로는 해결 안 됨
    # (실제로 pyright 1.1.409로 재현해서 확인함). 런타임 타입은 항상 2튜플로
    # 확정돼 있으므로, cast()로 타입 체커에게 직접 알려주는 게 맞는 해법이다.
    b, a = cast(tuple[np.ndarray, np.ndarray],
                butter(order, [low_n, high_n], btype="band", output="ba"))
    return filtfilt(b, a, sig)


@dataclass
class HREstimate:
    bpm: float
    quality: float  # 0~1, 신호대잡음비 기반 — 낮으면 이 추정치를 버려야 함


def chrom_pulse(R: np.ndarray, G: np.ndarray, B: np.ndarray, fs: float) -> np.ndarray:
    """CHROM(De Haan & Jeanne, 2013) 맥파 신호 복원. R/G/B는 ROI 평균 색상 시계열."""
    Rn, Gn, Bn = R / R.mean(), G / G.mean(), B / B.mean()
    Xs = 3 * Rn - 2 * Gn
    Ys = 1.5 * Rn + Gn - 1.5 * Bn
    Xf, Yf = bandpass(Xs, fs), bandpass(Ys, fs)
    alpha = Xf.std() / (Yf.std() + 1e-8)
    return Xf - alpha * Yf


def estimate_hr(pulse_signal: np.ndarray, fs: float) -> HREstimate:
    """맥파 신호 → FFT 피크 → bpm. quality는 피크 대역이 전체 스펙트럼에서 차지하는
    비중으로 근사 — 또렷한 단일 피크일수록 1에 가깝다 (SQI, 신호품질지표의 단순화 버전)."""
    n = len(pulse_signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    power = np.abs(np.fft.rfft(pulse_signal)) ** 2

    band = (freqs >= 0.7) & (freqs <= 3.0)
    if not band.any() or power[band].sum() == 0:
        return HREstimate(bpm=float("nan"), quality=0.0)

    band_power = power[band]
    peak_idx = np.argmax(band_power)
    peak_freq = freqs[band][peak_idx]

    # 품질: 피크 주변 ±0.1Hz에 에너지가 얼마나 집중되어 있는지 (급 좁을수록 신뢰도 높음)
    peak_window = (freqs[band] >= peak_freq - 0.1) & (freqs[band] <= peak_freq + 0.1)
    quality = float(band_power[peak_window].sum() / (band_power.sum() + 1e-8))

    return HREstimate(bpm=float(peak_freq * 60), quality=quality)


class CHROMExtractor:
    """프레임마다 ROI 평균 RGB를 넣으면, 윈도우가 찰 때마다 HR을 추정하는 스트리밍 래퍼.

    window_sec: HR 추정에 쓰는 버퍼 길이. 60초면 충분하지만 반응성을 위해
                기본값은 조금 짧게(20초) 두고, 필요하면 늘려 쓴다.
                (HRV가 아니라 HR만 뽑으므로 120초씩 기다릴 필요는 없다 — 6절 참고)
    """

    def __init__(self, fs: float, window_sec: float = 20.0, step_sec: float = 1.0):
        self.fs = fs
        self.window_n = int(window_sec * fs)
        self.step_n = max(1, int(step_sec * fs))
        self._r = deque(maxlen=self.window_n)
        self._g = deque(maxlen=self.window_n)
        self._b = deque(maxlen=self.window_n)
        self._frames_since_estimate = 0
        self._last: HREstimate | None = None

    def update(self, rgb_mean: tuple[float, float, float]) -> HREstimate | None:
        """새 프레임의 ROI 평균 RGB를 추가. 버퍼가 다 찼고 step 주기가 되면 새 추정치 반환."""
        r, g, b = rgb_mean
        self._r.append(r)
        self._g.append(g)
        self._b.append(b)
        self._frames_since_estimate += 1

        if len(self._r) < self.window_n:
            return None  # 아직 윈도우가 안 참
        if self._frames_since_estimate < self.step_n:
            return self._last  # 아직 갱신 주기 아님 — 직전 추정치 유지

        self._frames_since_estimate = 0
        pulse = chrom_pulse(np.array(self._r), np.array(self._g), np.array(self._b), self.fs)
        self._last = estimate_hr(pulse, self.fs)
        return self._last
