# ai/stress_index.py
"""
로컬 스트레스 지수 (0~10) — 1학기의 'DeepFace 확률 → VAD 표' 계산을 대체한다.

지수 = 10 × ( W_NEG    × 부정 정서 정도      ← 표정 (신뢰도 반영 정서가)
            + W_HR     × 심박 상승 정도      ← 웹캠 심박, 개인 평상시 대비
            + W_ARO    × 표정 각성 정도      ← 표정 (눈 크게 뜸, 입 벌림 등)
            + W_INSTAB × 감정 기복 정도 )    ← 최근 1분 MSSD

  부정 정서 정도 = clip( -최근 10초 정서가 평균 / VALENCE_REF , 0, 1 )
  심박 상승 정도 = clip(  최근 10초 (심박 − 평상시) 평균 / HR_REF_BPM , 0, 1 )   (+20bpm → 1.0)
  표정 각성 정도 = clip(  최근 10초 표정 각성도 평균 / AROUSAL_REF , 0, 1 )
  기복 정도      = clip(  최근 1분 MSSD(1초 평균) / MSSD_REF , 0, 1 )

2026-10-30 변경: 심박 상승을 별도 항으로 넣었다(제안서 4장 마의 S = w1·z_HR + w2·A + w3·M 구조).
  이전에는 심박이 '융합 각성도' 안에만 들어 있었는데, 적응형 융합은 표정이 뚜렷할수록 각성도의 70%를
  표정에서 가져오기 때문에, 찡그린 표정 + 심박 +20bpm이 5.1점으로 개입 기준(6) 미만이었다.
  이제 심박은 표정과 상관없이 그대로 반영된다. (적응형 융합 자체는 화면·기록용 융합 정서가·각성도에 그대로 쓴다.)

설계 의도 (기본 가중치 0.35 · 0.35 · 0.10 · 0.20):
  - 부정 표정만, 또는 심박 상승만으로는 3.5점 → 개입 기준(6) 미만
  - 부정 표정 + 심박 상승이 함께면 7점 이상 → 개입
  - 웃으면서 심박이 오르는 경우(신남, 운동 직후)는 부정 정서가 0이라 개입하지 않는다
  - 심박을 모르는 구간(보정 실패, 끊김)은 심박 항을 빼고 나머지 가중치로 다시 나눈다(재정규화).
    이 경우 부정 표정만으로 약 5.4점이라, 강한 표정에 각성이나 기복이 더해져야 개입한다.
⚠️ 모든 가중치·기준값은 초기값이다. .env로 바꿀 수 있고, 실제 데이터로 보정해야 한다.
"""
from __future__ import annotations

import os
import statistics
import threading
from collections import deque
from dataclasses import dataclass, asdict

import numpy as np

from ai.emotion_label import dominant_label


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value.split("#")[0].strip())


@dataclass
class StressConfig:
    w_neg: float = 0.35
    w_hr: float = 0.35
    w_aro: float = 0.10
    w_instab: float = 0.20
    hr_ref_bpm: float = 20.0     # 평상시보다 +20bpm이면 심박 상승 정도 1.0
    min_hr_ratio: float = 0.5    # 구간 중 심박을 아는 프레임 비율이 이보다 낮으면 심박 항을 빼고 재정규화
    valence_ref: float = 0.4     # 정서가 평균이 -0.4면 부정 정서 정도 1.0
    arousal_ref: float = 0.5     # 표정 각성도 평균이 +0.5면 표정 각성 정도 1.0
    mssd_ref: float = 0.05       # 1초 평균 시계열의 MSSD가 0.05면 기복 정도 1.0
    window_sec: float = 10.0     # 정서가·각성도 평균을 내는 최근 구간
    instab_window_sec: float = 60.0
    min_face_ratio: float = 0.5  # 구간 중 얼굴이 잡힌 프레임 비율이 이보다 낮으면 계산 보류

    @classmethod
    def from_env(cls) -> "StressConfig":
        d = cls()
        return cls(
            w_neg=_env_float("STRESS_W_NEG", d.w_neg),
            w_hr=_env_float("STRESS_W_HR", d.w_hr),
            w_aro=_env_float("STRESS_W_AROUSAL", d.w_aro),
            hr_ref_bpm=_env_float("STRESS_HR_REF_BPM", d.hr_ref_bpm),
            min_hr_ratio=_env_float("STRESS_MIN_HR_RATIO", d.min_hr_ratio),
            w_instab=_env_float("STRESS_W_INSTAB", d.w_instab),
            valence_ref=_env_float("STRESS_VALENCE_REF", d.valence_ref),
            arousal_ref=_env_float("STRESS_AROUSAL_REF", d.arousal_ref),
            mssd_ref=_env_float("STRESS_MSSD_REF", d.mssd_ref),
            window_sec=_env_float("STRESS_WINDOW_SEC", d.window_sec),
            instab_window_sec=_env_float("STRESS_INSTAB_WINDOW_SEC", d.instab_window_sec),
            min_face_ratio=_env_float("STRESS_MIN_FACE_RATIO", d.min_face_ratio),
        )


@dataclass
class StressReading:
    index: float          # 0~10
    neg: float            # 부정 정서 정도 0~1
    hr_up: float | None   # 심박 상승 정도 0~1 (심박을 모르면 None → 재정규화)
    arousal: float        # 표정 각성 정도 0~1
    instab: float         # 기복 정도 0~1
    valence_mean: float
    arousal_mean: float
    mssd: float
    face_confidence: float
    label: str            # 구간 대표 감정 라벨 (한국어)
    n_frames: int

    def to_dict(self) -> dict:
        return asdict(self)


def mssd(x) -> float:
    """Mean Square Successive Difference (emotion_pipeline.instability.mssd와 같은 식).
    mediapipe를 불러오지 않고 테스트할 수 있게 여기에 한 줄로 둔다."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(np.diff(x) ** 2)) if len(x) >= 2 else 0.0


def _per_second_means(ts, vs, as_):
    """프레임별 값을 1초 단위 평균으로 묶는다. 프레임 단위 MSSD는 잡음에 지배되므로
    1초 평균 시계열에서 기복을 잰다."""
    buckets: dict[int, list] = {}
    for t, v, a in zip(ts, vs, as_):
        buckets.setdefault(int(t), []).append((v, a))
    keys = sorted(buckets)
    v_sec = [float(np.mean([p[0] for p in buckets[k]])) for k in keys]
    a_sec = [float(np.mean([p[1] for p in buckets[k]])) for k in keys]
    return v_sec, a_sec


class StressIndexer:
    """카메라 스레드가 프레임마다 add()하고, 판단 스레드가 compute()로 지수를 읽는다."""

    def __init__(self, config: StressConfig | None = None, max_sec: float = 120.0):
        self.cfg = config or StressConfig()
        self._lock = threading.Lock()
        self._max_sec = max(max_sec, self.cfg.instab_window_sec)
        # (t, detected, valence, arousal, confidence, label, hr_delta)
        self._buf: deque = deque()

    def add(self, t: float, detected: bool, valence: float, arousal: float,
            confidence: float, label: str, hr_delta: float | None = None) -> None:
        """valence: 신뢰도를 반영한 표정 정서가, arousal: 표정 각성도,
        hr_delta: 현재 심박 − 평상시 심박 (bpm). 심박을 모르면 None."""
        with self._lock:
            self._buf.append((t, detected, valence, arousal, confidence, label, hr_delta))
            while self._buf and t - self._buf[0][0] > self._max_sec:
                self._buf.popleft()

    def clear(self) -> None:
        """새 관람객: 앞사람의 기록을 지운다."""
        with self._lock:
            self._buf.clear()

    def compute(self, now: float | None = None) -> StressReading | None:
        with self._lock:
            buf = list(self._buf)
        if not buf:
            return None
        now = buf[-1][0] if now is None else now
        cfg = self.cfg

        recent = [r for r in buf if now - r[0] <= cfg.window_sec]
        if len(recent) < 5:
            return None
        faces = [r for r in recent if r[1]]
        if len(faces) / len(recent) < cfg.min_face_ratio:
            return None   # 얼굴이 대부분 안 잡힌 구간은 판단하지 않는다

        v_mean = float(np.mean([r[2] for r in faces]))
        a_mean = float(np.mean([r[3] for r in faces]))
        conf_mean = float(np.mean([r[4] for r in faces]))

        long_win = [r for r in buf if r[1] and now - r[0] <= cfg.instab_window_sec]
        v_sec, a_sec = _per_second_means([r[0] for r in long_win],
                                         [r[2] for r in long_win],
                                         [r[3] for r in long_win])
        m = (mssd(v_sec) + mssd(a_sec)) / 2 if len(v_sec) >= 3 else 0.0

        neg = float(np.clip(-v_mean / cfg.valence_ref, 0.0, 1.0))
        aro = float(np.clip(a_mean / cfg.arousal_ref, 0.0, 1.0))
        instab = float(np.clip(m / cfg.mssd_ref, 0.0, 1.0))

        hr_vals = [r[6] for r in faces if r[6] is not None]
        if len(hr_vals) / len(faces) >= cfg.min_hr_ratio:
            hr_up: float | None = float(np.clip(np.mean(hr_vals) / cfg.hr_ref_bpm, 0.0, 1.0))
            index = 10.0 * (cfg.w_neg * neg + cfg.w_hr * hr_up + cfg.w_aro * aro + cfg.w_instab * instab)
        else:
            hr_up = None   # 심박을 모름 → 심박 항을 빼고 나머지 가중치로 다시 나눔
            denom = cfg.w_neg + cfg.w_aro + cfg.w_instab
            index = 10.0 * (cfg.w_neg * neg + cfg.w_aro * aro + cfg.w_instab * instab) / denom

        return StressReading(
            index=round(float(np.clip(index, 0.0, 10.0)), 2),
            neg=round(neg, 3), hr_up=None if hr_up is None else round(hr_up, 3),
            arousal=round(aro, 3), instab=round(instab, 3),
            valence_mean=round(v_mean, 3), arousal_mean=round(a_mean, 3),
            mssd=round(m, 4), face_confidence=round(conf_mean, 3),
            label=dominant_label([r[5] for r in faces]),
            n_frames=len(recent),
        )


class BaselineHR:
    """개인 평상시 심박수(기준선).

    심박 상승을 '평상시 대비'로 봐야 하는데, 사람마다 평상시 심박이 50~90bpm으로
    다르다. 그래서 실행 직후 calib_sec 동안 품질 기준을 통과한 심박값의 중앙값을
    기준선으로 잡는다. RESTING_HR를 .env에 적으면 보정 없이 그 값을 쓴다.
    보정이 끝나기 전에는 ready=False → 심박을 각성도에 넣지 않고, 개입도 하지 않는다.

    timeout_sec(실행 시작 기준) 안에 보정이 끝나지 않으면 status()가 "unavailable"을
    돌려준다 → 에이전트는 표정만으로 동작한다. 그 뒤에도 표본은 계속 모으므로,
    나중에 심박이 잡혀 보정이 끝나면 자동으로 융합 모드로 돌아간다.
    """

    def __init__(self, calib_sec: float = 60.0, fixed_bpm: float | None = None,
                 min_samples: int = 10, timeout_sec: float = 120.0):
        self.calib_sec = calib_sec
        self.timeout_sec = timeout_sec
        self.min_samples = min_samples
        self._samples: list[float] = []
        self._start: float | None = None
        self.value: float | None = fixed_bpm

    @property
    def ready(self) -> bool:
        return self.value is not None

    def add(self, t: float, bpm: float) -> None:
        if self.ready:
            return
        if self._start is None:
            self._start = t
        self._samples.append(bpm)
        if t - self._start >= self.calib_sec and len(self._samples) >= self.min_samples:
            self.value = majority_octave_median(self._samples)

    def status(self, now: float) -> str:
        """now는 실행 시작부터의 경과 초. 반환: ready / calibrating / unavailable"""
        if self.ready:
            return "ready"
        return "unavailable" if now >= self.timeout_sec else "calibrating"

    def progress(self, t: float) -> float:
        if self.ready:
            return 1.0
        if self._start is None:
            return 0.0
        return min(1.0, (t - self._start) / self.calib_sec)


# ── 절반·두 배 심박 오류 방지 (2026-10-30 실제 웹캠 기록 반영) ─────────────────
# 실제 실행에서 품질 기준을 통과한 심박이 '진짜 값(약 104)'과 '그 절반(약 52)' 두 무리로 갈렸다.
# (손목 맥박 15초 측정 104bpm, 워치 101~107bpm. 커피를 마신 상태.)
# 절반 위치의 봉우리는 진짜 맥박의 배음이 아니라 따로 존재하는 성분이라 스펙트럼만 보고 고칠 수 없었다.
# 그래서 시간 흐름으로 막는다: 심박은 몇 초 만에 두 배나 절반으로 바뀌지 않는다.
#   1) 평상시 심박 보정: 표본을 '옥타브 무리'로 나누고 가장 많은 무리의 중앙값을 쓴다
#      (단순 중앙값은 두 무리가 비슷하게 섞이면 엉뚱한 값이 된다 — 실제로 54로 잡혔다).
#   2) 보정 후: 평상시 심박의 0.4~0.6배 또는 1.7~2.3배인 추정은 버린다.
# ⚠️ 한계: 운동 등으로 심박이 몇 분에 걸쳐 실제로 두 배가 되는 상황은 버려진다(앉아서 쓰는 상황을 가정).

OCTAVE_LOW = (0.4, 0.6)
OCTAVE_HIGH = (1.7, 2.3)


def is_octave_jump(bpm: float, ref: float | None) -> bool:
    """기준(ref) 대비 절반 또는 두 배 근처면 True."""
    if ref is None or ref <= 0 or bpm != bpm:
        return False
    r = bpm / ref
    return OCTAVE_LOW[0] <= r <= OCTAVE_LOW[1] or OCTAVE_HIGH[0] <= r <= OCTAVE_HIGH[1]


def majority_octave_median(samples: list[float], min_share: float = 0.30) -> float:
    """평상시 심박 보정값. 표본이 서로 절반·두 배 관계인 두 무리로 갈리면, 각 무리가 min_share(30%) 이상일 때
    **높은 무리**의 중앙값을 쓴다. 한 무리가 압도적이면 그 무리의 중앙값.

    높은 쪽을 고르는 근거는 2026-10-30 실제 기록 두 번뿐이다: 두 번 다 진짜 심박(손목 104, 워치 101~107)은
    높은 무리였고, 낮은 무리가 오히려 더 많은 경우(56% 대 37%)도 있어서 '많은 쪽'을 고르면 틀렸다.
    다른 사람·다른 환경에서도 그런지는 확인하지 못했다. 확실히 하려면 .env의 RESTING_HR에 워치 값을 넣는다.
    예: [52, 54, 104, 106, 108] → 두 무리 모두 30% 이상 → 높은 무리 → 106
        [70, 71, 72, 140] → 140은 25%라 무시 → 71"""
    xs = sorted(float(x) for x in samples)
    groups: list[list[float]] = []
    for x in xs:                                   # 정렬된 값을 ±약 30% 안이면 같은 무리로 묶는다
        if groups and x / groups[-1][0] <= 1.33:
            groups[-1].append(x)
        else:
            groups.append([x])
    big = [g for g in groups if len(g) / len(xs) >= min_share]
    if not big:
        big = [max(groups, key=len)]
    return float(statistics.median(big[-1]))   # 정렬돼 있으므로 마지막이 가장 높은 무리
