# tests/test_rppg_timed.py
"""
시각 기반 rPPG(detection/rppg_timed.py) 합성 데이터 검증.
실행: emotion-detection-fusion 폴더에서  python tests/test_rppg_timed.py
       (빠르게: python tests/test_rppg_timed.py --quick)

피부색 모델: 채널별 기본색 × (1 + 조명 변화) × (1 + 맥박 강도 × 맥파) + 영역별 독립 잡음.
맥박 강도의 채널 비율(R:G:B = 0.33:0.77:0.53)은 POS 논문(Wang et al., 2017)의 피부 반사 모델 값.
"""
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

from detection.rppg_timed import TimedRPPG
from emotion_pipeline.rppg import CHROMExtractor

QUICK = "--quick" in sys.argv
SEEDS = 10 if QUICK else 40
PULSE_W = np.array([0.33, 0.77, 0.53])
BASE = {"forehead": np.array([150.0, 110.0, 90.0]), "cheek": np.array([160.0, 105.0, 88.0])}
passed = 0


def check(cond, msg):
    global passed
    assert cond, "실패: " + msg
    passed += 1
    print("  ✓", msg)


def frame_times(rng, secs, fps=30.0, jitter=0.15, drop=0.03):
    """실제 웹캠처럼 불규칙한 프레임 시각. jitter: 간격의 상대 흔들림, drop: 프레임 누락 확률."""
    t, out = 0.0, []
    while t < secs:
        dt = (1 / fps) * (1 + rng.uniform(-jitter, jitter))
        if rng.random() < drop:
            dt *= 2
        t += dt
        out.append(t)
    return np.array(out)


def skin(rng, ts, bpm, amp, noise, light=None, harmonic=0.3):
    """ROI별 평균 RGB 시계열. amp=0이면 맥박 없음(잡음만)."""
    f = bpm / 60
    p = np.sin(2 * np.pi * f * ts) + harmonic * np.sin(4 * np.pi * f * ts + 0.7)
    light = np.zeros_like(ts) if light is None else light
    out = {}
    for name, base in BASE.items():
        sig = base[None, :] * (1 + light[:, None]) * (1 + amp * PULSE_W[None, :] * p[:, None])
        out[name] = sig * (1 + rng.normal(0, noise, sig.shape))
    return out


def run_timed(ts, rois, forehead_only=False):
    ex = TimedRPPG(window_sec=20)
    ests = []
    for k, t in enumerate(ts):
        e = ex.update(t, rois["forehead"][k], None if forehead_only else rois["cheek"][k])
        if e is not None:
            ests.append(e)
    return ests


def run_old(ts, rois, fs):
    """2학기 방식: 프레임 간격이 일정하다고 가정하고 고정 fs 사용, 이마만, CHROM만."""
    ex = CHROMExtractor(fs=fs, window_sec=20)
    ests, last = [], None
    for k in range(len(ts)):
        e = ex.update(tuple(rois["forehead"][k]))
        if e is not None and e is not last:
            last = e
            ests.append(e)
    return ests


print("== 1. fps 문제: 실제 30fps(불규칙)인데 카메라가 15fps라고 알려줄 때 (정답 72 bpm)")
rng = np.random.default_rng(1)
ts = frame_times(rng, 40)
rois = skin(rng, ts, 72, amp=0.004, noise=0.001)
old15 = run_old(ts, rois, 15.0)[-1]
old_probe = run_old(ts, rois, 27.5)[-1]
new = run_timed(ts, rois)[-1]
print(f"  2학기 방식, fps 15 (카메라 표기)     : {old15.bpm:6.1f} bpm  품질 {old15.quality:.2f}")
print(f"  fusion 1차, fps 27.5 (첫 3초 측정)   : {old_probe.bpm:6.1f} bpm  품질 {old_probe.quality:.2f}")
print(f"  새 방식, 프레임 시각 기반 (실제 {new.fps_actual:.1f}fps): {new.bpm:6.1f} bpm  품질 {new.quality:.2f} ({new.method})")
check(abs(new.bpm - 72) <= 3, "프레임 시각 기반이면 fps 표기와 상관없이 정답에 맞음")
print("  (참고) fps를 절반(15)으로 잡으면 기본 맥박 72는 36bpm으로 보여 검색 범위(42~180) 밖으로 빠지고,\n"
      "         맥파의 두 배 주파수 성분(144)이 절반인 72로 보여 우연히 정답 근처가 나온다.")
print("  심박이 오른 경우 (스트레스 상황):")
for true_bpm in (90, 105):
    r2 = np.random.default_rng(true_bpm)
    ts2 = frame_times(r2, 40)
    ro2 = skin(r2, ts2, true_bpm, amp=0.004, noise=0.001)
    o = run_old(ts2, ro2, 15.0)[-1]
    nn = run_timed(ts2, ro2)[-1]
    print(f"    정답 {true_bpm} bpm → 2학기 방식(15fps) {o.bpm:5.1f} bpm / 새 방식 {nn.bpm:5.1f} bpm")
    check(abs(nn.bpm - true_bpm) <= 3, f"정답 {true_bpm}: 새 방식은 정확")
    check(abs(o.bpm - true_bpm / 2) <= 4,
          f"정답 {true_bpm}: 2학기 방식은 절반({o.bpm:.0f})으로 나옴 → 심박이 오를수록 낮게 보임")

ts_var = []
t = 0.0
while t < 40:    # 처리 속도가 중간에 30 → 18fps로 떨어지는 경우 (다른 프로그램이 CPU를 쓸 때)
    t += 1 / (30 if t < 20 else 18)
    ts_var.append(t)
ts_var = np.array(ts_var)
rois_v = skin(rng, ts_var, 80, amp=0.004, noise=0.001)
nv = run_timed(ts_var, rois_v)[-1]
ov = run_old(ts_var, rois_v, 30.0)[-1]
print(f"  처리 속도가 20초에 30→18fps로 떨어질 때 (정답 80): 새 방식 {nv.bpm:.1f} / 고정 30fps {ov.bpm:.1f} bpm")
check(abs(nv.bpm - 80) <= 3, "처리 속도가 중간에 바뀌어도 새 방식은 정답 유지")

print(f"\n== 2. 잡음만 있을 때 품질 분포 → 품질 기준 다시 정하기 ({SEEDS}회)")
q_old, q_new = [], []
for s in range(SEEDS):
    r = np.random.default_rng(100 + s)
    ts_n = frame_times(r, 45)
    rn = skin(r, ts_n, 72, amp=0.0, noise=0.002)
    q_old += [e.quality for e in run_old(ts_n, rn, float(len(ts_n) / ts_n[-1]))]
    q_new += [e.quality for e in run_timed(ts_n, rn)]
q_old, q_new = np.array(q_old), np.array(q_new)
for name, q in (("2학기 방식(CHROM 하나)", q_old), ("새 방식(둘 중 높은 쪽)", q_new)):
    print(f"  {name:22} p50 {np.percentile(q, 50):.2f}  p90 {np.percentile(q, 90):.2f}  "
          f"p99 {np.percentile(q, 99):.2f}  최대 {q.max():.2f}  | ≥0.35 {np.mean(q >= .35):.1%}  ≥0.40 {np.mean(q >= .40):.1%}")
THRESH = float(np.ceil(np.percentile(q_new, 99.5) * 20) / 20)   # 잡음 통과 0.5% 이하가 되는 0.05 단위 값
print(f"  → 새 방식 품질 기준: {THRESH:.2f} (잡음의 99.5%가 이 아래)")
check(np.mean(q_new >= THRESH) <= 0.005, f"새 기준 {THRESH:.2f}에서 잡음 통과율 {np.mean(q_new >= THRESH):.1%}")

print(f"\n== 3. 약한 맥박: 맥북 카메라처럼 신호가 잡음에 묻힐 때 ({SEEDS}회, 정답 66 bpm)")
print("  맥박세기   방식                       기준 통과율   통과한 값 중 ±5bpm")
rows = {}
for amp in (0.0015, 0.0025, 0.004):
    for label, kind in (("2학기(이마, CHROM, 0.35)", "old"), (f"새 방식(이마+볼, 최고값, {THRESH:.2f})", "new"),
                        ("새 방식이지만 이마만", "new_fh")):
        ok = acc = 0; tot = 0
        for s in range(SEEDS):
            r = np.random.default_rng(500 + s)
            ts_w = frame_times(r, 40)
            rw = skin(r, ts_w, 66, amp=amp, noise=0.002)
            if kind == "old":
                es = run_old(ts_w, rw, float(len(ts_w) / ts_w[-1])); th = 0.35
            else:
                es = run_timed(ts_w, rw, forehead_only=(kind == "new_fh")); th = THRESH
            for e in es:
                tot += 1
                if e.quality >= th:
                    ok += 1
                    acc += abs(e.bpm - 66) <= 5
        rows[(amp, kind)] = (ok / tot, acc / ok if ok else float("nan"))
        print(f"  {amp:.4f}   {label:28} {ok / tot:7.0%}       {acc / ok if ok else float('nan'):7.0%}")
check(rows[(0.0025, "new")][0] > rows[(0.0025, "old")][0],
      f"중간 세기 맥박에서 새 방식이 더 자주 통과 ({rows[(0.0025, 'new')][0]:.0%} vs {rows[(0.0025, 'old')][0]:.0%})")
check(rows[(0.0025, "new")][0] >= rows[(0.0025, "new_fh")][0],
      "이마+볼이 이마만 쓸 때보다 통과율이 같거나 높음")
check(all(np.isnan(v[1]) or v[1] >= 0.95 for k, v in rows.items() if k[1] == "new"),
      "새 방식에서 기준을 통과한 값은 95% 이상이 정답 ±5bpm 안")

print("\n== 4. 조명 변화: 화면 밝기가 바뀌는 상황 (천천히 변함 + 2초마다 계단식 변화, 정답 70 bpm)")
res = {"chrom": [], "pos": []}
for s in range(SEEDS // 2):
    r = np.random.default_rng(900 + s)
    ts_l = frame_times(r, 40)
    light = 0.03 * np.sin(2 * np.pi * 0.05 * ts_l) + 0.01 * (np.floor(ts_l / 2) % 2)   # 1% 계단을 2초마다
    rl = skin(r, ts_l, 70, amp=0.003, noise=0.001, light=light)
    for e in run_timed(ts_l, rl):
        res["chrom"].append(abs(e.chrom_bpm - 70) <= 5)
        res["pos"].append(abs(e.pos_bpm - 70) <= 5)
print(f"  ±5bpm 정확도: CHROM {np.mean(res['chrom']):.0%}, POS {np.mean(res['pos']):.0%}")
check(max(np.mean(res["chrom"]), np.mean(res["pos"])) >= 0.9, "조명이 바뀌어도 둘 중 하나는 90% 이상 정확")

print("\n== 5. 얼굴을 놓쳤다가 다시 잡을 때")
ex = TimedRPPG(window_sec=20)
r = np.random.default_rng(7)
ts_g = frame_times(r, 50)
rg = skin(r, ts_g, 75, amp=0.004, noise=0.001)
last = None
for k, t in enumerate(ts_g):
    if 25 < t < 27:
        continue          # 2초 동안 얼굴 놓침
    e = ex.update(t, rg["forehead"][k], rg["cheek"][k])
    if e is not None:
        last = (t, e)
check(last is not None and last[0] > 46 and abs(last[1].bpm - 75) <= 3,
      f"끊긴 뒤 버퍼를 새로 채워 다시 정확히 추정 ({last[1].bpm:.1f} bpm, {last[0]:.0f}초)")

print(f"\n모든 검증 통과: {passed}개  (품질 기준 권장값 HR_QUALITY_MIN={THRESH:.2f})")
