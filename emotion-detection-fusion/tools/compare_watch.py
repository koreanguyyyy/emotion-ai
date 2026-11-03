# tools/compare_watch.py
"""
웹캠 심박(rPPG)을 갤럭시 워치 기록과 시간대별로 비교한다. 논문의 정확도 표·그래프용.

준비: 측정할 때 워치에서 운동(예: 기타 운동)을 켜 두고 main.py를 같이 실행한다.
      끝나면 휴대폰 Samsung Health → 설정 → 개인 데이터 다운로드 → 폴더(또는 zip)를 노트북으로 옮긴다.
사용: python tools/compare_watch.py --shealth ~/Downloads/samsunghealth.zip              (가장 최근 세션)
      python tools/compare_watch.py logs/session_20261104_003812.csv --shealth <폴더|zip>
옵션: --tol 5     워치 값 하나에 대응시키는 웹캠 구간(±초)
      --lag auto  웹캠이 워치보다 늦게 반응하는 시간(초). auto면 -20~+20초에서 오차가 가장 작은 값

워치(Samsung Health 운동 기록)는 약 10초마다 심박을 남긴다(com.samsung.health.exercise.live_data).
운동 기록과 겹치지 않으면 평소 측정(tracker.heart_rate, 보통 10분 간격)을 쓴다.

비교하는 웹캠 값
  채택값 : 품질 기준(HR_QUALITY_MIN)과 절반·두 배 검사를 통과해 지수에 쓴 값 (hr_bpm 또는 cam_hr_bpm)
  원추정 : 품질과 상관없이 매초 나온 추정 (hr_raw_bpm) — 품질 기준이 무엇을 걸렀는지 보기 위함
출력: 화면 요약 + logs/compare_watch_<세션>.md, .png(시간대별 그래프 + Bland–Altman)
워치도 오차가 있다(손목 광학식, 움직이면 커짐). 여기서는 '정답'이 아니라 '기준 기기'로 쓴다.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import glob
import io
import json
import math
import os
import statistics as st
import sys
import zipfile
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    from ai.stress_index import is_octave_jump
except Exception:   # 단독 실행 대비
    def is_octave_jump(bpm, resting):
        if not bpm or not resting:
            return False
        r = bpm / resting
        return 0.4 <= r <= 0.6 or 1.7 <= r <= 2.3


# ── Samsung Health 읽기 ──────────────────────────────────────────
class SHealth:
    """폴더나 zip 어느 쪽이든 파일 이름으로 읽는다."""

    def __init__(self, path: str):
        self.path = os.path.expanduser(path)
        if zipfile.is_zipfile(self.path):
            self.zip = zipfile.ZipFile(self.path)
            self.names = self.zip.namelist()
        else:
            self.zip = None
            self.names = [os.path.relpath(p, self.path)
                          for p in glob.glob(os.path.join(self.path, "**", "*"), recursive=True) if os.path.isfile(p)]

    def read(self, name: str) -> str:
        if self.zip:
            return self.zip.read(name).decode("utf-8-sig")
        with open(os.path.join(self.path, name), encoding="utf-8-sig") as f:
            return f.read()

    def find(self, suffix: str) -> list[str]:
        return [n for n in self.names if os.path.basename(n).endswith(suffix)]

    def find_contains(self, part: str) -> list[str]:
        return [n for n in self.names if part in os.path.basename(n)]

    def table(self, part: str) -> list[dict]:
        """Samsung Health CSV: 첫 줄은 '테이블이름,버전,개수' 메타 정보, 둘째 줄이 열 이름."""
        rows = []
        for n in self.find_contains(part):
            if not n.endswith(".csv"):
                continue
            lines = self.read(n).splitlines()
            rows += list(csv.DictReader(io.StringIO("\n".join(lines[1:]))))
        return rows


def _short(row: dict, key: str):
    for k, v in row.items():
        if k == key or k.endswith("." + key):
            return v
    return None


def _utc_ms(text: str) -> float:
    """Samsung Health CSV의 시각은 UTC다 (time_offset 열이 따로 있음)."""
    t = datetime.strptime(text[:23], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
    return t.timestamp() * 1000


def load_watch(sh: SHealth):
    """[(epoch초, bpm)] 운동 기록 실시간 심박(약 10초 간격) + 평소 측정, 그리고 운동 목록."""
    samples, sessions = [], []
    for row in sh.table("shealth.exercise."):
        live = _short(row, "live_data")
        start, end = _short(row, "start_time"), _short(row, "end_time")
        if not start:
            continue
        sessions.append((_utc_ms(start) / 1000, _utc_ms(end) / 1000 if end else None, live))
        names = [n for n in sh.names if live and os.path.basename(n) == os.path.basename(live)]
        if not names:
            continue
        for x in json.loads(sh.read(names[0])):
            if x.get("heart_rate") and x.get("start_time"):
                samples.append((x["start_time"] / 1000, float(x["heart_rate"]), "exercise"))
    for row in sh.table("tracker.heart_rate."):
        hr, start = _short(row, "heart_rate"), _short(row, "start_time")
        try:
            if hr and start and float(hr) > 0:
                samples.append((_utc_ms(start) / 1000, float(hr), "tracker"))
        except ValueError:
            pass
    samples.sort()
    return samples, sessions


# ── 세션 CSV 읽기 ────────────────────────────────────────────────
def _f(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def load_session(path: str):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            t = datetime.fromisoformat(r["wall_time"])   # 노트북 현지 시각
            if t.tzinfo is None:
                t = t.astimezone()                       # 이 컴퓨터의 시간대로 해석
            cam = _f(r.get("cam_hr_bpm")) if r.get("hr_source") == "watch" else (_f(r.get("cam_hr_bpm")) or _f(r.get("hr_bpm")))
            face = r.get("face_detected") == "True"
            # 얼굴이 안 보이면 hr_raw_bpm은 마지막 추정이 그대로 남아 있으므로 비교에서 뺀다
            rows.append({"t": t.timestamp(), "cam": cam, "raw": _f(r.get("hr_raw_bpm")) if face else None,
                         "q": _f(r.get("hr_quality")), "resting": _f(r.get("resting_hr")),
                         "face": face})
    return rows


# ── 짝짓기와 지표 ────────────────────────────────────────────────
def pair(watch, sess, key, lag, tol):
    """워치 값마다 [t+lag-tol, t+lag+tol] 안의 웹캠 값 중앙값과 짝짓는다."""
    ts = [r["t"] for r in sess]
    out = []
    for t, w, _src in watch:
        a = bisect.bisect_left(ts, t + lag - tol)
        b = bisect.bisect_right(ts, t + lag + tol)
        vals = [sess[i][key] for i in range(a, b) if sess[i][key] is not None]
        if vals:
            out.append((t, w, st.median(vals)))
    return out


def metrics(pairs):
    if len(pairs) < 2:
        return None
    w = [p[1] for p in pairs]; c = [p[2] for p in pairs]
    d = [ci - wi for wi, ci in zip(w, c)]
    bias = st.mean(d); sd = st.stdev(d)
    try:
        r = st.correlation(w, c)
    except Exception:
        r = float("nan")
    return {"n": len(d), "bias": bias, "mae": st.mean(abs(x) for x in d),
            "rmse": math.sqrt(st.mean(x * x for x in d)), "r": r,
            "loa": (bias - 1.96 * sd, bias + 1.96 * sd),
            "within5": sum(abs(x) <= 5 for x in d) / len(d), "within10": sum(abs(x) <= 10 for x in d) / len(d),
            "mape": st.mean(abs(x) / wi for x, wi in zip(d, w)) * 100}


def classify(est, truth):
    r = est / truth
    if abs(est - truth) <= 10:
        return "맞음(±10)"
    if 0.4 <= r <= 0.6:
        return "절반"
    if 1.7 <= r <= 2.3:
        return "두 배"
    return "기타 오차"


def fmt(m):
    if not m:
        return "짝 2개 미만"
    return (f"n={m['n']}  평균차 {m['bias']:+.1f}  MAE {m['mae']:.1f}  RMSE {m['rmse']:.1f}  r={m['r']:.2f}  "
            f"±5bpm {m['within5']*100:.0f}%  ±10bpm {m['within10']*100:.0f}%  "
            f"일치한계 {m['loa'][0]:+.1f}~{m['loa'][1]:+.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", help="logs/session_*.csv (생략하면 가장 최근)")
    ap.add_argument("--shealth", required=True, help="Samsung Health 개인 데이터 폴더 또는 zip")
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--lag", default="auto")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)

    path = a.session or max(glob.glob(os.path.join(ROOT, "logs", "session_*.csv")), key=os.path.getmtime)
    sess = load_session(path)
    watch_all, sessions = load_watch(SHealth(a.shealth))
    t0, t1 = sess[0]["t"], sess[-1]["t"]
    watch = [w for w in watch_all if t0 - 30 <= w[0] <= t1 + 30]
    kst = lambda s: datetime.fromtimestamp(s).astimezone().strftime("%m-%d %H:%M:%S")
    print(f"세션: {os.path.basename(path)}  {kst(t0)} ~ {kst(t1)} ({(t1 - t0) / 60:.1f}분)")
    print(f"워치: 전체 {len(watch_all)}개 중 세션과 겹치는 값 {len(watch)}개 "
          f"(운동 기록 {sum(w[2] == 'exercise' for w in watch)}, 평소 측정 {sum(w[2] == 'tracker' for w in watch)})")
    if len(watch) < 2:
        print("겹치는 워치 기록이 없습니다. 세션 시각과 운동 기록 시각을 확인하세요:")
        for s, e, _ in sessions:
            print(f"  운동 {kst(s)} ~ {kst(e) if e else '?'}")
        return 1

    # 지연 시간: 웹캠은 최근 20초 창으로 계산하므로 워치보다 늦게 반응할 수 있다
    if a.lag == "auto":
        cands = [(metrics(pair(watch, sess, "cam", L, a.tol)), L) for L in range(-20, 21, 2)]
        cands = [(m["mae"], L) for m, L in cands if m and m["n"] >= max(5, len(watch) // 5)]
        lag = min(cands)[1] if cands else 0
    else:
        lag = float(a.lag)

    p_cam = pair(watch, sess, "cam", lag, a.tol)
    p_raw = pair(watch, sess, "raw", lag, a.tol)
    m_cam, m_raw = metrics(p_cam), metrics(p_raw)
    coverage = len(p_cam) / len(watch)
    ts_all = [r["t"] for r in sess]

    def face_near(t):
        i = bisect.bisect_left(ts_all, t + lag)
        near = sess[max(0, i - int(a.tol)):i + int(a.tol) + 1]
        return near and sum(r["face"] for r in near) / len(near) >= 0.5

    watch_face = [w for w in watch if face_near(w[0])]
    cov_face = sum(1 for p in p_cam if face_near(p[0])) / max(1, len(watch_face))
    face_pct = sum(r["face"] for r in sess) / len(sess)

    # 원추정 분류 + 절반·두 배 규칙 점검 (원추정 1초 단위, 그 시각의 워치 값을 선형 보간)
    wt = [w[0] for w in watch]; wv = [w[1] for w in watch]

    def watch_at(t):
        i = bisect.bisect_left(wt, t - lag)
        if i <= 0 or i >= len(wt):
            return None
        t_a, t_b = wt[i - 1], wt[i]
        if t_b - t_a > 30:
            return None
        f = (t - lag - t_a) / (t_b - t_a)
        return wv[i - 1] + f * (wv[i] - wv[i - 1])

    cls_raw, cls_cam, rule = {}, {}, {"맞게 버림": 0, "잘못 버림": 0, "놓침": 0}
    for r in sess:
        tv = watch_at(r["t"])
        if tv is None:
            continue
        if r["raw"] is not None:
            c = classify(r["raw"], tv)
            cls_raw[c] = cls_raw.get(c, 0) + 1
            if r["resting"]:
                jump = is_octave_jump(r["raw"], r["resting"])
                if jump and c in ("절반", "두 배"):
                    rule["맞게 버림"] += 1
                elif jump:
                    rule["잘못 버림"] += 1
                elif c in ("절반", "두 배"):
                    rule["놓침"] += 1
        if r["cam"] is not None:
            c = classify(r["cam"], tv)
            cls_cam[c] = cls_cam.get(c, 0) + 1

    pct = lambda d: ", ".join(f"{k} {v} ({v / max(1, sum(d.values())) * 100:.0f}%)" for k, v in
                              sorted(d.items(), key=lambda kv: -kv[1]))
    lines = [
        f"# 웹캠 심박 vs 갤럭시 워치 — {os.path.basename(path)}",
        "",
        f"- 구간: {kst(t0)} ~ {kst(t1)} ({(t1 - t0) / 60:.1f}분), 워치 값 {len(watch)}개(약 10초 간격)",
        f"- 짝짓기: 워치 값마다 웹캠 ±{a.tol:.0f}초 중앙값, 웹캠 지연 {lag:+.0f}초"
        + (" (오차가 가장 작은 값 자동 선택)" if a.lag == "auto" else ""),
        f"- 워치 심박 범위 {min(wv):.0f}~{max(wv):.0f}bpm, 중앙값 {st.median(wv):.0f}",
        "",
        "| 웹캠 값 | 짝 수 | 평균차 | MAE | RMSE | r | ±5bpm | ±10bpm | 95% 일치한계 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in (("채택값(지수에 쓴 값)", m_cam), ("원추정(품질 무관)", m_raw)):
        if m:
            lines.append(f"| {name} | {m['n']} | {m['bias']:+.1f} | {m['mae']:.1f} | {m['rmse']:.1f} | {m['r']:.2f} | "
                         f"{m['within5'] * 100:.0f}% | {m['within10'] * 100:.0f}% | {m['loa'][0]:+.1f} ~ {m['loa'][1]:+.1f} |")
        else:
            lines.append(f"| {name} | 0 | | | | | | | |")
    lines += [
        "",
        f"- **채택률**: 워치 값 중 웹캠 채택값이 있던 비율 {coverage * 100:.0f}% "
        f"(나머지는 품질 미달·얼굴 미검출로 웹캠이 심박을 내지 않음)",
        f"- 얼굴이 보인 시간 {face_pct * 100:.0f}% · 얼굴이 보인 구간의 워치 값 {len(watch_face)}개 중 채택률 {cov_face * 100:.0f}%",
        f"- 원추정 1초 단위 분류: {pct(cls_raw)}",
        f"- 채택값 1초 단위 분류: {pct(cls_cam)}",
        f"- 절반·두 배 규칙(원추정에 적용해 봄): 맞게 버림 {rule['맞게 버림']}, 잘못 버림 {rule['잘못 버림']}, "
        f"놓침(절반·두 배인데 규칙에 안 걸림) {rule['놓침']}",
        "",
        "해석 주의: 워치도 손목 광학식이라 오차가 있다(움직이면 커짐). 한 사람·한 번의 기록이다.",
    ]
    report = "\n".join(lines)
    print("\n" + report)

    base = os.path.join(ROOT, "logs", "compare_watch_" + os.path.basename(path).replace("session_", "").replace(".csv", ""))
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n저장: {base}.md")

    if not a.no_plot:
        try:
            plot(base + ".png", sess, watch, p_cam, lag, t0, m_cam)
            print(f"저장: {base}.png")
        except ImportError:
            print("(matplotlib이 없어 그래프는 건너뜀: pip install matplotlib)")
    return 0


def plot(out, sess, watch, p_cam, lag, t0, m_cam):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR", "Malgun Gothic"):
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    m = lambda t: (t - t0) / 60
    raw = [(m(r["t"] - lag), r["raw"]) for r in sess if r["raw"] is not None]
    cam = [(m(r["t"] - lag), r["cam"]) for r in sess if r["cam"] is not None]
    if raw:
        ax1.scatter(*zip(*raw), s=4, c="#C9C2B4", label="웹캠 원추정(품질 무관)")
    if cam:
        ax1.scatter(*zip(*cam), s=7, c="#B8501F", label="웹캠 채택값")
    ax1.plot([m(w[0]) for w in watch], [w[1] for w in watch], "-o", ms=3, lw=1.6, c="#2B5F9E", label="갤럭시 워치")
    ax1.set_xlabel("세션 시작 후 (분)"); ax1.set_ylabel("심박 (bpm)")
    ax1.set_title("시간대별 심박"); ax1.legend(loc="upper right", fontsize=8); ax1.grid(alpha=.3)

    if p_cam:
        mean = [(w + c) / 2 for _, w, c in p_cam]; diff = [c - w for _, w, c in p_cam]
        ax2.scatter(mean, diff, s=14, c="#B8501F")
        if m_cam:
            for y, ls in ((m_cam["bias"], "-"), (m_cam["loa"][0], "--"), (m_cam["loa"][1], "--")):
                ax2.axhline(y, color="#15202B", ls=ls, lw=1)
        ax2.set_xlabel("두 기기 평균 (bpm)"); ax2.set_ylabel("웹캠 − 워치 (bpm)")
        ax2.set_title("Bland–Altman (채택값)"); ax2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    sys.exit(main())
