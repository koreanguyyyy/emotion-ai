# tests/test_compare_watch.py
"""
워치 비교 도구 검증 (2026-11-04): Samsung Health 내보내기 읽기(UTC 시각), 짝짓기, 지표, 얼굴 없는 구간 제외.
실행: python tests/test_compare_watch.py   (가짜 Samsung Health 폴더와 세션 CSV를 만들어 시험)
"""
import csv, json, os, sys, tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT))
import compare_watch as cw

passed = 0
def check(cond, msg):
    global passed
    assert cond, "실패: " + msg
    passed += 1; print("  ✓", msg)

KST = timezone(timedelta(hours=9))
t0 = datetime(2026, 11, 4, 0, 38, 0, tzinfo=KST)
d = tempfile.mkdtemp()
# 워치: 10초마다, 70 → 90 (운동 기록 live_data, CSV 시각은 UTC)
live = [{"start_time": int((t0 + timedelta(seconds=10 * i)).timestamp() * 1000), "heart_rate": 70.0 + i} for i in range(21)]
json.dump(live, open(os.path.join(d, "abc.com.samsung.health.exercise.live_data.json"), "w"))
utc = lambda t: t.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.000")
with open(os.path.join(d, "com.samsung.shealth.exercise.20261104.csv"), "w") as f:
    f.write("com.samsung.shealth.exercise,7013001,3\n")
    f.write("com.samsung.health.exercise.start_time,com.samsung.health.exercise.end_time,com.samsung.health.exercise.live_data\n")
    f.write(f"{utc(t0)},{utc(t0 + timedelta(seconds=200))},abc.com.samsung.health.exercise.live_data.json\n")
# 세션: 매초, 웹캠 = 워치 + 2, 100~150초는 얼굴 없음(원추정은 마지막 값이 남아 있음)
sess = os.path.join(d, "session_test.csv")
with open(sess, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["wall_time", "face_detected", "hr_bpm", "hr_raw_bpm", "hr_quality", "resting_hr", "hr_source", "cam_hr_bpm"])
    for s in range(201):
        face = not (100 <= s <= 150)
        truth = 70 + s / 10
        w.writerow([(t0 + timedelta(seconds=s)).replace(tzinfo=None).isoformat(timespec="seconds"), face,
                    round(truth + 2, 1) if face else "", round(truth + 2, 1) if face else 50, 0.5, 70, "", ""])

samples, sessions = cw.load_watch(cw.SHealth(d))
check(len(samples) == 21 and abs(samples[0][0] - t0.timestamp()) < 0.01, "운동 기록 심박 21개, UTC 시각을 올바르게 읽음")
os.environ["TZ"] = "Asia/Seoul"
try:
    import time; time.tzset()
except AttributeError:
    pass
rows = cw.load_session(sess)
check(rows[120]["raw"] is None and rows[50]["raw"] is not None, "얼굴이 없는 구간의 원추정(남은 값)은 제외")
m = cw.metrics(cw.pair(samples, rows, "cam", 0, 5))
check(m and abs(m["bias"] - 2.0) < 0.6 and m["mae"] < 2.6, f"평균차 약 +2bpm (얻은 값 {m['bias']:+.2f})")
check(m["n"] < 21, "얼굴 없는 구간은 짝에서 빠짐")
check(cw.classify(36, 72) == "절반" and cw.classify(75, 72) == "맞음(±10)" and cw.classify(150, 75) == "두 배", "절반·두 배 분류")
out = cw.main([sess, "--shealth", d, "--no-plot"])
check(out == 0 and os.path.exists(os.path.join(ROOT, "logs", "compare_watch_test.md")), "보고서 저장")
try:
    os.remove(os.path.join(ROOT, "logs", "compare_watch_test.md"))
except OSError:
    pass
print(f"\n모든 검증 통과: {passed}개")
