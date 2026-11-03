# tests/test_watch_hr.py
"""
갤럭시 워치 심박 받기 검증 (2026-11-03): 값 검사, 끊김 판단, 웹캠/워치 선택, 받는 서버, 실시간 통로.
실행: python tests/test_watch_hr.py   (워치 없이 — 워치 앱이 보내는 요청을 흉내 낸다)
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from ai.watch_hr import HRSelector, WatchHR, start_watch_server
from ai.live_server import build_live_payload
from ai.session_logger import SESSION_FIELDS

passed = 0


def check(cond, msg):
    global passed
    assert cond, "실패: " + msg
    passed += 1
    print("  ✓", msg)


class Clock:
    def __init__(self): self.t = 100.0
    def __call__(self): return self.t


print("[1] 받은 값 검사")
clk = Clock()
w = WatchHR(clock=clk)
check(w.latest() == (None, None, 0), "아직 받은 값 없음")
check(w.push(82, 731) and w.latest() == (82.0, 731.0, 1), "정상 값 저장 (심박 + 심박 간격)")
check(not w.push(0) and not w.push(-3) and not w.push(250) and not w.push("abc") and not w.push(None),
      "0·음수(미착용)·250·글자·빈 값은 버림")
check(w.latest()[0] == 82.0 and w.rejected == 5, "버린 값은 마지막 값을 바꾸지 않음")
w.push(90, 5000)
check(w.latest()[1] is None, "말이 안 되는 심박 간격(5000ms)은 간격만 버림")
clk.t += 4.9
check(w.latest(5.0)[0] == 90.0, "5초 안의 값은 유효")
clk.t += 0.2
check(w.latest(5.0)[0] is None and w.latest(5.0)[2] == 2, "5초가 지나면 끊긴 것으로 봄")

print("[2] 웹캠 / 워치 / 자동 선택")
s = HRSelector("webcam")
check(s.use_cam(watch_fresh=True) and not s.use_watch(80, 1), "webcam: 워치가 보내도 웹캠만")
s = HRSelector("watch")
check(not s.use_cam(False) and s.use_watch(80, 1), "watch: 웹캠은 안 쓰고 워치 사용")
check(not s.use_watch(80, 1) and s.use_watch(81, 2), "같은 워치 값은 프레임마다 다시 넣지 않음")
check(not s.use_watch(None, 3), "끊긴 워치 값은 안 씀")
s = HRSelector("auto")
check(not s.use_cam(watch_fresh=True) and s.use_cam(watch_fresh=False), "auto: 워치가 보내면 워치, 끊기면 웹캠")
check(HRSelector("garmin").source == "webcam", "잘못된 설정은 webcam으로")

print("[3] 받는 서버 (워치 앱 요청 흉내)")
store = WatchHR()
srv = start_watch_server(18767, store, host="127.0.0.1")
U = "http://127.0.0.1:18767/hr"


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


code, out = post(U, {"bpm": 77, "rr": 780, "ts": 1793700000000, "seq": 1, "device": "watch"})
check(code == 200 and out["ok"] and store.latest()[0] == 77.0, "POST /hr → 저장")
code, out = post(U, {"bpm": -3})
check(code == 400 and store.latest()[0] == 77.0, "미착용 값은 400")
got = json.loads(urllib.request.urlopen(U, timeout=3).read())
check(got["bpm"] == 77.0 and got["received"] == 1 and got["ageSec"] is not None, "GET /hr로 연결 확인")
check(post("http://127.0.0.1:18767/other", {"bpm": 70})[0] == 404, "다른 경로는 404")
srv.shutdown()

tok = WatchHR()
srv = start_watch_server(18768, tok, host="127.0.0.1", token="abc")
check(post("http://127.0.0.1:18768/hr", {"bpm": 70})[0] == 403, "토큰을 정하면 토큰 없는 요청은 403")
check(post("http://127.0.0.1:18768/hr?token=abc", {"bpm": 70})[0] == 200, "토큰이 맞으면 저장")
srv.shutdown()

print("[4] 기록·실시간 통로")
check(all(k in SESSION_FIELDS for k in ("hr_source", "watch_hr_bpm", "watch_rr_ms", "cam_hr_bpm")),
      "세션 CSV에 워치·웹캠 심박 칸")
p = build_live_payload({"hr_source": "watch", "watch_hr_bpm": 83.04, "cam_hr_bpm": 88.6}, None)
check(p["hrSource"] == "watch" and p["watchHeartRate"] == 83.0 and p["webcamHeartRate"] == 88.6,
      "웹페이지에 출처·워치·웹캠 심박 전달")

print(f"\n모든 검증 통과: {passed}개")
