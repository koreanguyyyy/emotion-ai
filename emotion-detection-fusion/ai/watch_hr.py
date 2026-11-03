# ai/watch_hr.py
"""
갤럭시 워치 심박 받기 — 워치 앱(emotionAi/watch-hr-app)이 1초마다 보내는 심박을 받는다.

  POST http://<이 노트북 IP>:8766/hr  {"bpm": 82, "rr": 731, "ts": <워치 ms>, "seq": 15}
  GET  http://<이 노트북 IP>:8766/hr  → 마지막 값 (연결 확인용, 브라우저로 열어 봐도 된다)

.env
  HR_SOURCE=webcam   지수에 쓸 심박: webcam(기본) / watch / auto(워치가 보내고 있으면 워치, 끊기면 웹캠)
  WATCH_PORT=8766    0이면 받지 않음
  WATCH_HOST=0.0.0.0 워치가 와이파이로 접속해야 하므로 같은 네트워크에 연다 (127.0.0.1이면 워치가 못 들어온다)
  WATCH_MAX_AGE_SEC=5  이보다 오래된 워치 값은 끊긴 것으로 본다

어느 설정이든 워치 값과 웹캠 값은 세션 CSV에 함께 기록된다 (watch_hr_bpm, cam_hr_bpm) → 정확도 비교 자료.
보안: 같은 와이파이의 누구나 숫자를 보낼 수 있다. 받는 값은 30~220bpm 숫자만이고, 그 밖의 것은 버린다.
WATCH_TOKEN을 정하면 워치 앱 주소 뒤에 ?token=... 을 붙인 요청만 받는다 (공용 와이파이에서 권장).
"""
from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

BPM_MIN, BPM_MAX = 30.0, 220.0
RR_MIN, RR_MAX = 250.0, 2000.0   # ms (240~30bpm)


class WatchHR:
    """워치가 보낸 마지막 심박. clock은 시험용으로 바꿀 수 있다(기본 time.monotonic)."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.bpm: float | None = None
        self.rr: float | None = None
        self.recv_t: float = -1e9
        self.seq = 0            # 받은 개수 (새 값인지 구분)
        self.rejected = 0

    def push(self, bpm, rr=None) -> bool:
        try:
            b = float(bpm)
        except (TypeError, ValueError):
            b = float("nan")
        if not (BPM_MIN <= b <= BPM_MAX):
            with self.lock:
                self.rejected += 1
            return False
        try:
            r = float(rr) if rr is not None else None
        except (TypeError, ValueError):
            r = None
        if r is not None and not (RR_MIN <= r <= RR_MAX):
            r = None
        with self.lock:
            self.bpm, self.rr, self.recv_t = b, r, self.clock()
            self.seq += 1
        return True

    def latest(self, max_age: float = 5.0):
        """(bpm, rr, seq) — max_age초보다 오래됐으면 (None, None, seq)."""
        with self.lock:
            if self.bpm is None or self.clock() - self.recv_t > max_age:
                return None, None, self.seq
            return self.bpm, self.rr, self.seq

    def age(self) -> float | None:
        with self.lock:
            return None if self.bpm is None else self.clock() - self.recv_t


class HRSelector:
    """지수에 쓸 심박을 웹캠과 워치 중에서 고른다 (face_fusion.py가 프레임마다 부른다).
    webcam: 웹캠만 / watch: 워치만 / auto: 워치가 최근에 보냈으면 워치, 끊기면 웹캠."""

    def __init__(self, source: str = "webcam"):
        if source not in ("webcam", "watch", "auto"):
            print(f"[Watch] HR_SOURCE={source!r}는 알 수 없는 값 → webcam")
            source = "webcam"
        self.source = source
        self.watch_seq_used = 0

    def use_cam(self, watch_fresh: bool) -> bool:
        """품질을 통과한 웹캠 추정이 나왔을 때 지수에 쓸지."""
        return self.source == "webcam" or (self.source == "auto" and not watch_fresh)

    def use_watch(self, bpm, seq: int) -> bool:
        """워치 값이 새로 들어왔을 때 지수에 쓸지 (같은 값을 프레임마다 다시 넣지 않는다)."""
        if self.source == "webcam" or bpm is None or seq == self.watch_seq_used:
            return False
        self.watch_seq_used = seq
        return True


WATCH = WatchHR()   # 프로그램 전체에서 하나 (main.py가 서버를 열고 face_fusion.py가 읽는다)


def lan_ip() -> str:
    """워치 앱에 입력할 이 노트북의 와이파이 IP (실제로 패킷을 보내지는 않는다)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def start_watch_server(port: int, store: WatchHR = WATCH, host: str = "0.0.0.0",
                       token: str | None = None) -> ThreadingHTTPServer | None:
    if not port:
        return None

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            if code == 204:
                self.end_headers()
                return
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not token:
                return True
            q = parse_qs(urlparse(self.path).query)
            return q.get("token", [""])[0] == token

        def do_OPTIONS(self):  # noqa: N802
            self._send(204, {})

        def do_GET(self):  # noqa: N802
            if urlparse(self.path).path != "/hr":
                self._send(404, {"error": "not found"})
                return
            bpm, rr, seq = store.latest(1e9)
            age = store.age()
            self._send(200, {"bpm": bpm, "rr": rr, "received": seq,
                             "ageSec": None if age is None else round(age, 1)})

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/hr":
                self._send(404, {"error": "not found"})
                return
            if not self._authorized():
                self._send(403, {"error": "token"})
                return
            try:
                n = min(int(self.headers.get("Content-Length") or 0), 4096)
                body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                ok = isinstance(body, dict) and store.push(body.get("bpm"), body.get("rr"))
            except (ValueError, json.JSONDecodeError):
                ok = False
            if ok:
                self._send(200, {"ok": True})
            else:
                self._send(400, {"ok": False, "error": f"bpm은 {BPM_MIN:.0f}~{BPM_MAX:.0f} 숫자여야 합니다"})

        def log_message(self, *args):
            pass

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        print(f"[Watch] 워치 심박 수신을 열지 못했습니다 (포트 {port}): {e}")
        return None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    where = lan_ip() if host in ("0.0.0.0", "") else host
    print(f"[Watch] 워치 심박 받는 중 — 워치 앱에 '{where}:{port}' 입력"
          + (" (WATCH_TOKEN 사용 중: 주소 뒤에 ?token=... )" if token else ""))
    return server
