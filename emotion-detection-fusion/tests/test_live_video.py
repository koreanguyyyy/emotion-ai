# tests/test_live_video.py — 얼굴 인식 화면(MJPEG) 통로 시험
import os
import socket
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ai.frame_store import FrameStore
from ai.live_server import start_live_server


def _frame(w=1280, h=720):
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[:, :, 1] = 120
    return f


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class FrameStoreTest(unittest.TestCase):
    def test_no_viewer_no_encode(self):
        fs = FrameStore(fps=10, width=640, enabled=True)
        self.assertFalse(fs.publish(_frame(), now=1.0))
        self.assertIsNone(fs.wait_next(0, timeout=0.01))

    def test_publish_and_resize(self):
        import cv2
        fs = FrameStore(fps=10, width=640, enabled=True)
        fs.add_viewer()
        self.assertTrue(fs.publish(_frame(), now=1.0))
        seq, jpg = fs.wait_next(0, timeout=0.1)
        self.assertEqual(seq, 1)
        self.assertTrue(jpg.startswith(b"\xff\xd8"))
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape[1], 640)
        self.assertEqual(img.shape[0], 360)

    def test_throttle(self):
        fs = FrameStore(fps=10, width=640, enabled=True)
        fs.add_viewer()
        self.assertTrue(fs.publish(_frame(), now=1.0))
        self.assertFalse(fs.publish(_frame(), now=1.05))   # 0.1초 안 됨
        self.assertTrue(fs.publish(_frame(), now=1.11))

    def test_disabled(self):
        fs = FrameStore(enabled=False)
        fs.add_viewer()
        self.assertFalse(fs.publish(_frame(), now=1.0))

    def test_viewer_count_never_negative(self):
        fs = FrameStore(enabled=True)
        fs.remove_viewer()
        self.assertEqual(fs.viewers, 0)

    def test_bad_frame_does_not_raise(self):
        fs = FrameStore(enabled=True)
        fs.add_viewer()
        self.assertFalse(fs.publish(np.zeros((0, 0, 3), dtype=np.uint8), now=1.0))


class VideoEndpointTest(unittest.TestCase):
    def test_mjpeg_stream(self):
        fs = FrameStore(fps=30, width=320, enabled=True)
        port = _free_port()
        server = start_live_server(port, lambda: {"ok": True}, frames=fs)
        self.assertIsNotNone(server)
        stop = threading.Event()

        def feeder():
            while not stop.is_set():
                fs.publish(_frame(640, 480))
                time.sleep(0.03)
        threading.Thread(target=feeder, daemon=True).start()
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/video", timeout=3)
            self.assertIn("multipart/x-mixed-replace", r.headers["Content-Type"])
            self.assertEqual(r.headers["Access-Control-Allow-Origin"], "*")
            data = b""
            while data.count(b"--frame") < 2 and len(data) < 500_000:
                data += r.read(1024)
            self.assertIn(b"Content-Type: image/jpeg", data)
            self.assertIn(b"\xff\xd8", data)
            self.assertGreaterEqual(fs.viewers, 1)
            r.close()
            for _ in range(50):           # 연결을 닫으면 보는 사람 수가 줄어든다
                if fs.viewers == 0:
                    break
                fs.publish(_frame(640, 480)); time.sleep(0.05)
            self.assertEqual(fs.viewers, 0)
        finally:
            stop.set()
            server.shutdown()

    def test_video_off_returns_404(self):
        fs = FrameStore(enabled=False)
        port = _free_port()
        server = start_live_server(port, lambda: {"ok": True}, frames=fs)
        try:
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/video", timeout=3)
            self.assertEqual(cm.exception.code, 404)
            # /state는 그대로 동작
            body = urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=3).read()
            self.assertIn(b"ok", body)
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
