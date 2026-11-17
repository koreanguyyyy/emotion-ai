# ai/frame_store.py
"""
얼굴 인식 화면을 웹페이지로 보내기 위한 '최신 장면 보관함'.

감지 프로그램(face_fusion)이 얼굴 박스·글자를 그린 화면을 publish()로 넘기면,
실시간 상태 통로(live_server)의 GET /video 가 이를 MJPEG(연속 JPEG)로 웹페이지에 흘려보낸다.

- 보는 사람(웹페이지)이 없으면 JPEG로 바꾸지 않는다 → 평소 CPU 부담 없음
- 초당 LIVE_VIDEO_FPS장까지만, 폭 LIVE_VIDEO_WIDTH로 줄여서 보낸다
- 저장하지 않는다: 메모리에 마지막 한 장만 두고 계속 덮어쓴다

.env: LIVE_VIDEO=on|off (기본 on), LIVE_VIDEO_FPS=12, LIVE_VIDEO_WIDTH=640
"""
from __future__ import annotations

import os
import threading
import time

VIDEO_ON = os.getenv("LIVE_VIDEO", "on").strip().lower() not in ("off", "0", "false", "no")
VIDEO_FPS = max(1.0, float(os.getenv("LIVE_VIDEO_FPS", "12") or 12))
VIDEO_WIDTH = max(160, int(os.getenv("LIVE_VIDEO_WIDTH", "640") or 640))
JPEG_QUALITY = 70


class FrameStore:
    def __init__(self, fps: float = VIDEO_FPS, width: int = VIDEO_WIDTH, enabled: bool = VIDEO_ON):
        self.enabled = enabled
        self.min_gap = 1.0 / fps
        self.width = width
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._last = 0.0
        self._viewers = 0

    # ── 감지 프로그램 쪽 ──
    def publish(self, frame, now: float | None = None) -> bool:
        """화면 한 장을 넘긴다. 보는 사람이 없거나 너무 자주 오면 건너뛴다. 실패해도 예외를 내지 않는다."""
        if not self.enabled or self._viewers <= 0 or frame is None:
            return False
        now = time.monotonic() if now is None else now
        if now - self._last < self.min_gap:
            return False
        try:
            import cv2
            h, w = frame.shape[:2]
            if w > self.width:
                frame = cv2.resize(frame, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                return False
        except Exception:
            return False
        with self._cond:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self._last = now
            self._cond.notify_all()
        return True

    # ── 웹 서버 쪽 ──
    def add_viewer(self) -> None:
        with self._cond:
            self._viewers += 1

    def remove_viewer(self) -> None:
        with self._cond:
            self._viewers = max(0, self._viewers - 1)

    @property
    def viewers(self) -> int:
        return self._viewers

    def wait_next(self, after_seq: int, timeout: float = 2.0):
        """after_seq보다 새로운 장면이 오면 (seq, jpeg), 시간 안에 안 오면 None."""
        with self._cond:
            if self._seq <= after_seq:
                self._cond.wait(timeout)
            if self._jpeg is None or self._seq <= after_seq:
                return None
            return self._seq, self._jpeg

    def clear(self) -> None:
        """새 관람객 등: 이전 사람 얼굴이 남지 않게 지운다."""
        with self._cond:
            self._jpeg = None


FRAMES = FrameStore()
