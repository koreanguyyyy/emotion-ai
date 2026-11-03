# ai/live_server.py
"""
실시간 상태 통로 — 웹페이지(capstone-frontend)의 '실시간 모니터링' 화면이 1초마다 읽는 작은 HTTP 서버.

이전에는 웹페이지가 백엔드의 감정 기록만 읽었는데, 기록은 스트레스 지수가 기준(6)을 넘을 때만 저장돼서
화면이 에이전트가 말을 걸 때만 바뀌었다(실시간이 아니었음). 감정 기록을 매초 DB에 쌓는 대신,
감지 프로그램이 지금 상태를 이 통로로 바로 보여준다.

  GET  http://127.0.0.1:8765/state       → 현재 표정·심박·스트레스 지수·모드 + 대화(chat) JSON
  POST http://127.0.0.1:8765/chat        {"message": "..."} → 웹 채팅으로 대답 (답은 /state의 chat에 나타남)
  POST http://127.0.0.1:8765/chat/start  → 개입 없이 대화 시작
  POST http://127.0.0.1:8765/chat/end    → 대화 끝내기
  .env: LIVE_PORT=8765 (0이면 끔)

보안: 이 컴퓨터 안(127.0.0.1)에서만 열린다. 영상·음성은 보내지 않고 계산된 숫자만 보낸다.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EMOTION_KEY = {"행복": "happy", "슬픔": "sad", "분노": "angry", "공포": "fear",
               "혐오": "disgust", "놀람": "surprise", "무감정": "neutral"}


def _num(v, nd=3):
    try:
        if v is None:
            return None
        f = float(v)
        return None if f != f else round(f, nd)
    except (TypeError, ValueError):
        return None


def build_live_payload(state: dict, reading, chat: dict | None = None) -> dict:
    """세션 상태와 최근 스트레스 계산 결과(+ 대화 상태)를 웹페이지용 JSON으로 만든다."""
    label = state.get("face_emotion") or "무감정"
    return {
        "chat": chat,
        "time": time.time(),
        "faceDetected": bool(state.get("face_detected")),
        "faceEmotion": label,
        "emotionKey": EMOTION_KEY.get(label, "neutral"),
        "faceConfidence": _num(state.get("face_confidence")),
        "faceBaselineReady": bool(state.get("face_baseline_ready")),
        "faceMethod": state.get("face_method"),
        "modelEmotion": state.get("model_label"),
        "valence": _num(state.get("fused_valence")),
        "arousal": _num(state.get("fused_arousal")),
        "heartRate": _num(state.get("hr_bpm"), 1),
        "heartRateQuality": _num(state.get("hr_quality")),
        "restingHeartRate": _num(state.get("resting_hr"), 1),
        "hrMode": state.get("hr_mode"),
        "hrSource": state.get("hr_source"),
        "watchHeartRate": _num(state.get("watch_hr_bpm"), 1),
        "webcamHeartRate": _num(state.get("cam_hr_bpm"), 1),
        "stressIndex": _num(getattr(reading, "index", None), 2),
        "stressNeg": _num(getattr(reading, "neg", None)),
        "stressHr": _num(getattr(reading, "hr_up", None)),
        "stressArousal": _num(getattr(reading, "arousal", None)),
        "stressInstab": _num(getattr(reading, "instab", None)),
        "agentStressLevel": state.get("stress_level"),
        "speechText": state.get("stt_text"),
        "lastInterventionId": state.get("last_intervention_id"),
    }


MAX_POST_BYTES = 8 * 1024


def start_live_server(port: int, get_payload, post_handlers: dict | None = None) -> ThreadingHTTPServer | None:
    """get_payload(): 호출할 때마다 최신 상태 dict를 돌려주는 함수.
    post_handlers: {경로: 함수(body dict) → dict} — 웹 채팅 등. 실패하면 경고만 하고 None."""
    post_handlers = post_handlers or {}
    if not port:
        return None

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes = b"") -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")       # 웹페이지(다른 포트)에서 읽을 수 있게
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_OPTIONS(self):  # noqa: N802
            self._send(204)

        def do_GET(self):  # noqa: N802
            if self.path.split("?")[0] != "/state":
                self._send(404, b'{"error":"not found"}')
                return
            try:
                body = json.dumps(get_payload(), ensure_ascii=False, default=float).encode("utf-8")
                self._send(200, body)
            except Exception as e:   # 상태 읽기 실패가 감지 프로그램을 멈추면 안 된다
                self._send(500, json.dumps({"error": str(e)}).encode("utf-8"))

        def do_POST(self):  # noqa: N802
            handler = post_handlers.get(self.path.split("?")[0])
            if handler is None:
                self._send(404, b'{"error":"not found"}')
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_POST_BYTES:
                    self._send(413, b'{"error":"too large"}')
                    return
                raw = self.rfile.read(n) if n else b""
                body = json.loads(raw.decode("utf-8")) if raw else {}
                if not isinstance(body, dict):
                    raise ValueError("JSON object required")
                out = handler(body) or {"ok": True}
                self._send(200, json.dumps(out, ensure_ascii=False, default=float).encode("utf-8"))
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"))

        def log_message(self, *args):   # 1초마다 접속 기록이 터미널에 찍히지 않게
            pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        print(f"[Live] 실시간 상태 통로를 열지 못했습니다 (포트 {port}): {e}")
        return None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[Live] 실시간 상태: http://127.0.0.1:{port}/state")
    return server
