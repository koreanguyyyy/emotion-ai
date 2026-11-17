# ai/stt_buffer.py
# 최근 30초 발화 버퍼. 1학기 ai/stress_calculator.py에서 STT 버퍼 부분만 옮겨 왔다
# (DeepFace 확률 기반 VAD 계산은 이 버전에서 ai/stress_index.py로 대체).
import threading
import time

import os
from collections import deque
from datetime import datetime

STT_BUFFER_SEC = 30
# 에이전트가 말을 걸 때 함께 보내는 '최근 몇 분 발화' (2026-11-04). 30초만 보면 앞뒤 사정을 모른다.
# 평소 켜 두고 쓰는 경우를 생각해 30분. 그보다 오래된 말은 30분마다 요약해 서버 기억에 쌓는다(main.py).
SPEECH_HISTORY_SEC = float((os.getenv("SPEECH_HISTORY_SEC") or "1800").split("#")[0].strip() or 1800)

_stt_buffer: list[dict] = []
_history: deque = deque(maxlen=300)
_stt_lock = threading.Lock()


def add_stt_text(text: str, in_chat: bool = False) -> None:
    """in_chat: 에이전트와 대화 중에 한 말 (대화 요약에 이미 들어가므로 혼잣말 요약에서는 뺀다)."""
    now = time.time()
    with _stt_lock:
        _stt_buffer.append({"text": text, "timestamp": now})
        _history.append({"text": text, "timestamp": now, "in_chat": in_chat})


def clear_stt() -> None:
    with _stt_lock:
        _stt_buffer.clear()
        _history.clear()


def get_speech_since(t: float) -> list[dict]:
    """t 이후 대화 밖에서 한 말 [{time, text, timestamp}] (혼잣말 요약용)."""
    with _stt_lock:
        return [{"time": datetime.fromtimestamp(h["timestamp"]).strftime("%m-%d %H:%M"), "text": h["text"],
                 "timestamp": h["timestamp"]} for h in _history if h["timestamp"] > t and not h.get("in_chat")]


def get_speech_history(sec: float | None = None) -> list[dict]:
    """최근 sec초(기본 SPEECH_HISTORY_SEC) 발화 [{time:'HH:MM:SS', text}] — 지우지 않는다."""
    cutoff = time.time() - (SPEECH_HISTORY_SEC if sec is None else sec)
    with _stt_lock:
        return [{"time": datetime.fromtimestamp(h["timestamp"]).strftime("%H:%M:%S"), "text": h["text"]}
                for h in _history if h["timestamp"] >= cutoff]


def get_recent_stt() -> str:
    now = time.time()
    with _stt_lock:
        recent = [s for s in _stt_buffer if now - s["timestamp"] < STT_BUFFER_SEC]
        _stt_buffer.clear()
        _stt_buffer.extend(recent)
        return " ".join(s["text"] for s in recent)
