# agent/state.py
from typing import TypedDict

class AgentState(TypedDict):
    vad_stress: float
    stt_text: str
    face_emotion: str
    stress_level: int
    recommendation: str
    user_id: int           # 메모리 조회용
    emotion_log_id: int    # 개입 저장용
    force: bool            # 시연용: 개입 기준을 건너뜀 (2026-11-04)
    memory_key: str        # 지난 대화 기억 공간
    speech_history: list   # 최근 몇 분 발화 [{time, text}]