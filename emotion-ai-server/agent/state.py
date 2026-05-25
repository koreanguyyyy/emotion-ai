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