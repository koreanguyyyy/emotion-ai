# agent/state.py
from typing import TypedDict

class AgentState(TypedDict):
    vad_stress: float
    stt_text: str
    face_emotion: str
    stress_level: int
    recommendation: str
