# state/emotion_state.py
import threading

state_lock = threading.Lock()
stop_event = threading.Event()

emotion_state = {
    "face_emotion": None,       # 한국어 감정
    "emotion_scores": None,     # DeepFace 전체 확률값 (VAD에 사용)
    "stt_text": None,           # 최신 STT 텍스트
    "stress_level": None,       # 최종 스트레스 (0~10 or None)
}

def update_state(**kwargs):
    with state_lock:
        for key, value in kwargs.items():
            if key in emotion_state:
                emotion_state[key] = value

def get_state():
    with state_lock:
        return emotion_state.copy()
