# ai/stress_calculator.py
import time
import threading

# ── VAD 테이블 ─────────────────────────────────────────
VAD_TABLE = {
    "happy":    {"valence": 0.9, "arousal": 0.7, "dominance": 0.8},
    "neutral":  {"valence": 0.5, "arousal": 0.2, "dominance": 0.5},
    "sad":      {"valence": 0.1, "arousal": 0.2, "dominance": 0.2},
    "angry":    {"valence": 0.1, "arousal": 0.9, "dominance": 0.7},
    "fear":     {"valence": 0.1, "arousal": 0.8, "dominance": 0.1},
    "disgust":  {"valence": 0.1, "arousal": 0.6, "dominance": 0.6},
    "surprise": {"valence": 0.6, "arousal": 0.8, "dominance": 0.4},
}

VAD_THRESHOLD = 6
STT_BUFFER_SEC = 30

# ── STT 버퍼 ───────────────────────────────────────────
stt_buffer = []
stt_lock = threading.Lock()

def add_stt_text(text: str):
    with stt_lock:
        stt_buffer.append({
            "text": text,
            "timestamp": time.time()
        })

def get_recent_stt() -> str:
    now = time.time()
    with stt_lock:
        recent = [s for s in stt_buffer if now - s["timestamp"] < STT_BUFFER_SEC]
        stt_buffer.clear()
        stt_buffer.extend(recent)
        return " ".join([s["text"] for s in recent])

# ── VAD 계산 (로컬 연산) ───────────────────────────────
def calculate_vad_stress(emotion_scores: dict) -> float:
    stress = 0.0
    total_weight = 0.0

    for emo, score in emotion_scores.items():
        if emo not in VAD_TABLE:
            continue
        vad = VAD_TABLE[emo]
        emo_stress = (
            (1 - vad["valence"]) * 0.5 +
            vad["arousal"] * 0.3 +
            (1 - vad["dominance"]) * 0.2
        )
        stress += emo_stress * score
        total_weight += score

    if total_weight == 0:
        return 0.0

    return round((stress / total_weight) * 10, 2)

def should_call_llm(emotion_scores: dict) -> tuple[bool, float]:
    """
    VAD 임계값 넘으면 LLM 호출 필요 여부 반환
    반환값: (호출 여부, vad_stress)
    """
    vad_stress = calculate_vad_stress(emotion_scores)
    print(f"[VAD] {vad_stress}/10")

    if vad_stress < VAD_THRESHOLD:
        print(f"[VAD] {vad_stress} < {VAD_THRESHOLD} → 스킵")
        return False, vad_stress

    return True, vad_stress
