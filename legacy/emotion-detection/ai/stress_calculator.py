# ai/stress_calculator.py
import time
import threading
import json
import requests

# ── VAD 테이블 ─────────────────────────────────────────
# Valence: 긍정(1.0) ↔ 부정(0.0)
# Arousal: 각성(1.0) ↔ 차분(0.0)
# Dominance: 통제감(1.0) ↔ 무력감(0.0)
# 참고: VA 값은 AffectNet(Mollahosseini et al., 2017) 기반
#       Dominance 값은 Mehrabian PAD 모델(1974) 참고
VAD_TABLE = {
    "happy":    {"valence": 0.9, "arousal": 0.7, "dominance": 0.8},
    "neutral":  {"valence": 0.5, "arousal": 0.2, "dominance": 0.5},
    "sad":      {"valence": 0.1, "arousal": 0.2, "dominance": 0.2},
    "angry":    {"valence": 0.1, "arousal": 0.9, "dominance": 0.7},
    "fear":     {"valence": 0.1, "arousal": 0.8, "dominance": 0.1},
    "disgust":  {"valence": 0.1, "arousal": 0.6, "dominance": 0.6},
    "surprise": {"valence": 0.6, "arousal": 0.8, "dominance": 0.4},
}

# ── 설정값 ─────────────────────────────────────────────
VAD_THRESHOLD = 6                # 최종 스트레스 점수 임계값
STT_BUFFER_SEC = 30              # STT 텍스트 버퍼 유지 시간(초)
COOLDOWN_SEC = 1800              # 개입 쿨다운 (30분)
FACE_WEIGHT = 0.5                # 표정 스트레스 가중치
TEXT_WEIGHT = 0.5                # 텍스트 감성 스트레스 가중치
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# ── STT 버퍼 ───────────────────────────────────────────
stt_buffer = []
stt_lock = threading.Lock()

# ── 쿨다운 관리 ────────────────────────────────────────
last_intervention_time = 0
cooldown_lock = threading.Lock()


def add_stt_text(text: str):
    """STT에서 변환된 텍스트를 버퍼에 추가"""
    with stt_lock:
        stt_buffer.append({
            "text": text,
            "timestamp": time.time()
        })


def get_recent_stt() -> str:
    """최근 30초 이내의 STT 텍스트를 합쳐서 반환"""
    now = time.time()
    with stt_lock:
        recent = [s for s in stt_buffer if now - s["timestamp"] < STT_BUFFER_SEC]
        stt_buffer.clear()
        stt_buffer.extend(recent)
        return " ".join([s["text"] for s in recent])


# ── 쿨다운 체크 ────────────────────────────────────────
def check_cooldown() -> bool:
    """쿨다운 시간이 지났는지 확인. True면 개입 가능"""
    with cooldown_lock:
        elapsed = time.time() - last_intervention_time
        if elapsed < COOLDOWN_SEC:
            remaining = int(COOLDOWN_SEC - elapsed)
            print(f"[쿨다운] {remaining}초 남음 → 개입 보류")
            return False
        return True


def update_cooldown():
    """개입 실행 시 쿨다운 타이머 갱신"""
    global last_intervention_time
    with cooldown_lock:
        last_intervention_time = time.time()
        print("[쿨다운] 타이머 리셋 (30분)")


# ── 1단계: 표정 VAD 스트레스 계산 (로컬 연산) ──────────
def calculate_face_stress(emotion_scores: dict) -> float:
    """
    DeepFace 감정 확률 → VAD 변환 → 스트레스 점수 (0~10)

    공식: emo_stress = (1 - valence) * 0.5 + arousal * 0.3 + (1 - dominance) * 0.2
    각 감정의 스트레스를 감정 확률로 가중 평균 후 10배
    """
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

    result = round((stress / total_weight) * 10, 2)
    print(f"[표정 스트레스] {result}/10")
    return result


# ── 2단계: 텍스트 감성 스트레스 계산 (Ollama 호출) ─────
def calculate_text_stress(stt_text: str) -> float:
    """
    STT 텍스트 → Ollama 감성 분석 → 스트레스 점수 (0~10)

    Ollama에게 sentiment(positive/negative/neutral)와
    intensity(0.0~1.0)를 JSON으로 받아서 스트레스 점수로 변환

    변환 규칙:
    - negative: intensity * 10 (예: 0.8 → 8.0)
    - neutral: 3.0 (기본값)
    - positive: (1 - intensity) * 3 (예: 0.7 → 0.9)
    """
    if not stt_text or stt_text.strip() == "":
        print("[텍스트 스트레스] 텍스트 없음 → 기본값 3.0/10")
        return 3.0  # 텍스트 없으면 neutral 기본값

    prompt = f"""다음 텍스트의 감성을 분석하세요.

텍스트: "{stt_text}"

반드시 아래 JSON 형식으로만 응답하세요. 다른 말은 하지 마세요.
{{"sentiment": "positive 또는 negative 또는 neutral", "intensity": 0.0에서 1.0 사이 숫자}}"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=10
        )
        response.raise_for_status()

        result_text = response.json().get("response", "").strip()
        print(f"[Ollama 응답] {result_text}")

        # JSON 파싱 (```json ... ``` 감싸기 제거)
        clean_text = result_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_text)

        sentiment = result.get("sentiment", "neutral").lower()
        intensity = float(result.get("intensity", 0.5))
        intensity = max(0.0, min(1.0, intensity))  # 0~1 클램핑

        # 스트레스 점수 변환
        if sentiment == "negative":
            text_stress = intensity * 10
        elif sentiment == "positive":
            text_stress = (1 - intensity) * 3
        else:  # neutral
            text_stress = 3.0

        text_stress = round(text_stress, 2)
        print(f"[텍스트 스트레스] sentiment={sentiment}, intensity={intensity} → {text_stress}/10")
        return text_stress

    except requests.exceptions.Timeout:
        print("[텍스트 스트레스] Ollama 타임아웃 → 기본값 3.0/10")
        return 3.0
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[텍스트 스트레스] 파싱 실패: {e} → 기본값 3.0/10")
        return 3.0
    except requests.exceptions.ConnectionError:
        print("[텍스트 스트레스] Ollama 연결 실패 → 기본값 3.0/10")
        return 3.0


# ── 3단계: 최종 스트레스 판단 ──────────────────────────
def calculate_final_stress(emotion_scores: dict) -> tuple[bool, float, dict]:
    """
    표정 + 텍스트 가중 평균으로 최종 스트레스 점수 산출

    반환값: (개입 여부, 최종 스트레스 점수, 상세 정보)
    """
    # 1단계: 표정 스트레스
    face_stress = calculate_face_stress(emotion_scores)

    # 2단계: 텍스트 감성 스트레스
    stt_text = get_recent_stt()
    text_stress = calculate_text_stress(stt_text)

    # 3단계: 가중 평균
    final_stress = round(face_stress * FACE_WEIGHT + text_stress * TEXT_WEIGHT, 2)
    print(f"[최종 스트레스] 표정({face_stress}) × {FACE_WEIGHT} + 텍스트({text_stress}) × {TEXT_WEIGHT} = {final_stress}/10")

    # 상세 정보
    detail = {
        "face_stress": face_stress,
        "text_stress": text_stress,
        "stt_text": stt_text,
        "final_stress": final_stress,
        "threshold": VAD_THRESHOLD,
    }

    # 4단계: 임계값 + 쿨다운 체크
    if final_stress < VAD_THRESHOLD:
        print(f"[판단] {final_stress} < {VAD_THRESHOLD} → 개입 불필요")
        return False, final_stress, detail

    if not check_cooldown():
        detail["skipped_reason"] = "cooldown"
        return False, final_stress, detail

    # 개입 실행
    update_cooldown()
    print(f"[판단] {final_stress} >= {VAD_THRESHOLD} → LLM 추천 호출!")
    return True, final_stress, detail


# ── 하위 호환용 (기존 코드에서 호출하던 함수) ──────────
def should_call_llm(emotion_scores: dict) -> tuple[bool, float]:
    """
    기존 코드 호환용 래퍼 함수
    반환값: (호출 여부, 최종 스트레스 점수)
    """
    should_intervene, final_stress, detail = calculate_final_stress(emotion_scores)
    return should_intervene, final_stress