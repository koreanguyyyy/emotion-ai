# backend/api_client.py
import os
import requests
import logging

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/api_client.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
USER_ID = int(os.getenv("USER_ID", 1))

# ── 감정 로그 저장 ─────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2)) 
# 네트워크 오류 시 최대 3회, 2초 간격으로 재시도
def save_emotion_log(
    face_emotion: str,
    face_confidence: float,
    speech_text: str,
    vad_stress_score: float,
    final_stress_level: int | None,
    intervention_triggered: bool = False,
    sigh_detected: bool = False,
    motion_pattern: str | None = None,
    extra: dict | None = None,
) -> int | None:
    """
    capstone-detection → Spring Boot 감정 로그 저장
    반환값: emotion_log_id (추후 intervention 저장 시 사용)
    """
    payload = {
        "userId": USER_ID,
        "primaryEmotion": face_emotion,
        "confidenceScore": round(face_confidence / 100, 4) if face_confidence > 1 else face_confidence,
        "faceEmotion": face_emotion,
        "faceConfidence": round(face_confidence / 100, 4) if face_confidence > 1 else face_confidence,  # float로 전달
        "speechText": speech_text,
        "vadStressScore": vad_stress_score, # float로 전달
        "finalStressLevel": final_stress_level,
        "sighDetected": sigh_detected,
        "motionPattern": motion_pattern,
        "interventionTriggered": intervention_triggered,
    }
    # fusion 버전 추가 정보 (heartRate, restingHeartRate, stressNeg/Hr/Arousal/Instab, hrMode, faceMethod,
    # modelEmotion, logType). 백엔드가 이 항목을 모르는 예전 버전이어도 무시될 뿐 저장은 된다.
    if extra:
        payload.update({k: v for k, v in extra.items() if v is not None})

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/emotion-logs",
            json=payload,
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        emotion_log_id = data["data"]["emotionLogId"]
        logger.info(f"[Backend] 감정 로그 저장 완료 (id: {emotion_log_id})")
        return emotion_log_id

    except Exception as e:
        logger.error(f"[Backend] 감정 로그 저장 실패: {e}")
        return None

# ── 쿨다운 확인 ────────────────────────────────────────
def check_cooldown() -> tuple[bool | None, int]:
    """
    DB 기반 쿨다운 확인
    반환값: (쿨다운 활성 여부, 남은 시간(분))
    백엔드 조회 실패 시 (None, 0) → 호출부에서 로컬 쿨다운으로 대체
    """
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/interventions/user/{USER_ID}/cooldown",
            timeout=5
        )
        response.raise_for_status()
        data = response.json()["data"]
        # 백엔드는 {"cooldownActive": bool}만 반환 (남은 시간 없음)
        active = bool(data.get("cooldownActive", data.get("active", False)))
        remaining = int(data.get("remainingMinutes", 0))

        if active:
            logger.info(f"[Backend] 쿨다운 중 → {remaining}분 남음")
        return active, remaining

    except Exception as e:
        logger.error(f"[Backend] 쿨다운 확인 실패: {e} → 로컬 쿨다운 사용")
        return None, 0

# ── 개입 저장 ──────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
# 네트워크 오류 시 최대 3회, 2초 간격으로 재시도
def save_intervention(
    emotion_log_id: int,
    agent_message: str,
    cooldown_minutes: int = 30,
) -> int | None:
    """
    개입 이력 저장
    반환값: intervention_id (추천 저장 시 사용)
    """
    payload = {
        "userId": USER_ID,
        "emotionLogId": emotion_log_id,
        "interventionType": "proactive",
        "agentMessage": agent_message,
        "ttsPlayed": False,
        "cooldownMinutes": cooldown_minutes,
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/interventions",
            json=payload,
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        intervention_id = data["data"]["interventionId"]
        logger.info(f"[Backend] 개입 저장 완료 (id: {intervention_id})")
        return intervention_id

    except Exception as e:
        logger.error(f"[Backend] 개입 저장 실패: {e}")
        return None

# ── 추천 저장 ──────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
# 네트워크 오류 시 최대 3회, 2초 간격으로 재시도
def save_recommendation(
    intervention_id: int,
    category: str,
    title: str,
    artist_or_author: str | None = None,
    external_url: str | None = None,
    reason: str | None = None,
) -> bool:
    """
    추천 항목 저장
    category: MUSIC / BOOK / PLACE / MOVIE
    """
    payload = {
        "interventionId": intervention_id,
        "category": category.upper(),
        "title": title,
        "artistOrAuthor": artist_or_author,
        "externalUrl": external_url,
        "reason": reason,
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/recommendations",
            json=payload,
            timeout=5
        )
        response.raise_for_status()
        logger.info(f"[Backend] 추천 저장 완료 ({category}: {title})")
        return True

    except Exception as e:
        logger.error(f"[Backend] 추천 저장 실패: {e}")
        return False

# ── 개입 피드백 제출 ───────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
# 네트워크 오류 시 최대 3회, 2초 간격으로 재시도
def submit_intervention_feedback(
    intervention_id: int,
    intervention_rating: str,        # HELPFUL / NEUTRAL / UNNECESSARY / ANNOYING
    emotion_accuracy: str = "CORRECT",  # CORRECT / PARTIALLY_CORRECT / WRONG
    feedback_text: str | None = None,
) -> bool:
    payload = {
        "interventionRating": intervention_rating,
        "emotionAccuracy": emotion_accuracy,
        "feedbackText": feedback_text,
    }
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/interventions/{intervention_id}/feedback",
            json=payload,
            timeout=5
        )
        response.raise_for_status()
        logger.info(f"[Backend] 개입 피드백 저장 완료 ({intervention_rating})")
        return True
    except Exception as e:
        logger.error(f"[Backend] 개입 피드백 저장 실패: {e}")
        return False

# ── 추천 피드백 제출 ───────────────────────────────────
def submit_recommendation_feedback(
    recommendation_id: int,
    user_feedback: str,              # LIKED / DISLIKED / SKIPPED
    relevance_rating: str = "RELEVANT",  # PERFECT / RELEVANT / IRRELEVANT
    played: bool = False,
    feedback_text: str | None = None,
) -> bool:
    payload = {
        "userFeedback": user_feedback,
        "relevanceRating": relevance_rating,
        "played": played,
        "feedbackText": feedback_text,
    }
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/recommendations/{recommendation_id}/feedback",
            json=payload,
            timeout=5
        )
        response.raise_for_status()
        logger.info(f"[Backend] 추천 피드백 저장 완료 ({user_feedback})")
        return True
    except Exception as e:
        logger.error(f"[Backend] 추천 피드백 저장 실패: {e}")
        return False