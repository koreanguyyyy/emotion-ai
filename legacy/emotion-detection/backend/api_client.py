# backend/api_client.py
import os
import requests
import logging

from tenacity import retry, stop_after_attempt, wait_fixed

from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")
USER_ID = int(os.getenv("USER_ID", 1))

# ── 감정 로그 저장 ─────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
# 감정 로그 저장 (재시도 3회, 2초 간격) - 네트워크 불안정 대비

def save_emotion_log(
    face_emotion: str,
    face_confidence: float,
    speech_text: str,
    vad_stress_score: float,
    final_stress_level: int | None,
    intervention_triggered: bool = False,
    sigh_detected: bool = False,
    motion_pattern: str | None = None,
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
        "faceConfidence": round(face_confidence / 100, 4) if face_confidence > 1 else face_confidence,
        "speechText": speech_text,
        "vadStressScore": vad_stress_score,
        "finalStressLevel": final_stress_level,
        "sighDetected": sigh_detected,
        "motionPattern": motion_pattern,
        "interventionTriggered": intervention_triggered,
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/emotion-logs",
            json=payload,
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        emotion_log_id = data["data"]["emotionLogId"]
        print(f"[Backend] 감정 로그 저장 완료 (id: {emotion_log_id})")
        return emotion_log_id

    except Exception as e:
        print(f"[Backend] 감정 로그 저장 실패: {e}")
        return None

# ── 쿨다운 확인 ────────────────────────────────────────
def check_cooldown() -> tuple[bool, int]:
    """
    DB 기반 쿨다운 확인
    반환값: (쿨다운 활성 여부, 남은 시간(분))
    """
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/interventions/user/{USER_ID}/cooldown",
            timeout=5
        )
        response.raise_for_status()
        data = response.json()["data"]
        active = data["active"]
        remaining = data["remainingMinutes"]

        if active:
            print(f"[Backend] 쿨다운 중 → {remaining}분 남음")
        return active, remaining

    except Exception as e:
        print(f"[Backend] 쿨다운 확인 실패: {e} → 로컬 쿨다운 사용")
        return False, 0

# ── 개입 저장 ──────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
# 개입 저장 (재시도 3회, 2초 간격) - 네트워크 불안정 대비
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
        print(f"[Backend] 개입 저장 완료 (id: {intervention_id})")
        return intervention_id

    except Exception as e:
        print(f"[Backend] 개입 저장 실패: {e}")
        return None

# ── 추천 저장 ──────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
# 추천 저장 (재시도 3회, 2초 간격) - 네트워크 불안정 대비
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
        print(f"[Backend] 추천 저장 완료 ({category}: {title})")
        return True

    except Exception as e:
        print(f"[Backend] 추천 저장 실패: {e}")
        return False
