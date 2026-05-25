# backend_client.py
import os
import re
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

# ── 개입 저장 ──────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
# 재시도 3회, 지수 백오프 (1s → 2s → 4s)
def save_intervention(
    user_id: int,
    emotion_log_id: int,
    agent_message: str,
    cooldown_minutes: int = 30,
) -> int | None:
    """
    개입 이력 저장
    반환값: intervention_id (추천 저장 시 사용)
    """
    payload = {
        "userId": user_id,
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
        intervention_id = response.json()["data"]["interventionId"]
        print(f"[Backend] 개입 저장 완료 (id: {intervention_id})")
        return intervention_id

    except Exception as e:
        print(f"[Backend] 개입 저장 실패: {e}")
        return None

# ── 추천 파싱 및 저장 ──────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
# 재시도 3회, 지수 백오프 (1s → 2s → 4s)
def save_recommendations(
    intervention_id: int,
    recommendation_text: str,
) -> int | None:
    """
    LLM 추천 결과 텍스트를 파싱해서 각 항목을 DB에 저장
    반환값: 첫 번째 recommendation_id (피드백 제출용)
    """
    category_map = {
        "음악": "MUSIC",
        "도서": "BOOK",
        "장소": "PLACE",
        "영화": "MOVIE",
    }

    current_category = None
    first_recommendation_id = None
    lines = recommendation_text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        for keyword, category in category_map.items():
            if keyword in line and ("추천" in line or ":" in line):
                current_category = category
                break

        if line.startswith("-") and current_category:
            item_line = line.lstrip("- ").strip()

            if "/" in item_line:
                parts = item_line.split("/", 1)
                title = parts[0].strip()
                artist_or_author = parts[1].strip()
            else:
                title = item_line
                artist_or_author = None

            external_url = None
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith("http"):
                    external_url = next_line
                    i += 1

            rec_id = _save_single_recommendation(
                intervention_id=intervention_id,
                category=current_category,
                title=title,
                artist_or_author=artist_or_author,
                external_url=external_url,
            )

            # 첫 번째 recommendation_id 저장
            if first_recommendation_id is None and rec_id:
                first_recommendation_id = rec_id

        i += 1

    return first_recommendation_id

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
# 재시도 3회, 지수 백오프 (1s → 2s → 4s)
def _save_single_recommendation(
    intervention_id: int,
    category: str,
    title: str,
    artist_or_author: str | None,
    external_url: str | None,
    reason: str | None = None,
) -> int | None:
    payload = {
        "interventionId": intervention_id,
        "category": category,
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
        rec_id = response.json()["data"]["recommendationId"]
        print(f"[Backend] 추천 저장 완료 ({category}: {title})")
        return rec_id

    except Exception as e:
        print(f"[Backend] 추천 저장 실패 ({title}): {e}")
        return None

# ── 최근 개입 이력 조회 ────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
# 재시도 3회, 지수 백오프 (1s → 2s → 4s)
def get_recent_interventions(user_id: int, size: int = 3) -> list:
    """
    최근 개입 이력 조회 (감정 변화 추이 파악용)
    반환값: [{interventionId, agentMessage, createdAt, recommendations: [...]}]
    """
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/interventions/user/{user_id}",
            params={"size": size},
            timeout=5
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        print(f"[Backend] 최근 개입 이력 {len(data)}건 조회")
        return data

    except Exception as e:
        print(f"[Backend] 개입 이력 조회 실패: {e}")
        return []

# ── 최근 추천 이력 조회 ────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=4))
# 재시도 3회, 지수 백오프 (1s → 2s → 4s)
def get_recent_recommendations(user_id: int, size: int = 5) -> list:
    """
    최근 추천 이력 조회 (중복 추천 방지용)
    반환값: [{category, title, artistOrAuthor, createdAt}]
    """
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/recommendations/user/{user_id}/recent",
            params={"size": size},
            timeout=5
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        print(f"[Backend] 최근 추천 이력 {len(data)}건 조회")
        return data

    except Exception as e:
        print(f"[Backend] 추천 이력 조회 실패: {e}")
        return []