# backend_client.py
import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

# ── 개입 저장 ──────────────────────────────────────────
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
def save_recommendations(
    intervention_id: int,
    recommendation_text: str,
) -> None:
    """
    LLM 추천 결과 텍스트를 파싱해서 각 항목을 DB에 저장
    추천 텍스트 형식:
      추천 음악:
      - 제목 / 아티스트
        https://...

      추천 도서:
      - 제목 / 저자
        https://...
    """
    # 카테고리 판단
    category_map = {
        "음악": "MUSIC",
        "도서": "BOOK",
        "장소": "PLACE",
        "영화": "MOVIE",
    }

    current_category = None
    lines = recommendation_text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # 카테고리 헤더 감지
        for keyword, category in category_map.items():
            if keyword in line and ("추천" in line or ":" in line):
                current_category = category
                break

        # 항목 감지 (- 로 시작)
        if line.startswith("-") and current_category:
            item_line = line.lstrip("- ").strip()

            # 제목 / 아티스트(저자) 분리
            if "/" in item_line:
                parts = item_line.split("/", 1)
                title = parts[0].strip()
                artist_or_author = parts[1].strip()
            else:
                title = item_line
                artist_or_author = None

            # 다음 줄에서 URL 찾기
            external_url = None
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith("http"):
                    external_url = next_line
                    i += 1

            _save_single_recommendation(
                intervention_id=intervention_id,
                category=current_category,
                title=title,
                artist_or_author=artist_or_author,
                external_url=external_url,
            )

        i += 1

def _save_single_recommendation(
    intervention_id: int,
    category: str,
    title: str,
    artist_or_author: str | None,
    external_url: str | None,
    reason: str | None = None,
) -> None:
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
        print(f"[Backend] 추천 저장 완료 ({category}: {title})")

    except Exception as e:
        print(f"[Backend] 추천 저장 실패 ({title}): {e}")
