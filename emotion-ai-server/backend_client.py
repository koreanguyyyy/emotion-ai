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
    챗봇 형식 추천 텍스트에서 URL 기반으로 항목 파싱 후 DB 저장
    """
    # 카테고리 키워드로 현재 카테고리 추적
    category_map = {
        "spotify": "MUSIC",
        "aladin": "BOOK",
        "kakao": "PLACE",
        "tmdb": "MOVIE",
    }

    first_recommendation_id = None
    lines = recommendation_text.split("\n")

    for i, line in enumerate(lines):
        line = line.strip()

        # URL 감지
        if line.startswith("http"):
            external_url = line

            # URL로 카테고리 판단
            category = "MUSIC"  # 기본값
            for keyword, cat in category_map.items():
                if keyword in external_url.lower():
                    category = cat
                    break

            # URL 바로 위 문장에서 제목/아티스트 추출
            title = None
            artist_or_author = None

            for j in range(i - 1, max(i - 4, -1), -1):
                prev_line = lines[j].strip()
                if not prev_line or prev_line.startswith("http"):
                    continue

                # '제목' 또는 '아티스트의 제목' 패턴 추출
                # 예) "Natalie Jane Hill의 'Never Left Me'를 추천드려요."
                import re

                # 작은따옴표로 감싼 제목 추출
                quoted = re.findall(r"'([^']+)'", prev_line)
                if quoted:
                    title = quoted[-1]

                # '~의' 패턴으로 아티스트 추출
                artist_match = re.search(r"([가-힣A-Za-z\s]+)의\s*'", prev_line)
                if artist_match:
                    artist_or_author = artist_match.group(1).strip()

                if title:
                    break

            # 제목 못 찾으면 URL 앞 문장 전체를 title로
            if not title:
                for j in range(i - 1, max(i - 4, -1), -1):
                    prev_line = lines[j].strip()
                    if prev_line and not prev_line.startswith("http"):
                        title = prev_line[:100]  # 최대 100자
                        break

            if not title:
                title = "추천 항목"

            rec_id = _save_single_recommendation(
                intervention_id=intervention_id,
                category=category,
                title=title,
                artist_or_author=artist_or_author,
                external_url=external_url,
            )

            if first_recommendation_id is None and rec_id:
                first_recommendation_id = rec_id

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