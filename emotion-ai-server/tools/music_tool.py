# tools/music_tool.py
import os
import random
import time
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# ── 토큰 캐싱 ──────────────────────────────────────────
_token_cache = {"token": None, "expires_at": 0}

def get_access_token() -> str:
    if _token_cache["token"] and time.time() < _token_cache["expires_at"]:
        return _token_cache["token"]

    response = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(str(SPOTIFY_CLIENT_ID), str(SPOTIFY_CLIENT_SECRET)),
    )
    response.raise_for_status()
    data = response.json()
    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = time.time() + data["expires_in"] - 60
    return _token_cache["token"]


# ── 감정 → 검색 키워드 풀 ──────────────────────────────
# 매 요청마다 키워드를 랜덤 조합하여 다른 결과 유도
EMOTION_KEYWORDS = {
    "분노": {
        "moods": ["calm", "peaceful", "relaxing", "soothing", "gentle"],
        "genres": ["acoustic", "piano", "ambient", "classical", "lo-fi"],
    },
    "슬픔": {
        "moods": ["comforting", "healing", "warm", "hopeful", "uplifting"],
        "genres": ["indie", "folk", "acoustic", "soul", "R&B"],
    },
    "행복": {
        "moods": ["happy", "upbeat", "party", "energetic", "fun"],
        "genres": ["pop", "dance", "disco", "k-pop", "funk"],
    },
    "공포": {
        "moods": ["calm", "safe", "soothing", "serene", "tranquil"],
        "genres": ["ambient", "classical", "piano", "new age", "chill"],
    },
    "혐오": {
        "moods": ["healing", "fresh", "clean", "peaceful", "nature"],
        "genres": ["acoustic", "folk", "indie", "singer-songwriter", "chill"],
    },
    "놀람": {
        "moods": ["chill", "steady", "mellow", "smooth", "easy"],
        "genres": ["jazz", "bossa nova", "lo-fi", "soul", "R&B"],
    },
    "무감정": {
        "moods": ["focus", "concentration", "study", "thinking", "chill"],
        "genres": ["lo-fi", "ambient", "electronic", "jazz", "instrumental"],
    },
}

DEFAULT_EMOTION = "슬픔"


@tool
def recommend_music(face_emotion: str) -> str:
    """
    사용자 감정에 맞는 Spotify 음악을 추천합니다.
    face_emotion: 한국어 감정 (슬픔, 분노, 공포, 혐오, 놀람, 무감정, 행복)
    """
    try:
        token = get_access_token()
        keywords = EMOTION_KEYWORDS.get(face_emotion, EMOTION_KEYWORDS[DEFAULT_EMOTION])

        # 매번 다른 키워드 조합 생성
        mood = random.choice(keywords["moods"])
        genre = random.choice(keywords["genres"])
        query = f"{mood} {genre}"

        # 랜덤 offset으로 다른 페이지의 결과 가져오기
        offset = random.randint(0, 20)

        response = requests.get(
            "https://api.spotify.com/v1/search",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "q": query,
                "type": "track",
                "limit": 10,
                "offset": offset,
                "market": "KR",
            },
        )

        # Rate limit 처리
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "1"))
            time.sleep(retry_after)
            
            
            response = requests.get(
                "https://api.spotify.com/v1/search",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "q": query,
                    "type": "track",
                    "limit": 10,
                    "offset": offset,
                    "market": "KR",
                },
            )

        response.raise_for_status()

        tracks = response.json().get("tracks", {}).get("items", [])
        if not tracks:
            return "추천할 음악을 찾지 못했습니다."

        # 20개 중 랜덤 5개 선택
        picks = random.sample(tracks, min(5, len(tracks)))

        result = []
        for track in picks:
            name = track["name"]
            artist = track["artists"][0]["name"]
            url = track["external_urls"]["spotify"]
            result.append(f"- {name} / {artist}\n  {url}")

        return f"[{face_emotion}] 감정 기반 추천 음악 (검색: {query}):\n" + "\n".join(result)

    except requests.exceptions.HTTPError as e:
        return f"Spotify API 오류: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"음악 추천 실패: {e}"