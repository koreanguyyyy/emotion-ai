# tools/music_tool.py
import os
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

SPOTIFY_CLIENT_ID_STR = str(SPOTIFY_CLIENT_ID)
SPOTIFY_CLIENT_SECRET_STR = str(SPOTIFY_CLIENT_SECRET)  

def get_access_token() -> str:
    response = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(SPOTIFY_CLIENT_ID_STR, SPOTIFY_CLIENT_SECRET_STR)
    )
    response.raise_for_status()
    return response.json()["access_token"]

# 감정 → 검색 키워드 매핑
EMOTION_QUERY_MAP = {
    "슬픔":   ("sad calm acoustic", 0.2, 0.3),   # (query, valence, energy)
    "분노":   ("calm peaceful relax", 0.4, 0.2),
    "공포":   ("calm soothing ambient", 0.3, 0.2),
    "혐오":   ("healing nature calm", 0.4, 0.3),
    "놀람":   ("uplifting positive", 0.7, 0.6),
    "무감정": ("focus lo-fi chill", 0.5, 0.4),
    "행복":   ("happy upbeat positive", 0.9, 0.8),
}

@tool
def recommend_music(face_emotion: str) -> str:
    """
    사용자 감정에 맞는 Spotify 음악을 추천합니다.
    face_emotion: 한국어 감정 (슬픔, 분노, 공포, 혐오, 놀람, 무감정, 행복)
    """
    try:
        token = get_access_token()
        query, valence, energy = EMOTION_QUERY_MAP.get(
            face_emotion, ("calm relax", 0.5, 0.4)
        )

        response = requests.get(
            "https://api.spotify.com/v1/search",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "q": query,
                "type": "track",
                "limit": 3,
                "market": "KR"
            }
        )
        response.raise_for_status()
        tracks = response.json()["tracks"]["items"]

        if not tracks:
            return "추천할 음악을 찾지 못했습니다."

        result = []
        for track in tracks:
            name = track["name"]
            artist = track["artists"][0]["name"]
            url = track["external_urls"]["spotify"]
            result.append(f"- {name} / {artist}\n  {url}")

        return "추천 음악:\n" + "\n".join(result)

    except Exception as e:
        return f"음악 추천 실패: {e}"
