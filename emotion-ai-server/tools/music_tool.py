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
            return fallback_music(face_emotion)

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
        print(f"[Music] Spotify API 오류 {e.response.status_code} → 기본 곡 목록 사용")
        return fallback_music(face_emotion)
    except Exception as e:
        print(f"[Music] 음악 추천 실패: {e} → 기본 곡 목록 사용")
        return fallback_music(face_emotion)


# ── Spotify를 못 쓸 때 쓰는 기본 곡 목록 (2026-11-04) ──────────────
# Spotify 검색 API가 앱 소유자의 프리미엄 구독을 요구하게 바뀌어(403) 추천이 실패했다.
# 감정별로 실제 곡을 골라 두고, 링크는 Spotify 검색 주소로 준다(곡 ID를 지어내지 않기 위해).
FALLBACK_TRACKS = {
    "분노": [("Clair de Lune", "Claude Debussy"), ("Weightless", "Marconi Union"), ("River Flows in You", "Yiruma"),
            ("Gymnopédie No.1", "Erik Satie"), ("밤편지", "아이유")],
    "슬픔": [("Fix You", "Coldplay"), ("Lean on Me", "Bill Withers"), ("수고했어, 오늘도", "옥상달빛"),
            ("Here Comes the Sun", "The Beatles"), ("위로", "권진아")],
    "공포": [("Weightless", "Marconi Union"), ("Clair de Lune", "Claude Debussy"), ("Kiss the Rain", "Yiruma"),
            ("Experience", "Ludovico Einaudi"), ("Holocene", "Bon Iver")],
    "혐오": [("Banana Pancakes", "Jack Johnson"), ("Bubbly", "Colbie Caillat"), ("봄봄봄", "로이킴"),
            ("Better Together", "Jack Johnson"), ("Riptide", "Vance Joy")],
    "놀람": [("Sunday Morning", "Maroon 5"), ("Fly Me to the Moon", "Frank Sinatra"), ("Put Your Records On", "Corinne Bailey Rae"),
            ("Valerie", "Amy Winehouse"), ("좋은 날", "아이유")],
    "무감정": [("Nuvole Bianche", "Ludovico Einaudi"), ("Experience", "Ludovico Einaudi"), ("Snowman", "Sia"),
             ("Coffee", "beabadoobee"), ("Blueming", "아이유")],
    "행복": [("Happy", "Pharrell Williams"), ("Walking on Sunshine", "Katrina and the Waves"), ("Dynamite", "BTS"),
            ("Good Day", "아이유"), ("Can't Stop the Feeling!", "Justin Timberlake")],
}


def fallback_music(face_emotion: str) -> str:
    from urllib.parse import quote
    tracks = FALLBACK_TRACKS.get(face_emotion, FALLBACK_TRACKS["무감정"])
    picks = random.sample(tracks, min(3, len(tracks)))
    lines = [f"- {name} / {artist}\n  https://open.spotify.com/search/{quote(name + ' ' + artist)}" for name, artist in picks]
    return f"[{face_emotion}] 감정 기반 추천 음악 (기본 목록):\n" + "\n".join(lines)