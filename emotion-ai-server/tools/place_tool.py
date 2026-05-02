# tools/place_tool.py
import os
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")

# 감정 → 장소 카테고리 매핑
EMOTION_PLACE_MAP = {
    "슬픔":   ("카페", "CE7"),       # 조용한 카페
    "분노":   ("공원", "AT4"),       # 자연 속 산책
    "공포":   ("카페", "CE7"),       # 따뜻한 실내
    "혐오":   ("공원", "AT4"),       # 공기 좋은 곳
    "놀람":   ("문화시설", "CT1"),   # 전시/문화
    "무감정": ("도서관", "PO3"),     # 조용한 공간
    "행복":   ("음식점", "FD6"),     # 맛있는 음식
}

@tool
def recommend_place(face_emotion: str) -> str:
    """
    사용자 감정에 맞는 주변 장소를 카카오 로컬 API로 추천합니다.
    face_emotion: 한국어 감정 (슬픔, 분노, 공포, 혐오, 놀람, 무감정, 행복)
    """
    try:
        keyword, category_code = EMOTION_PLACE_MAP.get(
            face_emotion, ("카페", "CE7")
        )

        response = requests.get(
            "https://dapi.kakao.com/v2/local/search/category.json",
            headers={"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"},
            params={
                "category_group_code": category_code,
                "radius": 1000,        # 1km 반경
                "size": 3,
                "sort": "accuracy",
                # 위치는 추후 사용자 위치로 교체
                # "x": "126.9784",      # 기본값: 서울 중심
                # "y": "37.5665"
                
                "x": "126.897543822094",
                "y": "35.2097395400161",
            }
        )
        response.raise_for_status()
        places = response.json().get("documents", [])

        if not places:
            return "추천할 장소를 찾지 못했습니다."

        result = []
        for place in places:
            name = place["place_name"]
            address = place["road_address_name"] or place["address_name"]
            url = place["place_url"]
            result.append(f"- {name} ({address})\n  {url}")

        return f"감정 '{face_emotion}'에 맞는 {keyword} 추천:\n" + "\n".join(result)

    except Exception as e:
        return f"장소 추천 실패: {e}"
