# tools/book_tool.py
import os
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

ALADIN_TTB_KEY = os.getenv("ALADIN_TTB_KEY")

# 감정 → 검색 키워드 매핑
EMOTION_KEYWORD_MAP = {
    "슬픔":   "위로 에세이",
    "분노":   "마음 치유 심리",
    "공포":   "불안 극복 자기계발",
    "혐오":   "힐링 명상 에세이",
    "놀람":   "동기부여 긍정",
    "무감정": "철학 사색 에세이",
    "행복":   "행복 성장 에세이",
}

@tool
def recommend_book(face_emotion: str) -> str:
    """
    사용자 감정에 맞는 책을 알라딘 API로 추천합니다.
    face_emotion: 한국어 감정 (슬픔, 분노, 공포, 혐오, 놀람, 무감정, 행복)
    """
    try:
        keyword = EMOTION_KEYWORD_MAP.get(face_emotion, "힐링 에세이")

        response = requests.get(
            "http://www.aladin.co.kr/ttb/api/ItemSearch.aspx",
            params={
                "ttbkey": ALADIN_TTB_KEY,
                "Query": keyword,
                "QueryType": "Keyword",
                "MaxResults": 3,
                "SearchTarget": "Book",
                "output": "js",
                "Version": "20131101"
            }
        )
        response.raise_for_status()
        data = response.json()
        items = data.get("item", [])

        if not items:
            return "추천할 책을 찾지 못했습니다."

        result = []
        for item in items:
            title = item["title"]
            author = item["author"]
            url = item["link"]
            result.append(f"- {title} / {author}\n  {url}")

        return "추천 도서:\n" + "\n".join(result)

    except Exception as e:
        return f"책 추천 실패: {e}"
