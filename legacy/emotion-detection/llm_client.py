# ai/llm_client.py
import requests

AI_SERVER_URL = "http://localhost:8000"  # 서버 배포 시 URL 변경

def request_llm(
    vad_stress: float,
    stt_text: str,
    face_emotion: str
) -> dict | None:
    """
    AI 서버로 LLM 보정 + 추천 요청
    반환값:
    - {"stress_level": int, "recommendation": str}
    - None → 서버 오류 시
    """
    payload = {
        "vad_stress": float(vad_stress),
        "stt_text": stt_text,
        "face_emotion": face_emotion,
    }

    try:
        response = requests.post(
            f"{AI_SERVER_URL}/analyze",
            json=payload,
            timeout=10  # 10초 타임아웃
        )
        response.raise_for_status()
        result = response.json()
        print(f"[LLM Client] 스트레스: {result['stress_level']}/10")
        return result

    except requests.Timeout:
        print("[LLM Client] 타임아웃 → VAD 값 사용")
        return None

    except Exception as e:
        print(f"[LLM Client] 서버 오류: {e} → VAD 값 사용")
        return None
