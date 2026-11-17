# ai/llm_client.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

AI_SERVER_URL = os.getenv("AI_SERVER_URL", "http://localhost:8000")  # .env에서 읽기
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", 30))
AI_TIMEOUT_SEC = float(os.getenv("AI_TIMEOUT_SEC", 40))
CHAT_TIMEOUT_SEC = float(os.getenv("CHAT_TIMEOUT_SEC", 40))   # 도구를 다시 부르면 오래 걸린다


def request_chat_stream(payload: dict):
    """이어지는 대화 한 턴을 조각으로 받는다 (에이전트 서버 POST /chat/stream, NDJSON).
    {"type":"delta","text":...} … {"type":"done",...}. 서버가 옛 버전이면(404) 예외 → 호출한 쪽이 /chat으로 다시 받는다."""
    import json
    with requests.post(f"{AI_SERVER_URL}/chat/stream", json=payload, stream=True,
                       timeout=(5, CHAT_TIMEOUT_SEC)) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line:
                yield json.loads(line)


def request_speech_memory(memory_key: str, items: list[dict]) -> bool:
    """대화 밖 발화를 서버에 보내 요약·기억하게 한다 (평소 켜 두고 쓸 때, 30분마다)."""
    try:
        r = requests.post(f"{AI_SERVER_URL}/memory/speech", timeout=5,
                          json={"memory_key": memory_key, "items": [{"time": x["time"], "text": x["text"]} for x in items]})
        return r.ok and r.json().get("accepted", False)
    except Exception as e:
        print(f"[LLM Client] 혼잣말 요약 요청 실패: {e}")
        return False


def request_chat_end(session_id: str) -> None:
    """대화가 끝났다고 알린다 → 서버가 요약해 기억에 남긴다. 실패해도 무시(옛 서버면 404)."""
    try:
        requests.post(f"{AI_SERVER_URL}/chat/end", json={"session_id": session_id}, timeout=5)
    except Exception as e:
        print(f"[LLM Client] 대화 종료 알림 실패: {e}")


def request_chat(payload: dict) -> dict | None:
    """이어지는 대화 한 턴 (에이전트 서버 POST /chat). 실패하면 None."""
    try:
        response = requests.post(f"{AI_SERVER_URL}/chat", json=payload, timeout=CHAT_TIMEOUT_SEC)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        print("[LLM Client] 대화 응답 타임아웃")
        return None
    except Exception as e:
        print(f"[LLM Client] 대화 서버 오류: {e}")
        return None

def request_llm(
    vad_stress: float,
    stt_text: str,
    face_emotion: str,
    user_id: int,
    emotion_log_id: int,
    force: bool = False,
    memory_key: str | None = None,
    speech_history: list | None = None,
) -> dict | None:
    payload = {
        "vad_stress": float(vad_stress),
        "stt_text": stt_text,
        "face_emotion": face_emotion,
        "user_id": user_id,
        "emotion_log_id": emotion_log_id,
        "cooldown_minutes": COOLDOWN_MINUTES,  # .env에서 읽기
        "force": force,   # 시연용 '지금 말 걸기': 서버의 개입 기준(INTERVENE_THRESHOLD)을 건너뜀
        "memory_key": memory_key,           # 지난 대화 기억 공간
        "speech_history": speech_history,   # 최근 몇 분 발화
    }

    try:
        response = requests.post(
            f"{AI_SERVER_URL}/analyze",
            json=payload,
            timeout=AI_TIMEOUT_SEC   # 추천 도구 + 대화체 변환까지 하면 10초를 넘기는 경우가 있다
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