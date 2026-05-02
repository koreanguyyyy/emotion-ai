# main.py
import threading
import time
import os
from dotenv import load_dotenv
from state.emotion_state import get_state, update_state, stop_event
from detection.face_detector import run_face_detection
from detection.stt_detector import run_stt
from ai.stress_calculator import should_call_llm, get_recent_stt
from ai.llm_client import request_llm
from backend.api_client import check_cooldown, save_emotion_log, save_intervention

load_dotenv()
DETECTION_INTERVAL = int(os.getenv("DETECTION_INTERVAL", 5))

def run_stress_calculator():
    print("[Stress] 스트레스 계산 시작...")
    while not stop_event.is_set():
        time.sleep(DETECTION_INTERVAL)

        state = get_state()
        emotion_scores = state["emotion_scores"]
        face_emotion = state["face_emotion"] or "무감정"

        if not emotion_scores:
            continue

        # 1단계: VAD 계산 (로컬)
        need_llm, vad_stress = should_call_llm(emotion_scores)
        if not need_llm:
            continue

        # 2단계: DB 기반 쿨다운 체크
        is_active, remaining = check_cooldown()
        if is_active:
            print(f"[Main] 쿨다운 중 → {remaining}분 남음")
            continue

        # 3단계: 감정 로그 저장
        stt_text = get_recent_stt()
        emotion_log_id = save_emotion_log(
            face_emotion=face_emotion,
            face_confidence=max(emotion_scores.values()),
            speech_text=stt_text,
            vad_stress_score=vad_stress,
            final_stress_level=None,         # LLM 보정 후 업데이트
            intervention_triggered=False,    # 개입 확정 후 업데이트
        )

        # 4단계: AI 서버로 LLM 요청
        result = request_llm(
            vad_stress=vad_stress,
            stt_text=stt_text,
            face_emotion=face_emotion,
        )

        # 서버 오류 시 VAD 값으로 fallback
        if result is None:
            stress_level = round(vad_stress)
            recommendation = None
        else:
            stress_level = result["stress_level"]
            recommendation = result["recommendation"]

        update_state(stress_level=stress_level)

        # 5단계: 개입 발생 시 DB 저장
        if recommendation and emotion_log_id:
            print(f"\n[추천 결과]\n{recommendation}\n")
            cooldown_minutes = int(os.getenv("COOLDOWN_MINUTES", 30))
            save_intervention(
                emotion_log_id=emotion_log_id,
                agent_message=recommendation,
                cooldown_minutes=cooldown_minutes,
            )

def print_state():
    while not stop_event.is_set():
        time.sleep(5)
        state = get_state()
        print(f"\n[State] 표정: {state['face_emotion']} | "
              f"텍스트: {state['stt_text']} | "
              f"스트레스: {state['stress_level']}\n")

if __name__ == "__main__":
    threads = [
        threading.Thread(target=run_stt, daemon=True),
        threading.Thread(target=run_stress_calculator, daemon=True),
        threading.Thread(target=print_state, daemon=True),
    ]
    for t in threads:
        t.start()

    # 카메라 창은 메인 스레드에서 실행 (macOS 필수)
    run_face_detection()