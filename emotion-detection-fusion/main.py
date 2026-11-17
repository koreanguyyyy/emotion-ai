# main.py
"""
감정 에이전트 — 표정 + 웹캠 심박 융합 버전 (emotion-detection-fusion)

1학기 emotion-detection과 같은 역할(로컬 감지 → 에이전트 서버 호출)을 하되,
스트레스 계산을 2학기 파이프라인(블렌드셰입 + rPPG + 적응형 융합 + MSSD)으로 바꿨다.
에이전트 서버(emotion-ai-server)와 백엔드(emotion-backend)는 수정 없이 그대로 쓴다.

스레드 구성
  메인 스레드 : 웹캠 → 표정·심박 → 융합 (detection/face_fusion.py)
  스레드 1    : 음성 → 텍스트 (Whisper, STT_ENABLED=false면 생략)
  스레드 2    : DETECTION_INTERVAL마다 스트레스 지수 → 개입 판단 → 서버 요청 → TTS → 대화 모드
  스레드 3    : 1초마다 세션 CSV 기록
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)   # logs/, models/ 상대경로를 위해 이 폴더 기준으로 실행
os.makedirs("logs", exist_ok=True)
os.makedirs("audio", exist_ok=True)

import logging
import threading
import time

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/agent_fusion.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

from ai.live_server import build_live_payload, start_live_server
from ai.cooldown import get_remaining_cooldown, is_cooldown_active, update_intervention_time
from ai.llm_client import request_chat, request_chat_end, request_chat_stream, request_llm, request_speech_memory
from ai.conversation import Conversation, configure as configure_conversation
from ai.session_logger import SessionLogger
from ai.watch_hr import WATCH, start_watch_server
from ai.stress_index import StressConfig, StressIndexer
from ai.stt_buffer import clear_stt, get_recent_stt, get_speech_history, get_speech_since
import ai.cooldown as local_cooldown
from backend.api_client import check_cooldown, save_emotion_log
from detection.tts import speak, speak_part, speak_wait
from state.emotion_state import get_state, manual_trigger, on_visitor_reset, request_visitor_reset, stop_event, update_state


def _env(name, default, cast=str):
    raw = os.getenv(name)
    if raw is None or raw.split("#")[0].strip() == "":
        return default
    return cast(raw.split("#")[0].strip())


DETECTION_INTERVAL = _env("DETECTION_INTERVAL", 5, int)
STRESS_THRESHOLD = _env("STRESS_THRESHOLD", 6.0, float)
USER_ID = _env("USER_ID", 1, int)
STT_ENABLED = _env("STT_ENABLED", "true").lower() == "true"
LIVE_PORT = _env("LIVE_PORT", 8765, int)                    # 웹페이지 실시간 모니터링용 (0이면 끔)
EMOTION_LOG_EVERY_SEC = _env("EMOTION_LOG_EVERY_SEC", 60, int)   # 이 간격마다 감정 기록 저장 (0이면 끔)
CHAT_ENABLED = _env("CHAT_ENABLED", "true").lower() == "true"     # 개입 뒤 대화 이어 가기
HR_SOURCE = _env("HR_SOURCE", "webcam").lower()                   # webcam / watch / auto (ai/watch_hr.py)
WATCH_PORT = _env("WATCH_PORT", 8766, int)                        # 갤럭시 워치 심박 받기 (0이면 끔)
WATCH_HOST = _env("WATCH_HOST", "0.0.0.0")
WATCH_TOKEN = _env("WATCH_TOKEN", None)

# 가장 최근 스트레스 계산 결과 (실시간 통로·주기 저장이 같이 씀)
LATEST = {"reading": None}


def chat_context() -> dict:
    """대화 한 턴마다 에이전트에게 알려 주는 지금 상태."""
    st, rd = get_state(), LATEST["reading"]
    return {"face_emotion": st.get("face_emotion") if st.get("face_detected") else None,
            "stress_index": getattr(rd, "index", None),
            "heart_rate": st.get("hr_bpm"), "resting_heart_rate": st.get("resting_hr")}


# 이어지는 대화 (음성 인식과 웹 채팅이 같이 씀)
CHAT_STREAM = _env("CHAT_STREAM", "true").lower() == "true"     # 답을 글자 단위로 받아 바로 보여주고 읽기
# 지난 대화 기억 공간. 평소에는 사용자마다 하나, '새 관람객'을 누르면 새 공간 (앞사람 이야기를 꺼내지 않게)
MEMORY = {"key": f"user-{USER_ID}", "speech_sent": time.time()}
SPEECH_SUMMARY_EVERY_SEC = _env("SPEECH_SUMMARY_EVERY_SEC", 1800, int)   # 대화 밖 발화 요약 주기 (0이면 끔)
CONV = Conversation(send=lambda p: request_chat(p), speak=lambda t: speak(t),   # 시험에서 바꿔 끼울 수 있게
                    get_context=chat_context, user_id=USER_ID,
                    send_stream=(lambda p: request_chat_stream(p)) if CHAT_STREAM else None,
                    speak_part=lambda t: speak_part(t), speak_wait=lambda: speak_wait(),
                    get_memory_key=lambda: MEMORY["key"], get_speech_history=lambda: get_speech_history(),
                    on_end=lambda sid: request_chat_end(sid))
configure_conversation(CONV)


def reset_for_new_visitor() -> None:
    """새 관람객 (카메라 루프는 기준선을 따로 다시 잡는다): 대화·쿨다운·발화 기록을 비운다."""
    CONV.reset()
    MEMORY["key"] = f"user-{USER_ID}-visitor-{int(time.time())}"
    MEMORY["speech_sent"] = time.time()
    manual_trigger.clear()
    clear_stt()
    local_cooldown.last_intervention_time = 0.0
    LATEST["reading"] = None
    update_state(stt_text=None, stress_level=None, stress_index=None,
                 last_intervention_id=None, last_recommendation_id=None, face_baseline_ready=False)
    logger.info("[Reset] 새 관람객 — 대화·쿨다운·발화 기록을 비웠습니다")


on_visitor_reset(reset_for_new_visitor)


def fusion_fields(state: dict, reading, log_type: str) -> dict:
    """백엔드 감정 기록에 함께 저장할 fusion 정보 (2026-11-03 백엔드에 추가한 항목)."""
    def r(v, nd=3):
        return None if v is None else round(float(v), nd)
    return {
        "heartRate": r(state.get("hr_bpm"), 1),
        "restingHeartRate": r(state.get("resting_hr"), 1),
        "stressNeg": r(getattr(reading, "neg", None)),
        "stressHr": r(getattr(reading, "hr_up", None)),
        "stressArousal": r(getattr(reading, "arousal", None)),
        "stressInstab": r(getattr(reading, "instab", None)),
        "hrMode": state.get("hr_mode"),
        "faceMethod": state.get("face_method"),
        "modelEmotion": state.get("model_label"),
        "logType": log_type,
    }


def run_stress_loop(indexer: StressIndexer, slog: SessionLogger) -> None:
    logger.info(f"[Stress] 판단 시작 (주기 {DETECTION_INTERVAL}초, 개입 기준 {STRESS_THRESHOLD}/10)")
    while not stop_event.is_set():
        time.sleep(DETECTION_INTERVAL)
        if stop_event.is_set():   # 대기 중에 종료(q)했으면 서버를 더 부르지 않음
            break

        # 시연용 '지금 말 걸기' (카메라 창 i 키 / 웹 모니터 버튼) — 지수·심박 보정·쿨다운을 건너뛴다
        manual = manual_trigger.is_set()
        if manual:
            manual_trigger.clear()

        reading = indexer.compute()
        state = get_state()
        if reading is None:
            logger.info("[Stress] 얼굴 데이터 부족 → 판단 보류" + (" (시연 요청도 보류 — 얼굴을 화면에)" if manual else ""))
            continue
        update_state(stress_index=reading.index)

        # 0단계: 평상시 심박 보정 중에는 개입하지 않음.
        #        보정 시간(HR_CALIB_TIMEOUT_SEC)이 지나도 심박이 안 잡히면 표정만으로 진행한다.
        mode = state["hr_mode"]
        if mode == "calibrating" and not manual:
            logger.info(f"[Stress] 지수 {reading.index}/10 (심박 기준선 보정 중 → 개입 보류)")
            continue

        hr_txt = (f"심박 {state['hr_bpm']:.0f} (평상시 {state['resting_hr']:.0f})"
                  if state["hr_bpm"] and state["resting_hr"] else "심박 없음")
        hr_up_txt = "모름(재정규화)" if reading.hr_up is None else f"{reading.hr_up:.2f}"
        logger.info(f"[Stress] 지수 {reading.index}/10 = 부정 {reading.neg:.2f} · 심박상승 {hr_up_txt}"
                    f" · 표정각성 {reading.arousal:.2f} · 기복 {reading.instab:.2f} | 표정 {reading.label}"
                    f" | {hr_txt} | 모드 {mode}")

        # 1단계: 로컬 지수 기준
        if reading.index < STRESS_THRESHOLD and not manual:
            continue

        # 2단계: 쿨다운 (백엔드 실패 시 로컬 쿨다운)
        is_active, remaining = (False, 0) if manual else check_cooldown()
        if is_active is None:
            level = round(reading.index)
            is_active, remaining = is_cooldown_active(level), get_remaining_cooldown(level)
        if is_active:
            logger.info(f"[Main] 쿨다운 중 (남은 시간 {remaining}분, 백엔드는 남은 시간을 주지 않으면 0으로 표시)")
            slog.log_event("cooldown_skip", reading.index, reading.label)
            continue

        # 2.5단계: 이미 대화 중이면 새로 말을 걸지 않는다 (대화가 30분을 넘겨 쿨다운이 끝나도)
        if CONV.is_active():
            logger.info("[Stress] 대화 중 → 새 개입 보류")
            slog.log_event("chat_skip", reading.index, reading.label)
            continue

        # 3단계: 감정 로그 저장 (백엔드 스키마는 1학기와 동일)
        stt_text = get_recent_stt()
        if manual:
            logger.info(f"[Demo] 시연 요청 → 지수 {reading.index}/10와 상관없이 개입 흐름 실행")
        slog.log_event("manual_trigger" if manual else "threshold_exceeded", reading.index, reading.label,
                       detail=f"[{mode}] {stt_text}")
        emotion_log_id = save_emotion_log(
            face_emotion=reading.label,
            face_confidence=float(reading.face_confidence),
            speech_text=stt_text,
            vad_stress_score=float(reading.index),
            # 이전에는 비워 둬서 웹페이지 스트레스가 0으로 보였다. 로컬 지수를 넣는다
            # (에이전트 서버의 LLM 보정값은 개입 이력에 따로 남는다).
            final_stress_level=round(reading.index),
            intervention_triggered=False,   # 실제 개입 여부는 에이전트 응답 뒤에 정해짐 → 개입 이력에 저장
            extra=fusion_fields(state, reading, "manual" if manual else "threshold"),
        )

        # 4단계: 에이전트 서버 요청 — vad_stress 자리에 새 스트레스 지수를 넣는다
        result = request_llm(
            vad_stress=float(reading.index),
            stt_text=stt_text,
            face_emotion=reading.label,
            user_id=USER_ID,
            emotion_log_id=emotion_log_id or 0,
            force=manual,
            memory_key=MEMORY["key"],
            speech_history=get_speech_history(),
        )
        if result is None:
            stress_level, recommendation = round(reading.index), None
            intervention_id = recommendation_id = None
        else:
            stress_level = result["stress_level"]
            recommendation = result["recommendation"]
            intervention_id = result.get("intervention_id")
            recommendation_id = result.get("recommendation_id")
        update_state(stress_level=stress_level)

        # 5단계: 개입 (TTS) — 에이전트가 먼저 말을 거는 부분
        if recommendation:
            logger.info(f"\n[에이전트 개입]\n{recommendation}\n")
            update_intervention_time()
            update_state(last_intervention_id=intervention_id, last_recommendation_id=recommendation_id)
            slog.log_event("intervention", reading.index, reading.label, stress_level, recommendation)
            if CHAT_ENABLED:   # 첫 메시지를 읽고 대화 모드로 (사용자가 말하거나 웹에 쓰면 이어서 답한다)
                threading.Thread(target=CONV.start, args=(recommendation, intervention_id), daemon=True).start()
            else:
                threading.Thread(target=speak, args=(recommendation,), daemon=True).start()
        else:
            slog.log_event("agent_no_intervention", reading.index, reading.label, stress_level)


def run_session_log(indexer: StressIndexer, slog: SessionLogger) -> None:
    while not stop_event.is_set():
        time.sleep(1.0)
        reading = indexer.compute()
        LATEST["reading"] = reading
        slog.log_tick(get_state(), reading)


def run_speech_memory() -> None:
    """평소 켜 두고 쓸 때: SPEECH_SUMMARY_EVERY_SEC마다 대화 밖에서 한 말을 서버에 보내 요약·기억하게 한다.
    (최근 SPEECH_HISTORY_SEC 안의 말은 그대로도 보내므로, 그보다 오래된 이야기를 기억하기 위한 것)"""
    if SPEECH_SUMMARY_EVERY_SEC <= 0 or not STT_ENABLED:
        return
    while not stop_event.wait(SPEECH_SUMMARY_EVERY_SEC):
        items = get_speech_since(MEMORY["speech_sent"])
        if len(items) >= 3:
            request_speech_memory(MEMORY["key"], items)
            MEMORY["speech_sent"] = items[-1]["timestamp"]


def run_periodic_emotion_log() -> None:
    """EMOTION_LOG_EVERY_SEC마다 감정 기록을 백엔드에 저장한다. 이전에는 개입 기준을 넘을 때만 저장돼서
    대시보드의 감정 분포·스트레스 추이가 비어 있었다."""
    if EMOTION_LOG_EVERY_SEC <= 0:
        return
    while not stop_event.wait(EMOTION_LOG_EVERY_SEC):
        state, reading = get_state(), LATEST["reading"]
        if reading is None or not state.get("face_detected"):
            continue
        save_emotion_log(
            face_emotion=reading.label,
            face_confidence=float(reading.face_confidence),
            speech_text=state.get("stt_text") or "",
            vad_stress_score=float(reading.index),
            final_stress_level=round(reading.index),
            intervention_triggered=False,
            extra=fusion_fields(state, reading, "periodic"),
        )


if __name__ == "__main__":
    from detection.face_fusion import run_face_fusion   # mediapipe/cv2 로딩

    indexer = StressIndexer(StressConfig.from_env())
    slog = SessionLogger("logs")
    logger.info(f"[Main] 세션 기록: {slog.session_path}")

    threads = [
        threading.Thread(target=run_stress_loop, args=(indexer, slog), daemon=True),
        threading.Thread(target=run_session_log, args=(indexer, slog), daemon=True),
        threading.Thread(target=run_periodic_emotion_log, daemon=True),
        threading.Thread(target=run_speech_memory, daemon=True),
    ]
    def _web_chat(body: dict) -> dict:
        msg = str(body.get("message") or "").strip()[:500]
        if not msg:
            raise ValueError("message가 비어 있습니다")
        if CONV.busy:
            return {"ok": False, "reason": "busy"}
        CONV.submit_async(msg, "web")
        return {"ok": True}

    def _web_chat_start(body: dict) -> dict:
        if not CONV.is_active():
            threading.Thread(target=CONV.start_free, daemon=True).start()
        return {"ok": True}

    def _web_intervene(body: dict) -> dict:
        if CONV.is_active():
            return {"ok": False, "reason": "chatting"}
        manual_trigger.set()
        return {"ok": True, "waitSec": DETECTION_INTERVAL}

    def _web_reset(body: dict) -> dict:
        request_visitor_reset()
        slog.log_event("new_visitor")
        return {"ok": True}

    def _web_chat_end(body: dict) -> dict:
        CONV.end("user")
        return {"ok": True}

    start_live_server(
        LIVE_PORT,
        lambda: build_live_payload(get_state(), LATEST["reading"], CONV.snapshot()),
        post_handlers={"/chat": _web_chat, "/chat/start": _web_chat_start, "/chat/end": _web_chat_end,
                       "/intervene": _web_intervene, "/reset": _web_reset},
    )
    # 워치 심박: HR_SOURCE=webcam이어도 받아서 CSV에 비교용으로 남긴다 (WATCH_PORT=0이면 끔)
    start_watch_server(WATCH_PORT, WATCH, host=WATCH_HOST, token=WATCH_TOKEN)
    if STT_ENABLED:
        from detection.stt_detector import run_stt   # Whisper 모델 로딩 (시간 걸림)
        threads.append(threading.Thread(target=run_stt, daemon=True))
    else:
        logger.info("[Main] STT_ENABLED=false → 음성 인식 없이 실행")

    for t in threads:
        t.start()
    try:
        run_face_fusion(indexer)   # macOS는 카메라 창을 메인 스레드에서 띄워야 함
    finally:
        stop_event.set()
        time.sleep(0.2)
        slog.close()
        logger.info(f"[Main] 종료. 세션 기록: {slog.session_path}")
