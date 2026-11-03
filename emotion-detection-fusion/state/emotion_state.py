# state/emotion_state.py
# 스레드 간 공유 상태. 1학기 emotion-detection의 구조를 그대로 두고 심박·융합 항목을 추가했다.
import threading

state_lock = threading.Lock()
stop_event = threading.Event()
# 시연용 '지금 말 걸기' (카메라 창 i 키 / 웹 모니터 버튼): 지수·쿨다운과 상관없이 다음 판단 주기에 개입
manual_trigger = threading.Event()
# 전시용 '새 관람객': 무표정·평상시 심박 기준선을 다시 잡고 대화·쿨다운을 비운다 (2026-11-04)
visitor_reset = threading.Event()      # 카메라 루프가 확인하고 기준선을 새로 만든다
_reset_callbacks: list = []            # main.py가 대화·쿨다운 정리 함수를 등록


def on_visitor_reset(fn) -> None:
    _reset_callbacks.append(fn)


def request_visitor_reset() -> None:
    """카메라 창 n 키 / 웹 '새 관람객' 버튼."""
    visitor_reset.set()
    for fn in list(_reset_callbacks):
        try:
            fn()
        except Exception as e:
            print(f"[Reset] 정리 중 오류: {e}")

emotion_state = {
    # ── 1학기와 같은 항목 ──
    "face_emotion": None,           # 한국어 감정 라벨 (에이전트 서버의 Tool 선택용)
    "stt_text": None,               # 최신 STT 텍스트
    "stress_level": None,           # 에이전트가 최종 판단한 스트레스 (0~10)
    "last_intervention_id": None,   # 최근 개입 ID (피드백 제출용)
    "last_recommendation_id": None, # 최근 추천 ID (피드백 제출용)
    # ── 2학기 파이프라인에서 추가된 항목 ──
    "face_detected": False,
    "face_confidence": 0.0,         # 표정 뚜렷함 (0~1)
    "fused_valence": 0.0,           # 융합 정서가 (-1~1)
    "fused_arousal": 0.0,           # 융합 각성도 (-1~1, 심박 포함)
    "face_weight": 0.0,             # 각성도 융합에서 표정이 차지한 비중
    "hr_bpm": None,                 # 웹캠 rPPG 심박수 (품질 통과한 값만)
    "hr_quality": None,
    "resting_hr": None,             # 개인 평상시 심박 (보정 후 채워짐)
    "stress_index": None,           # 로컬 스트레스 지수 (0~10)
    "hr_mode": "calibrating",       # calibrating(보정 중) / fusion(표정+심박) / face_only(심박 없음) / face_only_gap(심박 일시 끊김)
    "hr_raw_bpm": None,             # 품질 기준과 상관없이 마지막으로 추정된 심박 (진단용)
    "hr_method": None,              # 채택된 rPPG 방식 (chrom / pos)
    "hr_q_chrom": None,             # CHROM 품질
    "hr_q_pos": None,               # POS 품질
    "fps_actual": None,             # 실제 프레임 속도 (최근 20초 평균)
    "hr_octave_rejects": 0,         # 평상시 심박의 절반·두 배라서 버린 추정 수 (누적)
    # ── 갤럭시 워치 (ai/watch_hr.py) ──
    "hr_source": None,              # 지금 hr_bpm이 어디서 왔는지: webcam / watch
    "watch_hr_bpm": None,           # 워치 심박 (최근 WATCH_MAX_AGE_SEC 안의 값)
    "watch_rr_ms": None,            # 워치 심박 간격 (HRV용, 센서가 줄 때만)
    "cam_hr_bpm": None,             # 품질을 통과한 웹캠 심박 (워치를 쓸 때도 비교용으로 기록)
    # ── 진단용 (세션 CSV에 기록) ──
    "diag_top_blendshapes": None,   # 가장 큰 블렌드셰입 3개 "이름:값"
    "diag_top_delta": None,         # 평소(무표정 기준선)보다 가장 많이 올라간 블렌드셰입 3개
    "face_baseline_ready": False,   # 무표정 기준선 보정 완료 여부
    # ── 표정 계산 방식 비교 (규칙 기반 vs 학습 모델) ──
    "face_method": "rule",          # 스트레스 계산에 쓴 방식: rule / model
    "rule_label": None, "rule_valence": None,
    "model_label": None, "model_valence": None, "model_arousal": None, "model_conf": None, "model_p_sad": None,
    "diag_forehead_brightness": None,  # 이마 영역 평균 밝기 (0~255)
    "diag_forehead_px": None,       # 이마 영역 크기 (픽셀 수)
}


def update_state(**kwargs):
    with state_lock:
        for key, value in kwargs.items():
            if key in emotion_state:
                emotion_state[key] = value


def get_state():
    with state_lock:
        return emotion_state.copy()
