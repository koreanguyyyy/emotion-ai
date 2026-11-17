# detection/face_fusion.py
"""
웹캠 루프 (메인 스레드) — 1학기 face_detector.py(DeepFace)를 대체한다.

한 프레임에서 두 가지를 동시에 뽑는다 (2학기 emotion_pipeline 재사용):
  1) MediaPipe 블렌드셰입 → 표정 정서가·각성도·신뢰도 + 감정 라벨
  2) 같은 얼굴의 이마 색 변화 → rPPG(CHROM) 심박수
그리고 표정 신뢰도 기반 적응형 융합 결과를 StressIndexer에 쌓는다.

1학기 대비 바뀐 점
  - 실제 처리 fps를 처음 몇 초 동안 측정해서 rPPG에 넣는다. 카메라가 30fps라고
    알려줘도 얼굴 분석 때문에 실제 루프가 느리면 심박수가 그 비율만큼 틀어지기 때문.
  - 심박 품질(quality)이 기준 미만인 추정치는 버린다 (2학기 코드에서 미구현이던 부분).
  - 평상시 심박(기준선)을 개인별로 보정한 뒤에만 심박을 각성도에 반영한다.
"""
import os
import time

import cv2
from dotenv import load_dotenv

from ai.frame_store import FRAMES
from ai.emotion_label import EN_LABEL
from ai.face_affect import FaceBaseline, face_affect
from ai.stress_index import BaselineHR, StressIndexer, is_octave_jump
from backend.api_client import submit_intervention_feedback, submit_recommendation_feedback
from emotion_pipeline.fusion import FusedAffect, adaptive_fuse, hr_to_arousal
from emotion_pipeline.roi import FaceAnalyzer
from detection.face_model import load_face_model
from detection.rppg_diag import RPPGDiagnostics
from detection.rppg_timed import TimedRPPG
from ai.watch_hr import WATCH, HRSelector
from state.emotion_state import get_state, manual_trigger, request_visitor_reset, stop_event, update_state, visitor_reset

load_dotenv()


def _env(name, default, cast=str):
    raw = os.getenv(name)
    if raw is None or raw.split("#")[0].strip() == "":
        return default
    return cast(raw.split("#")[0].strip())


CAMERA_INDEX = _env("CAMERA_INDEX", 0, int)
MODEL_PATH = _env("FACE_MODEL_PATH", "models/face_landmarker.task")
# 품질 기준: CHROM·POS 중 높은 쪽을 고르므로 잡음 점수도 올라간다. 잡음만 넣은 시험(40회)에서
# 0.35는 1.2% 통과, 0.40은 0% → 0.40 (tests/test_rppg_timed.py)
HR_QUALITY_MIN = _env("HR_QUALITY_MIN", 0.40, float)
DIAG_EVERY_SEC = _env("RPPG_DIAG_EVERY_SEC", 30.0, float)   # 0이면 진단 그래프 저장 안 함
RPPG_WINDOW_SEC = _env("RPPG_WINDOW_SEC", 20.0, float)
BASELINE_SEC = _env("BASELINE_SEC", 60.0, float)
FACE_BASELINE_SEC = _env("FACE_BASELINE_SEC", BASELINE_SEC, float)   # 무표정 기준선 보정 시간
FACE_MODEL = _env("FACE_MODEL", "compare").lower()   # off / compare(기록만) / use(지수 계산에 사용)
FACE_MODEL_EVERY = _env("FACE_MODEL_EVERY", 3, int)
RESTING_HR = _env("RESTING_HR", None, float)
HR_CALIB_TIMEOUT_SEC = _env("HR_CALIB_TIMEOUT_SEC", 120.0, float)   # 이 시간 안에 보정 안 되면 표정만으로 동작
HR_SOURCE = _env("HR_SOURCE", "webcam").lower()           # webcam / watch / auto (ai/watch_hr.py)
WATCH_MAX_AGE_SEC = _env("WATCH_MAX_AGE_SEC", 5.0, float)
HR_HOLD_SEC = _env("HR_HOLD_SEC", 5.0, float)   # 마지막 유효 심박을 이 시간 동안 유지 (잠깐 끊겨도 모드가 흔들리지 않게)


def handle_feedback(key: int, state: dict) -> None:
    """키보드 단축키로 피드백 처리 (1학기와 동일)."""
    intervention_id = state.get("last_intervention_id")
    recommendation_id = state.get("last_recommendation_id")
    if not intervention_id:
        print("[Feedback] 최근 개입 없음 → 스킵")
        return
    if key == ord("l"):
        submit_intervention_feedback(intervention_id, "HELPFUL")
        if recommendation_id:
            submit_recommendation_feedback(recommendation_id, "LIKED", played=True)
        print("[Feedback] 👍 LIKED")
    elif key == ord("d"):
        submit_intervention_feedback(intervention_id, "UNNECESSARY")
        if recommendation_id:
            submit_recommendation_feedback(recommendation_id, "DISLIKED")
        print("[Feedback] 👎 DISLIKED")
    elif key == ord("s"):
        submit_intervention_feedback(intervention_id, "NEUTRAL")
        if recommendation_id:
            submit_recommendation_feedback(recommendation_id, "SKIPPED")
        print("[Feedback] ⏭ SKIPPED")


MODE_TEXT = {
    "calibrating": "평상시 심박 보정 중 (개입 보류)",
    "fusion": "표정 + 심박 융합",
    "face_only": f"표정만 (심박 보정 {HR_CALIB_TIMEOUT_SEC:.0f}초 안에 실패 — 조명·앞머리·움직임 확인)",
    "face_only_gap": "표정만 (심박 일시 끊김)",
}


def _diagnostics(reading) -> dict:
    """원인 분석용 값. 블렌드셰입 상위 3개, 이마 영역 밝기와 크기."""
    if not reading.detected:
        return {"diag_top_blendshapes": None, "diag_forehead_brightness": None, "diag_forehead_px": None}
    top = sorted(((k, v) for k, v in reading.blendshapes.items() if k != "_neutral"),
                 key=lambda kv: kv[1], reverse=True)[:3]
    fh = reading.boxes.get("forehead")
    px = max(0, fh[2] - fh[0]) * max(0, fh[3] - fh[1]) if fh else None
    rgb = reading.forehead_rgb
    return {
        "diag_top_blendshapes": " ".join(f"{k}:{v:.2f}" for k, v in top),
        "diag_forehead_brightness": round(sum(rgb) / 3, 1) if rgb else None,
        "diag_forehead_px": px,
    }


def _put(frame, text, row, color):
    y = 28 + 26 * row
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def run_face_fusion(indexer: StressIndexer) -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[Face] 카메라 {CAMERA_INDEX}번을 열 수 없습니다. .env의 CAMERA_INDEX를 확인하세요.")
        stop_event.set()
        return

    analyzer = FaceAnalyzer(MODEL_PATH)
    nominal_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # 맥북 카메라는 15fps라고 알려주지만 실제로는 30fps로 보낸다(2026-10-29 측정, 중복 0개).
    # 이 값은 MediaPipe 내부 시각에만 쓰이고, 심박 계산은 프레임마다 실제 시각을 쓴다.
    analyzer.set_fps(30.0)
    baseline = BaselineHR(calib_sec=BASELINE_SEC, fixed_bpm=RESTING_HR, timeout_sec=HR_CALIB_TIMEOUT_SEC)
    face_base = FaceBaseline(calib_sec=FACE_BASELINE_SEC)   # 개인별 무표정 기준선
    face_model = load_face_model(FACE_MODEL, FACE_MODEL_EVERY)   # 학습된 표정 모델 (선택)
    model_mode = FACE_MODEL if face_model is not None else "off"

    print("[Face] 표정 + 웹캠 심박 측정 시작... (첫 1분은 평상시 심박 보정 — 편하게 앉아 계세요)")
    print("[Feedback] l: LIKED / d: DISLIKED / s: SKIPPED / q: 종료")

    t0 = time.monotonic()
    rppg = TimedRPPG(window_sec=RPPG_WINDOW_SEC)      # 이마+볼, CHROM·POS, 실제 시각 기반
    diag = RPPGDiagnostics("logs", DIAG_EVERY_SEC)
    fps_reported = False
    hr_method = hr_q_chrom = hr_q_pos = fps_actual = None
    hr_bpm = hr_q = hr_raw = None
    last_valid_t = -1e9
    octave_rejects = 0
    cam_bpm, cam_t = None, -1e9         # 품질을 통과한 마지막 웹캠 심박 (어느 설정이든 기록용)
    selector = HRSelector(HR_SOURCE)    # 지수에 쓸 심박: 웹캠 / 워치 / 자동
    hr_source_used = None
    if HR_SOURCE != "webcam":
        print(f"[Face] 심박 출처: {HR_SOURCE} (워치 앱이 보내는 심박{'을 우선 사용' if HR_SOURCE == 'auto' else '만 사용'})")
    last_mode = None

    score_ema: dict = {}
    try:
        while not stop_event.is_set():
            if visitor_reset.is_set():   # 새 관람객: 기준선·심박·지수 기록을 처음부터
                visitor_reset.clear()
                t0 = time.monotonic()
                baseline = BaselineHR(calib_sec=BASELINE_SEC, fixed_bpm=RESTING_HR, timeout_sec=HR_CALIB_TIMEOUT_SEC)
                face_base = FaceBaseline(calib_sec=FACE_BASELINE_SEC)
                rppg = TimedRPPG(window_sec=RPPG_WINDOW_SEC)
                diag = RPPGDiagnostics("logs", DIAG_EVERY_SEC)
                selector = HRSelector(HR_SOURCE)
                hr_bpm = hr_q = hr_raw = None
                cam_bpm, cam_t, last_valid_t = None, -1e9, -1e9
                octave_rejects, last_mode, hr_source_used = 0, None, None
                indexer.clear()
                score_ema = {}
                print(f"[Face] 새 관람객 — 무표정 {FACE_BASELINE_SEC:.0f}초 · 평상시 심박 {BASELINE_SEC:.0f}초 다시 보정합니다")
            ok, frame = cap.read()
            if not ok:
                print("[Face] 프레임을 읽지 못했습니다 (카메라 권한/번호 확인).")
                break
            now = time.monotonic() - t0
            reading = analyzer.analyze_bgr(frame)

            # ── 심박: 프레임마다 실제 시각과 이마·볼 색을 넣고, 약 1초마다 새 추정 ──
            if reading.detected and reading.forehead_rgb is not None:
                est = rppg.update(now, reading.forehead_rgb, reading.cheek_rgb)
                if est is not None:
                    if not fps_reported:
                        print(f"[Face] 실제 프레임 속도 {est.fps_actual:.1f} fps (카메라 표기 {nominal_fps:.0f}) "
                              f"— 심박은 실제 시각 기준으로 계산")
                        fps_reported = True
                    hr_method, hr_q_chrom, hr_q_pos, fps_actual = (
                        est.method, est.chrom_quality, est.pos_quality, est.fps_actual)
                    diag.maybe_save(now, est, HR_QUALITY_MIN)
                    hr_raw, hr_q = (est.bpm if est.bpm == est.bpm else None), est.quality
                    # 평상시 심박이 정해진 뒤에는 그 절반·두 배 근처 값을 버린다 (ai/stress_index.py 참고)
                    octave = baseline.ready and is_octave_jump(est.bpm, baseline.value)
                    if octave:
                        octave_rejects += 1
                    if est.bpm == est.bpm and est.quality >= HR_QUALITY_MIN and not octave:
                        cam_bpm, cam_t = est.bpm, now
                        if selector.use_cam(watch_fresh=WATCH.latest(WATCH_MAX_AGE_SEC)[0] is not None):
                            hr_bpm, last_valid_t, hr_source_used = est.bpm, now, "webcam"
                            baseline.add(now, est.bpm)
                    # 품질 미달 추정치는 버리고, 직전 유효값은 HR_HOLD_SEC 동안만 유지

            # ── 워치 심박: 새 값이 들어왔을 때만 반영 (워치는 접촉식이라 절반·두 배 검사를 하지 않는다) ──
            w_bpm, w_rr, w_seq = WATCH.latest(WATCH_MAX_AGE_SEC)
            if selector.use_watch(w_bpm, w_seq):
                hr_bpm, last_valid_t, hr_source_used = w_bpm, now, "watch"
                baseline.add(now, w_bpm)
            cam_now = cam_bpm if (now - cam_t) <= HR_HOLD_SEC else None

            hr_valid = hr_bpm is not None and (now - last_valid_t) <= HR_HOLD_SEC
            hr_now = hr_bpm if hr_valid else None
            bstatus = baseline.status(now)

            # ── 표정: 처음 FACE_BASELINE_SEC 동안 무표정 기준선을 잡고, 이후 평소 대비 변화로 읽는다 ──
            if reading.detected:
                was_ready = face_base.ready
                face_base.add(now, reading.blendshapes)
                if face_base.ready and not was_ready:
                    print("[Face] 무표정 기준선 보정 완료 — 이제 평소 대비 표정 변화로 읽습니다")
            fa = face_affect(reading.blendshapes, face_base, reading.detected)
            if fa.scores:   # 감정 막대: 프레임마다 떨리지 않게 살짝 부드럽게 (웹은 1초마다 읽음)
                score_ema = {k: 0.7 * score_ema.get(k, v) + 0.3 * v for k, v in fa.scores.items()}
            elif not reading.detected:
                score_ema = {}
            rule_label, rule_valence = fa.label, fa.valence * fa.confidence
            ma = None
            if face_model is not None:
                ma = face_model.update(frame, reading.boxes.get("face")) if reading.detected else None
                if model_mode == "use":
                    # 모델 결과로 대신한다. 모델 정서가는 이미 −1~1이라 신뢰도(1 − 무표정 확률)를 곱해 쓴다.
                    fa = (face_affect({}, None, False) if ma is None else
                          type(fa)(ma.valence, ma.arousal, ma.confidence, ma.label, fa.top_delta))

            # ── 융합 ──
            conf = fa.confidence
            if hr_valid and baseline.ready:
                mode = "fusion"
                bio_arousal = hr_to_arousal(hr_now, baseline.value)
                fused = adaptive_fuse(fa.valence, fa.arousal, conf, bio_arousal)
            else:
                # 심박을 모를 때는 0(=평상시 심박)을 넣지 않는다. 0을 넣으면 '심박이 평상시'라고
                # 단정하는 것이 되어 표정의 각성도까지 깎인다. 각성도는 표정에서만 가져온다.
                mode = "calibrating" if bstatus == "calibrating" else (
                    "face_only_gap" if baseline.ready else "face_only")
                fused = FusedAffect(valence=conf * fa.valence, arousal=fa.arousal, face_weight=1.0)
            if mode != last_mode:
                print(f"[Face] 동작 모드: {MODE_TEXT[mode]}")
                last_mode = mode
            label = fa.label
            # 스트레스 지수에는 표정(정서가·각성도)과 심박 상승을 따로 넣는다 (ai/stress_index.py 참고)
            hr_delta = (hr_now - baseline.value) if (mode == "fusion") else None
            indexer.add(now, reading.detected, conf * fa.valence, fa.arousal, conf, label, hr_delta)

            update_state(
                face_detected=reading.detected, face_emotion=label, face_confidence=conf,
                face_scores=dict(score_ema) if score_ema else None,
                fused_valence=fused.valence, fused_arousal=fused.arousal,
                face_weight=fused.face_weight, hr_bpm=hr_now, hr_quality=hr_q,
                resting_hr=baseline.value, hr_mode=mode, hr_raw_bpm=hr_raw,
                hr_method=hr_method, hr_q_chrom=hr_q_chrom, hr_q_pos=hr_q_pos, fps_actual=fps_actual,
                hr_octave_rejects=octave_rejects,
                hr_source=hr_source_used if hr_valid else None, watch_hr_bpm=w_bpm, watch_rr_ms=w_rr,
                cam_hr_bpm=cam_now,
                face_baseline_ready=face_base.ready, diag_top_delta=fa.top_delta,
                face_method="model" if model_mode == "use" else "rule",
                rule_label=rule_label, rule_valence=rule_valence,
                model_label=None if ma is None else ma.label,
                model_valence=None if ma is None else ma.valence * ma.confidence,
                model_arousal=None if ma is None else ma.arousal,
                model_conf=None if ma is None else ma.confidence,
                model_p_sad=None if ma is None else ma.p_sad,
                **_diagnostics(reading),
            )

            # ── 화면 표시 (cv2는 한글 미지원 → 영어) ──
            state = get_state()
            if reading.detected:
                box = reading.boxes.get("face")
                if box:
                    cv2.rectangle(frame, box[:2], box[2:], (0, 200, 0), 2)
                fh = reading.boxes.get("forehead")
                if fh:
                    cv2.rectangle(frame, fh[:2], fh[2:], (0, 220, 255), 1)
            green, red, yellow = (0, 200, 0), (0, 0, 255), (0, 220, 255)
            if not reading.detected:
                _put(frame, "face: NOT detected", 0, red)
            elif not face_base.ready:
                _put(frame, f"face: calibrating neutral {face_base.progress(now) * 100:.0f}% (keep a relaxed face)", 0, yellow)
            else:
                _put(frame, f"face: {EN_LABEL.get(label, label)}  conf {conf:.2f}  valence {fa.valence * conf:+.2f}", 0, green)
            if hr_now and hr_source_used == "watch":
                cam_txt = f"{cam_now:.0f}" if cam_now else "--"
                hr_text, hr_color = f"HR {hr_now:.0f} bpm (watch)  webcam {cam_txt}", green
            elif selector.source == "watch":
                age = WATCH.age()
                hr_text = "HR: waiting for watch app" if age is None else f"HR: watch silent {age:.0f}s"
                hr_color = yellow
            elif rppg.last is None:
                hr_text, hr_color = f"HR: collecting {rppg.buffered_sec():.0f}/{RPPG_WINDOW_SEC:.0f}s", yellow
            elif hr_now:
                hr_text, hr_color = f"HR {hr_now:.0f} bpm (q {hr_q:.2f}, {hr_method})", green
            else:
                q_txt = f"q {hr_q:.2f} < {HR_QUALITY_MIN}" if hr_q is not None else f"{RPPG_WINDOW_SEC:.0f}s window"
                hr_text, hr_color = f"HR: no reliable pulse ({q_txt})", yellow
            _put(frame, hr_text, 1, hr_color)
            if mode == "fusion":
                _put(frame, f"mode: face+HR  resting {baseline.value:.0f}  arousal {fused.arousal:+.2f}", 2, green)
            elif mode == "calibrating":
                _put(frame, f"calibrating resting HR {baseline.progress(now) * 100:.0f}%  "
                            f"(face-only in {max(0, HR_CALIB_TIMEOUT_SEC - now):.0f}s)", 2, yellow)
            else:
                _put(frame, "mode: FACE ONLY (no reliable heart rate)", 2, red)
            si = state.get("stress_index")
            line = f"stress index {si:.1f}/10" if si is not None else "stress index: warming up"
            if state.get("stress_level") is not None:
                line += f"  |  agent {state['stress_level']}/10"
            _put(frame, line, 3, red if (si or 0) >= 6 else green)
            if ma is not None:
                _put(frame, f"model{'(used)' if model_mode == 'use' else ''}: {ma.label_en.lower()} "
                            f"p={ma.probs.get(ma.label_en, 0):.2f}  valence {ma.valence * ma.confidence:+.2f}", 4,
                     green if model_mode == "use" else (200, 200, 200))
            if state.get("last_intervention_id"):
                _put(frame, "l:LIKED  d:DISLIKED  s:SKIPPED  i:talk now  n:new visitor", 5, (255, 255, 0))

            FRAMES.publish(frame)   # 웹페이지 '얼굴 인식' 화면 (보는 사람이 있을 때만)
            cv2.imshow("Emotion Agent - fusion", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("l"), ord("d"), ord("s")):
                handle_feedback(key, state)
            if key == ord("n"):   # 전시용: 새 관람객
                request_visitor_reset()
            if key == ord("i"):   # 시연용: 지금 말 걸기
                manual_trigger.set()
                print("[Demo] 'i' → 다음 판단 주기(최대 몇 초)에 에이전트가 말을 겁니다")
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        analyzer.close()
