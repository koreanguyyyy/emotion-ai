# ai/session_logger.py
"""
세션 기록 (논문 그래프·평가용). 실행할 때마다 logs/ 아래에 CSV 두 개를 만든다.
  - session_<시각>.csv : 1초마다 표정·심박·융합값·스트레스 지수
  - events_<시각>.csv  : 개입 판단·에이전트 응답·피드백 등 사건
원본 영상·음성은 저장하지 않는다 (1학기 Privacy-by-Design 원칙 유지).
"""
import csv
import os
import threading
import time
from datetime import datetime

SESSION_FIELDS = [
    "wall_time", "t_sec", "face_detected", "face_emotion", "face_confidence",
    "fused_valence", "fused_arousal", "face_weight", "hr_bpm", "hr_quality",
    "resting_hr", "stress_index", "neg", "hr_up", "arousal", "instab", "mssd", "agent_stress_level",
    # 진단용
    "hr_source", "watch_hr_bpm", "watch_rr_ms", "cam_hr_bpm",   # 워치 vs 웹캠 비교 (2026-11-03)
    "hr_mode", "hr_raw_bpm", "hr_method", "hr_q_chrom", "hr_q_pos", "fps_actual", "hr_octave_rejects",
    "face_baseline_ready", "diag_top_delta",
    "face_method", "rule_label", "rule_valence", "model_label", "model_valence", "model_arousal", "model_conf", "model_p_sad",
    "diag_forehead_brightness", "diag_forehead_px", "diag_top_blendshapes",
]
EVENT_FIELDS = ["wall_time", "t_sec", "event", "stress_index", "face_emotion",
                "agent_stress_level", "detail"]


class SessionLogger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = os.path.join(log_dir, f"session_{stamp}.csv")
        self.events_path = os.path.join(log_dir, f"events_{stamp}.csv")
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        self._sf = open(self.session_path, "w", newline="", encoding="utf-8-sig")
        self._ef = open(self.events_path, "w", newline="", encoding="utf-8-sig")
        self._sw = csv.DictWriter(self._sf, fieldnames=SESSION_FIELDS, extrasaction="ignore")
        self._ew = csv.DictWriter(self._ef, fieldnames=EVENT_FIELDS, extrasaction="ignore")
        self._sw.writeheader(); self._ew.writeheader()

    def _base(self) -> dict:
        return {"wall_time": datetime.now().isoformat(timespec="seconds"),
                "t_sec": round(time.monotonic() - self._t0, 1)}

    @staticmethod
    def _fmt(v):
        return round(v, 4) if isinstance(v, float) else v

    def log_tick(self, state: dict, reading) -> None:
        row = {**self._base(), **{k: self._fmt(state.get(k)) for k in SESSION_FIELDS if k in state},
               "agent_stress_level": state.get("stress_level")}
        if reading is not None:
            row.update({k: getattr(reading, k) for k in ("neg", "hr_up", "arousal", "instab", "mssd")})
            row["stress_index"] = reading.index
        with self._lock:
            self._sw.writerow(row); self._sf.flush()

    def log_event(self, event: str, stress_index=None, face_emotion=None,
                  agent_stress_level=None, detail: str = "") -> None:
        row = {**self._base(), "event": event, "stress_index": stress_index,
               "face_emotion": face_emotion, "agent_stress_level": agent_stress_level,
               "detail": detail.replace("\n", " ")[:500]}
        with self._lock:
            self._ew.writerow(row); self._ef.flush()

    def close(self) -> None:
        with self._lock:
            self._sf.close(); self._ef.close()
