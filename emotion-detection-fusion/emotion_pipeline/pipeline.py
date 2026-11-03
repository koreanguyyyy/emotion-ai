"""
전체 파이프라인 오케스트레이터 — 적용방안 문서의 4계층 구조를 한 클래스로 통합.

    영상 프레임
      → FaceAnalyzer (roi.py)        : 표정(valence/arousal/confidence) + rPPG용 ROI 색상
      → CHROMExtractor (rppg.py)     : 심박수(HR) 추정
      → adaptive_fuse (fusion.py)    : 표정 신뢰도 기반 융합
      → InstabilityTracker (instability.py) : 시계열 누적 → MSSD 등 기복 지표

⚠️ 실행 환경 제약: 이 모듈은 OpenCV VideoCapture로 '웹캠(index=0)' 또는 '영상
파일 경로' 둘 다 받을 수 있게 짰다. 클라우드 샌드박스에는 카메라도 없고
MediaPipe 모델 파일을 받을 네트워크 경로도 막혀 있어서, 로컬 PC에서
실행하는 것을 전제로 한다. 사용법은 맨 아래 __main__ 참고.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .fusion import adaptive_fuse, hr_to_arousal
from .instability import InstabilityReport, InstabilityTracker
from .roi import FaceAnalyzer, FaceReading
from .rppg import CHROMExtractor


@dataclass
class FrameResult:
    timestamp: float
    face_detected: bool
    face_valence: float
    face_arousal: float
    face_confidence: float
    hr_bpm: float | None
    hr_quality: float | None
    fused_valence: float
    fused_arousal: float
    face_weight: float


class EmotionPipeline:
    def __init__(self, model_path: str, fps: float = 30.0,
                 resting_hr_bpm: float = 70.0, rppg_window_sec: float = 20.0):
        self.face_analyzer = FaceAnalyzer(model_path)
        self.face_analyzer.set_fps(fps)
        self.rppg = CHROMExtractor(fs=fps, window_sec=rppg_window_sec)
        self.instability = InstabilityTracker()
        self.resting_hr_bpm = resting_hr_bpm
        self._last_hr = None  # 윈도우가 덜 찼을 때 직전 값을 유지
        self.last_reading: FaceReading | None = None  # 화면 표시용

    def process_frame(self, frame_bgr: np.ndarray, timestamp: float) -> FrameResult:
        reading = self.face_analyzer.analyze_bgr(frame_bgr)
        self.last_reading = reading

        hr_bpm, hr_quality = None, None
        if reading.detected and reading.forehead_rgb is not None:
            est = self.rppg.update(reading.forehead_rgb)
            if est is not None:
                self._last_hr = est
        if self._last_hr is not None:
            hr_bpm, hr_quality = self._last_hr.bpm, self._last_hr.quality

        bio_arousal = hr_to_arousal(hr_bpm, self.resting_hr_bpm) if hr_bpm else 0.0

        fused = adaptive_fuse(
            face_valence=reading.valence,
            face_arousal=reading.arousal,
            face_confidence=reading.confidence if reading.detected else 0.0,
            bio_arousal=bio_arousal,
        )
        self.instability.add(timestamp, fused.valence, fused.arousal)

        return FrameResult(
            timestamp=timestamp,
            face_detected=reading.detected,
            face_valence=reading.valence,
            face_arousal=reading.arousal,
            face_confidence=reading.confidence,
            hr_bpm=hr_bpm,
            hr_quality=hr_quality,
            fused_valence=fused.valence,
            fused_arousal=fused.arousal,
            face_weight=fused.face_weight,
        )

    def instability_report(self) -> InstabilityReport:
        return self.instability.report()

    def close(self) -> None:
        self.face_analyzer.close()


def _draw_overlay(cv2, frame, r: FrameResult, reading: FaceReading | None, brightness: float) -> None:
    """프레임 위에 얼굴 박스, rPPG 측정 영역, 현재 수치를 그린다. (cv2.putText는 한글이 안 나와 영어로 표시)"""
    green, red, yellow, cyan = (0, 200, 0), (0, 0, 255), (0, 220, 255), (255, 200, 0)
    if reading is not None and reading.detected:
        for name, color in (("face", green), ("forehead", yellow), ("left_cheek", cyan), ("right_cheek", cyan)):
            box = reading.boxes.get(name)
            if box:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2 if name == "face" else 1)
        lines = [
            "FACE: detected",
            f"valence {r.face_valence:+.2f}  confidence {r.face_confidence:.2f}",
            (f"HR {r.hr_bpm:.0f} bpm (q {r.hr_quality:.2f})" if r.hr_bpm else "HR: collecting (need 20s)"),
        ]
        color = green
    else:
        lines = ["FACE: NOT detected", "face the camera, add light, move closer"]
        color = red
    if brightness < 15:
        lines.append("DARK FRAME: check camera permission/index")
        color = red
    for k, text in enumerate(lines):
        cv2.putText(frame, text, (10, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, 28 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def run_on_video(model_path: str, source: str | int, resting_hr_bpm: float = 70.0,
                  max_seconds: float | None = None, print_every: float | None = 2.0,
                  show: bool = True) -> tuple[list[FrameResult], InstabilityReport]:
    """source: 웹캠이면 카메라 번호(정수), 영상 파일이면 경로(str).
    print_every초마다 현재 상태를 터미널에 한 줄씩 출력한다 (None이면 출력 안 함).
    show=True면 카메라 화면 창을 띄운다 (q 또는 Esc로 종료).
    Ctrl+C나 q로 중간에 멈추면 그때까지 쌓인 결과로 리포트를 만든다."""
    import cv2  # 로컬 실행 시에만 필요

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"영상 소스를 열 수 없습니다: {source}  (카메라 권한/번호를 확인하세요: --camera=1)")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipeline = EmotionPipeline(model_path, fps=fps, resting_hr_bpm=resting_hr_bpm)

    results: list[FrameResult] = []
    frame_idx = 0
    next_print = 0.0
    warned_dark = False
    window = "emotion_pipeline (q: quit)"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("카메라에서 프레임을 읽지 못했습니다 (카메라 권한 또는 카메라 번호 확인).")
                break
            timestamp = frame_idx / fps
            r = pipeline.process_frame(frame, timestamp)
            results.append(r)
            frame_idx += 1

            brightness = float(frame.mean())
            if brightness < 15 and not warned_dark and frame_idx > 5:
                warned_dark = True
                print("⚠️ 화면이 거의 검정입니다 — 터미널 앱의 카메라 권한(시스템 설정 > 개인정보 보호 및 보안 > 카메라)이나 "
                      "카메라 번호(--camera=1)를 확인하세요.", flush=True)

            if print_every and timestamp >= next_print:
                next_print = timestamp + print_every
                hr = f"{r.hr_bpm:5.1f}bpm(품질 {r.hr_quality:.2f})" if r.hr_bpm else "  측정 대기(20초 필요)"
                face = "얼굴 O" if r.face_detected else "얼굴 X"
                print(f"[{timestamp:5.1f}s] {face} | 표정 정서가 {r.face_valence:+.2f} 신뢰도 {r.face_confidence:.2f} "
                      f"| 심박 {hr} | 융합 정서가 {r.fused_valence:+.2f} 각성도 {r.fused_arousal:+.2f}", flush=True)

            if show:
                _draw_overlay(cv2, frame, r, pipeline.last_reading, brightness)
                cv2.imshow(window, frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    print("\n창에서 종료(q)했습니다 — 지금까지의 결과로 요약합니다.")
                    break
            if max_seconds and timestamp >= max_seconds:
                break
    except KeyboardInterrupt:
        print("\n중단됨 — 지금까지의 결과로 요약합니다.")
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
        pipeline.close()

    return results, pipeline.instability_report()


if __name__ == "__main__":
    import sys

    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 1:
        print("사용법: python -m emotion_pipeline.pipeline <face_landmarker.task 경로> [영상파일 경로] "
              "[--camera=0] [--seconds=40] [--no-show]")
        sys.exit(1)

    def flag_value(name: str) -> str | None:
        for f in flags:
            if f.startswith(f"--{name}="):
                return f.split("=", 1)[1]
        return None

    model_path = args[0]
    source: str | int = args[1] if len(args) > 1 else int(flag_value("camera") or 0)
    is_webcam = not isinstance(source, str)
    # 웹캠은 끝이 없으므로 기본 40초만 측정 (심박은 20초 뒤부터 나옴). 영상 파일은 끝까지.
    seconds_opt = flag_value("seconds")
    seconds = float(seconds_opt) if seconds_opt else (40.0 if is_webcam else None)

    if is_webcam:
        print(f"웹캠(번호 {source}) 측정 시작 — 화면 창이 뜹니다. 정면을 보세요 (창에서 q 또는 Ctrl+C로 중단)")
    results, report = run_on_video(model_path, source, max_seconds=seconds, show="--no-show" not in flags)

    n = len(results)
    detected = [r for r in results if r.face_detected]
    hrs = [r.hr_bpm for r in results if r.hr_bpm]
    print("\n" + "=" * 50)
    print(f"처리한 프레임 수: {n}  (얼굴 인식된 프레임 {len(detected)}/{n})")
    if n and not detected:
        print("→ 얼굴이 한 번도 인식되지 않았습니다. 화면 창에서 내 얼굴이 보이는지, 카메라 번호가 맞는지(--camera=1) 확인하세요.")
    if hrs:
        print(f"심박수 마지막 추정: {hrs[-1]:.1f} bpm (이 값을 손목시계/맥박계 값과 비교해 보세요)")
    else:
        print("심박수: 측정 못 함 (20초 이상 얼굴이 계속 인식되어야 함)")
    print(f"감정 기복(MSSD) — valence: {report.valence_mssd:.4f}, arousal: {report.arousal_mssd:.4f}")
    print(f"단순 분산(SD)   — valence: {report.valence_sd:.4f}, arousal: {report.arousal_sd:.4f}")