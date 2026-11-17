"""
합성(synthetic) 데이터로 각 모듈의 수학적 로직을 검증한다.

⚠️ 정직하게 밝혀둘 것: 이 테스트는 '진짜 얼굴 영상'을 쓰지 않는다. 이 샌드박스에는
카메라가 없고, MediaPipe 모델 파일(face_landmarker.task) 다운로드 경로도 막혀 있다.
그래서 검증 범위는 다음과 같이 나뉜다.

  ✅ 직접 실행해서 검증한 것 (이 파일):
     - CHROM 알고리즘 수식 (rppg.py) — 합성 RGB 시계열에서 정답 심박수 복원
     - 블렌드셰입 → valence/arousal/confidence 매핑 (roi.py) — 가상 표정 점수로 검증
     - 적응형 융합 (fusion.py) — 신뢰도에 따른 가중치 전환
     - 감정 기복 MSSD (instability.py) — SD와의 구분력

  ❌ 아직 검증 못 한 것 (로컬 웹캠 필요):
     - 실제 얼굴에서 MediaPipe가 랜드마크/블렌드셰입을 정확히 뽑는지
     - roi.py의 ROI 좌표 비율(이마/볼 박스)이 실제 얼굴에 잘 맞는지
     - 실제 조명/움직임 환경에서 CHROM 심박수 오차가 문헌 수준(~5bpm)인지

  → 로컬에서 face_landmarker.task를 받고 웹캠으로 pipeline.py를 돌려보면
    이 부분까지 채워진다 (README.md 참고).
"""
from __future__ import annotations

import numpy as np

from rppg import CHROMExtractor, chrom_pulse, estimate_hr
from roi import blendshapes_to_affect
from fusion import adaptive_fuse, hr_to_arousal
from instability import InstabilityTracker, mssd


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ── 1. CHROM 알고리즘: 합성 RGB → 심박수 복원 ──────────────────────
def test_chrom() -> None:
    section("1. rPPG CHROM — 합성 신호에서 심박수 복원")
    np.random.seed(0)
    fs, duration, true_bpm = 30, 25, 78.0
    t = np.arange(0, duration, 1 / fs)
    freq = true_bpm / 60
    base = np.array([180.0, 130.0, 110.0])
    gain = np.array([0.3, 1.0, 0.6])
    pulse = np.sin(2 * np.pi * freq * t)
    rgb = base[None, :] + 0.6 * pulse[:, None] * gain[None, :]
    rgb += np.random.normal(0, 0.8, rgb.shape)
    rgb += (2.0 * np.sin(2 * np.pi * 0.05 * t))[:, None]  # 저주파 조명 흔들림

    pulse_signal = chrom_pulse(rgb[:, 0], rgb[:, 1], rgb[:, 2], fs)
    est = estimate_hr(pulse_signal, fs)
    err = abs(est.bpm - true_bpm)
    print(f"정답 {true_bpm:.1f}bpm → 추정 {est.bpm:.1f}bpm (오차 {err:.2f}bpm, 품질 {est.quality:.2f})")
    assert err < 2.0, "CHROM 추정 오차가 너무 큼"

    # 스트리밍 래퍼(CHROMExtractor)도 같은 결과를 내는지 확인
    section("1-1. CHROMExtractor 스트리밍 버퍼 동작 확인")
    extractor = CHROMExtractor(fs=fs, window_sec=20.0, step_sec=1.0)
    last = None
    for i in range(len(t)):
        last = extractor.update(tuple(rgb[i]))
    assert last is not None, "윈도우가 찼는데도 추정치가 안 나옴"
    print(f"스트리밍 추정 {last.bpm:.1f}bpm (오차 {abs(last.bpm - true_bpm):.2f}bpm)")
    assert abs(last.bpm - true_bpm) < 2.0


# ── 2. 블렌드셰입 → valence/arousal/confidence ─────────────────────
def test_blendshapes() -> None:
    section("2. 블렌드셰입 → valence/arousal/confidence 매핑")

    neutral = {}
    v, a, c = blendshapes_to_affect(neutral)
    print(f"무표정            : valence={v:+.2f} arousal={a:+.2f} confidence={c:.2f}")
    assert c < 0.05, "무표정인데 신뢰도가 0이 아님"

    small_smile = {"mouthSmileLeft": 0.08, "mouthSmileRight": 0.08}
    v, a, c = blendshapes_to_affect(small_smile)
    print(f"작은 미소(교수님 지적 사례): valence={v:+.2f} arousal={a:+.2f} confidence={c:.2f}")
    assert c < 0.3, "작은 표정인데 신뢰도가 너무 높게 나옴"

    big_smile = {"mouthSmileLeft": 0.9, "mouthSmileRight": 0.9, "cheekSquintLeft": 0.4, "cheekSquintRight": 0.4}
    v, a, c = blendshapes_to_affect(big_smile)
    print(f"뚜렷한 미소        : valence={v:+.2f} arousal={a:+.2f} confidence={c:.2f}")
    assert v > 0.5 and c > 0.8

    anger = {"browDownLeft": 0.9, "browDownRight": 0.9, "noseSneerLeft": 0.8, "noseSneerRight": 0.8}
    v, a, c = blendshapes_to_affect(anger)
    print(f"분노 표정          : valence={v:+.2f} arousal={a:+.2f} confidence={c:.2f}")
    assert v < -0.3, "분노 표정인데 valence가 충분히 음수가 아님"

    surprise = {"eyeWideLeft": 0.9, "eyeWideRight": 0.9, "jawOpen": 0.8, "browInnerUp": 0.7}
    v, a, c = blendshapes_to_affect(surprise)
    print(f"놀람 표정          : valence={v:+.2f} arousal={a:+.2f} confidence={c:.2f}")
    assert a > 0.5, "놀람 표정인데 arousal이 높지 않음"


# ── 3. 적응형 융합: 표정 신뢰도에 따른 가중치 전환 ──────────────────
def test_fusion() -> None:
    section("3. 적응형 융합 — 표정이 애매할수록 생체신호 비중 증가")
    bio_arousal = hr_to_arousal(bpm=95, resting_bpm=70)  # 평소보다 빠른 심박
    print(f"심박 기반 arousal : {bio_arousal:+.2f}")

    cases = [
        ("뚜렷한 표정", 0.9),
        ("보통 표정", 0.5),
        ("작은 표정(교수님 지적)", 0.08),
        ("완전 무표정", 0.0),
    ]
    for name, conf in cases:
        fused = adaptive_fuse(face_valence=0.6, face_arousal=0.2, face_confidence=conf, bio_arousal=bio_arousal)
        print(f"{name:20s} conf={conf:.2f} → face_weight={fused.face_weight:.2f}  "
              f"fused_valence={fused.valence:+.2f}  fused_arousal={fused.arousal:+.2f}")

    # 핵심 주장 검증: 신뢰도가 낮을수록 face_weight가 단조 감소해야 함
    weights = [adaptive_fuse(0.6, 0.2, c, bio_arousal).face_weight for _, c in cases]
    assert weights == sorted(weights, reverse=True), "신뢰도 감소에 따라 face_weight가 단조감소하지 않음"


# ── 4. 감정 기복 MSSD: SD와의 구분력 재확인 ─────────────────────────
def test_mssd() -> None:
    section("4. MSSD — SD로는 구별 안 되는 두 패턴을 구별하는지")
    stable = [0.5, 0.52, 0.5, 0.51, 0.49, 0.5]
    swingy = [0.5, -0.4, 0.6, -0.5, 0.55, -0.45]
    broad_slow = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]

    for name, series in [("안정", stable), ("자주 급변", swingy), ("넓지만 완만", broad_slow)]:
        print(f"{name:10s} MSSD={mssd(np.array(series)):.4f}  SD={np.std(series):.4f}")

    assert mssd(np.array(broad_slow)) < mssd(np.array(swingy)) / 5, \
        "완만한 변화와 급변하는 변화를 MSSD가 구별 못 함"


# ── 5. 전체 흐름 통합 (실시간 프레임 루프 모사) ─────────────────────
def test_end_to_end_timeseries() -> None:
    section("5. 통합 — 표정+맥박을 20초간 흘려보내며 기복 리포트 생성")
    tracker = InstabilityTracker()
    rng = np.random.default_rng(1)
    bio_arousal = 0.3
    t = 0.0
    # 10초는 안정, 이후 10초는 급변하는 표정 시나리오를 흘려보낸다
    for i in range(200):
        t = i * 0.1
        if t < 10:
            face_v, face_c = 0.1 + rng.normal(0, 0.02), 0.6
        else:
            face_v, face_c = rng.choice([-0.8, 0.8]) + rng.normal(0, 0.05), 0.6
        fused = adaptive_fuse(face_valence=face_v, face_arousal=0.1, face_confidence=face_c, bio_arousal=bio_arousal)
        tracker.add(t, fused.valence, fused.arousal)

    report = tracker.report()
    print(f"전체 {report.n}개 샘플 → valence MSSD={report.valence_mssd:.4f}, SD={report.valence_sd:.4f}")
    print("→ 10초 지점부터 감정이 급변하도록 설계했으므로 MSSD가 뚜렷하게 커야 한다")
    assert report.valence_mssd > 0.1


if __name__ == "__main__":
    test_chrom()
    test_blendshapes()
    test_fusion()
    test_mssd()
    test_end_to_end_timeseries()
    print("\n" + "=" * 60)
    print("모든 합성 데이터 테스트 통과")
    print("=" * 60)
