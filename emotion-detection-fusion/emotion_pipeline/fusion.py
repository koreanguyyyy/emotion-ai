"""
적응형 융합 — 교수님 피드백 1번("작은 표정만으로는 안 된다")의 직접적인 구현.

설계(적용방안 문서 3절 '계층적 역할 분담'을 그대로 따름):
- valence(기분 좋고 나쁨)는 표정 쪽이 유일한 정보원이다. 생체신호는 밸런스를
  신뢰성 있게 주지 못한다는 게 문헌 결론이므로, 표정 신뢰도가 낮을 때는
  "모른다(0, 중립)" 쪽으로 당기는 것이 맥박로 억지로 메우는 것보다 정직하다.
- arousal(각성/흥분 정도)는 표정과 맥박(HR) 둘 다 정보를 준다. 여기서
  신뢰도 기반 가중 평균을 적용한다 — 표정이 뚜렷하면 표정, 애매하면 맥박 쪽으로.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusedAffect:
    valence: float
    arousal: float
    face_weight: float  # 이번 프레임에서 arousal 융합에 표정이 쓰인 비중


def hr_to_arousal(bpm: float, resting_bpm: float, hr_range: float = 40.0) -> float:
    """심박수를 각성도 점수로 변환. resting_bpm(평상시 심박수) 대비 얼마나
    올라갔는지를 [-1, 1] 근방으로 정규화한다. hr_range는 '이 정도 차이면
    각성도가 꽉 찬다'고 볼 폭(기본 40bpm)."""
    if bpm != bpm:  # NaN 체크
        return 0.0
    return max(-1.0, min(1.0, (bpm - resting_bpm) / hr_range))


def adaptive_fuse(face_valence: float, face_arousal: float, face_confidence: float,
                   bio_arousal: float, w_max: float = 0.7) -> FusedAffect:
    """
    face_confidence가 높을수록(표정이 뚜렷할수록) arousal 추정에서 표정 비중이
    커지고, 낮을수록(작은 표정) 맥박 기반 각성도 쪽으로 자동 이동한다.

    valence는 표정 신뢰도로 직접 스케일링한다 — 신뢰도가 0이면 valence도 0(중립)
    으로 수렴. "정보가 없으면 아무 것도 모른다고 말하는" 쪽이 틀린 확신보다 낫다.
    """
    face_weight = w_max * max(0.0, min(1.0, face_confidence))
    arousal = face_weight * face_arousal + (1.0 - face_weight) * bio_arousal
    valence = face_confidence * face_valence
    return FusedAffect(valence=valence, arousal=arousal, face_weight=face_weight)
