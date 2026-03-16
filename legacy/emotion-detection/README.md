# emotion-detection

감정 인식 멀티모달 감지 엔진

## 모듈 구성
| 모듈 | 설명 |
|------|------|
| `face/` | DeepFace 표정 인식 |
| `audio/` | Whisper STT + librosa 한숨 감지 |
| `motion/` | OpenCV 동작 패턴 감지 |
| `core/` | 감지 결과 통합 & ai-server 전송 |

## 실행 방법
```bash
pip install -r requirements.txt
python core/main.py
```

## Privacy-by-Design
- 원시 오디오/영상은 처리 즉시 삭제
- 외부 서버로 원본 데이터 전송 없음
