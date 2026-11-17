# detection/face_detector.py
import cv2
from deepface import DeepFace
from state.emotion_state import update_state, get_state, stop_event

ANALYZE_EVERY = 10

EMOTION_KO = {
    "happy": "행복",
    "sad": "슬픔",
    "angry": "분노",
    "fear": "공포",
    "disgust": "혐오",
    "surprise": "놀람",
    "neutral": "무감정"
}

def run_face_detection():
    cap = cv2.VideoCapture(1)
    print("[Face] 얼굴 감지 시작...")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % ANALYZE_EVERY == 0:
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    detector_backend="mtcnn",
                    enforce_detection=False
                )
                data = result[0] if isinstance(result, list) else result
                dominant = data["dominant_emotion"] # type: ignore
                emotion_ko = EMOTION_KO.get(dominant, dominant)
                scores = data["emotion"] # type: ignore

                update_state(
                    face_emotion=emotion_ko,
                    emotion_scores=scores
                )

            except Exception:
                pass

        # 화면 표시
        state = get_state()
        emotion_ko = state["face_emotion"] or "감지 중..."
        stress = state["stress_level"]

        label = f"감정: {emotion_ko}"
        if stress is not None:
            label += f"  |  스트레스: {stress}/10"

        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Multimodal Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()
