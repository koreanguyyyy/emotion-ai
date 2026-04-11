# detection/stt_detector.py
import sounddevice as sd
import soundfile as sf
import whisper
import os
from state.emotion_state import update_state, stop_event
from ai.stress_calculator import add_stt_text

SAMPLE_RATE = 16000
DURATION = 5
TEMP_PATH = "temp_audio.wav"

print("Whisper 모델 로딩 중...")
whisper_model = whisper.load_model("medium")
print("Whisper 로딩 완료!")

def run_stt():
    print("[STT] 음성 감지 시작...")
    while not stop_event.is_set():
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        sf.write(TEMP_PATH, audio.flatten(), SAMPLE_RATE)
        result = whisper_model.transcribe(TEMP_PATH, language="ko")
        text = str(result["text"]).strip()

        if os.path.exists(TEMP_PATH):
            os.remove(TEMP_PATH)

        if text:
            add_stt_text(text)
            update_state(stt_text=text)
            print(f"[STT] {text}")
