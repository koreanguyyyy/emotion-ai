# detection/stt_detector.py
import sounddevice as sd
import soundfile as sf
import whisper
import os
import time
import numpy as np
import webrtcvad
from state.emotion_state import update_state, stop_event
from ai.stress_calculator import add_stt_text
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 16000
TEMP_PATH = "temp_audio.wav"

# .env에서 방법 선택
# STT_MODE=fixed   → 고정 길이 녹음 (방법 2)
# STT_MODE=vad     → VAD 기반 녹음 (방법 1)
STT_MODE = os.getenv("STT_MODE", "fixed")
DURATION = int(os.getenv("STT_DURATION", 10))        # fixed 모드용
VAD_SILENCE_SEC = float(os.getenv("VAD_SILENCE_SEC", 1.5))  # vad 모드용 무음 감지 시간

print("Whisper 모델 로딩 중...")
whisper_model = whisper.load_model("medium")
print("Whisper 로딩 완료!")

# ── 공통: STT 변환 ─────────────────────────────────────
def transcribe(audio: np.ndarray) -> str:
    sf.write(TEMP_PATH, audio, SAMPLE_RATE)
    result = whisper_model.transcribe(TEMP_PATH, language="ko")
    text = str(result["text"]).strip()

    if os.path.exists(TEMP_PATH):
        os.remove(TEMP_PATH)

    return text

# ── 방법 2: 고정 길이 녹음 ─────────────────────────────
def run_stt_fixed():
    print(f"[STT-Fixed] {DURATION}초 단위 녹음 시작...")
    while not stop_event.is_set():
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        text = transcribe(audio.flatten())

        if text:
            add_stt_text(text)
            update_state(stt_text=text)
            print(f"[STT-Fixed] {text}")

# ── 방법 1: VAD 기반 녹음 ──────────────────────────────
def run_stt_vad():
    print(f"[STT-VAD] 말소리 감지 대기 중... (무음 {VAD_SILENCE_SEC}초 후 STT 처리)")

    vad = webrtcvad.Vad(2)  # 민감도 0~3 (높을수록 민감)

    FRAME_DURATION = 30   # ms (10, 20, 30만 지원)
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)  # 프레임당 샘플 수
    SILENCE_FRAMES = int(VAD_SILENCE_SEC * 1000 / FRAME_DURATION)  # 무음 판단 프레임 수

    recorded_frames = []
    silence_count = 0
    is_speaking = False

    def callback(indata, frames, time_info, status):
        nonlocal silence_count, is_speaking

        frame = (indata[:, 0] * 32768).astype(np.int16).tobytes()

        try:
            speech_detected = vad.is_speech(frame, SAMPLE_RATE)
        except Exception:
            speech_detected = False

        if speech_detected:
            is_speaking = True
            silence_count = 0
            recorded_frames.append(indata[:, 0].copy())
        else:
            if is_speaking:
                silence_count += 1
                recorded_frames.append(indata[:, 0].copy())

                # 무음이 SILENCE_FRAMES 이상 지속되면 발화 종료
                if silence_count >= SILENCE_FRAMES:
                    process_audio()

    def process_audio():
        nonlocal recorded_frames, silence_count, is_speaking

        if not recorded_frames:
            return

        audio = np.concatenate(recorded_frames)
        recorded_frames.clear()
        silence_count = 0
        is_speaking = False

        text = transcribe(audio)
        if text:
            add_stt_text(text)
            update_state(stt_text=text)
            print(f"[STT-VAD] {text}")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SIZE,
        callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)

# ── 진입점 ─────────────────────────────────────────────
def run_stt():
    if STT_MODE == "vad":
        run_stt_vad()
    else:
        run_stt_fixed()