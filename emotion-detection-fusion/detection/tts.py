# detection/tts.py
"""
음성 출력 (2026-11-04 교체). 예전 macOS `say`(Yuna)는 딱딱해서 신경망 음성을 기본으로 쓴다.

.env
  TTS_ENABLED=true
  TTS_ENGINE=edge         edge(기본, Microsoft 신경망 음성, 무료·인터넷 필요) / openai(OpenAI TTS, 키 필요) / say(맥 기본)
  EDGE_VOICE=ko-KR-SunHiNeural    여성. 남성: ko-KR-InJoonNeural, ko-KR-HyunsuMultilingualNeural
  EDGE_RATE=+5%                   말하기 속도 (-20% ~ +30%)
  OPENAI_API_KEY=...              TTS_ENGINE=openai일 때 (감지 프로그램 .env에 따로 넣어야 함)
  OPENAI_TTS_MODEL=gpt-4o-mini-tts   OPENAI_TTS_VOICE=nova
  TTS_VOICE=Yuna  TTS_RATE=180    say용 (실패했을 때도 이걸로 읽는다)
  pip install edge-tts

동작
- 문장 단위로 나눠서, 다음 문장을 만드는 동안 앞 문장을 재생한다(첫 소리가 빨리 나온다).
  대화 답이 스트리밍으로 오면 문장이 완성되는 대로 speak_part()로 넣는다.
- 신경망 음성이 실패하면(인터넷 끊김 등) 그 문장은 `say`로 읽고, 3번 연속 실패하면 이번 실행 동안 `say`만 쓴다.
- 말하는 동안 SPEAKING이 켜져 있어, 마이크가 에이전트 목소리를 사용자 말로 알아듣지 않는다(was_speaking_since).
"""
from __future__ import annotations

import asyncio
import os
import queue
import re
import subprocess
import tempfile
import threading
import time

from dotenv import load_dotenv

load_dotenv()


def _env(name, default):
    v = os.getenv(name)
    return default if v is None or v.split("#")[0].strip() == "" else v.split("#")[0].strip()


TTS_ENABLED = _env("TTS_ENABLED", "true").lower() == "true"
TTS_ENGINE = _env("TTS_ENGINE", "edge").lower()
EDGE_VOICE = _env("EDGE_VOICE", "ko-KR-SunHiNeural")
EDGE_RATE = _env("EDGE_RATE", "+5%")
OPENAI_TTS_MODEL = _env("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = _env("OPENAI_TTS_VOICE", "nova")
TTS_VOICE = _env("TTS_VOICE", "Yuna")        # say용 한국어 음성
TTS_RATE = int(_env("TTS_RATE", "180"))
MAX_CHARS = 600

# 에이전트가 말하는 동안 마이크가 그 소리를 사용자 말로 알아듣지 않게, 말하는 시간을 기록한다.
SPEAKING = threading.Event()
_last_speak_end = 0.0
ECHO_TAIL_SEC = 0.6                 # 말이 끝난 뒤 잔향까지 무시하는 시간


def was_speaking_since(t: float) -> bool:
    """t 이후로 에이전트가 말한 적이 있으면 True (그 사이 녹음된 소리는 버린다)."""
    return SPEAKING.is_set() or _last_speak_end + ECHO_TAIL_SEC > t


# ── 문장 나누기 ───────────────────────────────────────────────
_END_RE = re.compile(r"(?<=[.!?…~。])\s+|\n+")
MIN_SENTENCE = 8   # 너무 짧은 조각("네.")은 다음 문장과 붙여 읽는다


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _END_RE.split(text or "") if p and p.strip()]
    out: list[str] = []
    for p in parts:
        if out and len(out[-1]) < MIN_SENTENCE:
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out


class SentenceBuffer:
    """스트리밍 조각을 받아 완성된 문장을 내놓는다. 끝나면 flush()로 나머지."""

    def __init__(self):
        self.buf = ""

    def feed(self, piece: str) -> list[str]:
        self.buf += piece
        out, pos = [], 0
        while True:
            m = _END_RE.search(self.buf, pos)
            if not m:
                break
            sent = self.buf[:m.start()].strip()
            if len(sent) >= MIN_SENTENCE:          # 충분히 길면 내보낸다
                out.append(sent)
                self.buf, pos = self.buf[m.end():], 0
            else:                                  # 짧으면 다음 문장 끝까지 붙여서 본다
                pos = m.end()
        return out

    def flush(self) -> str:
        rest, self.buf = self.buf.strip(), ""
        return rest


# ── 합성 / 재생 ───────────────────────────────────────────────
_tmpdir = tempfile.mkdtemp(prefix="agent_tts_")
_fail_count = 0
_engine = TTS_ENGINE


def _synth_edge(text: str, path: str) -> None:
    import edge_tts   # pip install edge-tts

    async def run():
        await asyncio.wait_for(edge_tts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE).save(path), timeout=15)
    asyncio.run(run())


def _synth_openai(text: str, path: str) -> None:
    import requests
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 감지 프로그램 .env에 없습니다")
    r = requests.post("https://api.openai.com/v1/audio/speech", timeout=20,
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": OPENAI_TTS_MODEL, "voice": OPENAI_TTS_VOICE, "input": text,
                            "response_format": "mp3"})
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)


def synth_audio(text: str):
    """("file", 경로) 또는 실패 시 ("say", 문장)."""
    global _fail_count, _engine
    if _engine == "say":
        return ("say", text)
    path = os.path.join(_tmpdir, f"{time.time_ns()}.mp3")
    try:
        (_synth_openai if _engine == "openai" else _synth_edge)(text, path)
        if os.path.getsize(path) < 200:
            raise RuntimeError("빈 음성 파일")
        _fail_count = 0
        return ("file", path)
    except Exception as e:
        _fail_count += 1
        print(f"[TTS] {_engine} 음성 실패({e.__class__.__name__}: {e}) → 이 문장은 say로 읽음")
        if _fail_count >= 3:
            print(f"[TTS] {_engine}이 계속 실패해서 이번 실행 동안 say를 씁니다 (인터넷·설정 확인)")
            _engine = "say"
        return ("say", text)


def play_audio(item) -> None:
    kind, value = item
    try:
        if kind == "file":
            subprocess.run(["afplay", value], check=True, timeout=120)
        else:
            subprocess.run(["say", "-v", TTS_VOICE, "-r", str(TTS_RATE), value], check=True, timeout=120)
    except FileNotFoundError:
        print("[TTS] afplay/say 명령어를 찾을 수 없습니다 (macOS 전용)")
    except Exception as e:
        print(f"[TTS] 재생 실패: {e}")
    finally:
        if kind == "file":
            try:
                os.remove(value)
            except OSError:
                pass


class Speaker:
    """문장을 넣으면 순서대로 합성·재생한다. 합성은 재생보다 한 발 앞서 간다."""

    def __init__(self, synth=synth_audio, play=play_audio):
        self.synth, self.play = synth, play
        self.text_q: queue.Queue = queue.Queue()
        self.audio_q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.pending = 0
        self.idle = threading.Event()
        self.idle.set()
        threading.Thread(target=self._synth_worker, daemon=True).start()
        threading.Thread(target=self._play_worker, daemon=True).start()

    def put(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self.lock:
            self.pending += 1
            self.idle.clear()
            SPEAKING.set()
        self.text_q.put(text)

    def wait(self, timeout: float = 180) -> bool:
        return self.idle.wait(timeout)

    def _synth_worker(self):
        while True:
            t = self.text_q.get()
            try:
                item = self.synth(t)
            except Exception as e:
                print(f"[TTS] 합성 오류: {e}")
                item = ("say", t)
            self.audio_q.put(item)

    def _play_worker(self):
        global _last_speak_end
        while True:
            item = self.audio_q.get()
            try:
                self.play(item)
            finally:
                with self.lock:
                    self.pending -= 1
                    if self.pending <= 0:
                        self.pending = 0
                        _last_speak_end = time.time()
                        SPEAKING.clear()
                        self.idle.set()


_speaker: Speaker | None = None


def _get_speaker() -> Speaker:
    global _speaker
    if _speaker is None:
        _speaker = Speaker()
        print(f"[TTS] 음성: {TTS_ENGINE}"
              + (f" ({EDGE_VOICE}, 속도 {EDGE_RATE})" if TTS_ENGINE == "edge" else
                 f" ({OPENAI_TTS_MODEL}, {OPENAI_TTS_VOICE})" if TTS_ENGINE == "openai" else f" ({TTS_VOICE})"))
    return _speaker


def speak_part(text: str) -> None:
    """한 문장을 재생 줄에 넣고 바로 돌아온다 (스트리밍 답용)."""
    if TTS_ENABLED and text and text.strip():
        _get_speaker().put(text)


def speak_wait(timeout: float = 180) -> bool:
    """넣어 둔 문장을 다 읽을 때까지 기다린다."""
    return True if _speaker is None else _speaker.wait(timeout)


def speak(text: str) -> bool:
    """텍스트를 음성으로 읽고 끝날 때까지 기다린다. 반환값: 읽었는지."""
    if not TTS_ENABLED:
        print("[TTS] 비활성화 상태 → 스킵")
        return False
    text = (text or "").strip()
    if not text:
        return False
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "..."
    for s in split_sentences(text):
        speak_part(s)
    speak_wait()
    print("[TTS] 재생 완료")
    return True


def get_available_voices() -> list:
    """say용 한국어 음성 목록 확인"""
    try:
        result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
        return [line.split()[0] for line in result.stdout.splitlines() if "ko_" in line or "Korean" in line]
    except Exception:
        return []
