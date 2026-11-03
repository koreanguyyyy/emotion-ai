# tests/test_web_link.py
"""
웹페이지 연동 검증 (2026-11-03): 실시간 상태 통로, 백엔드 저장 항목, 1분 주기 저장.
실행: python tests/test_web_link.py   (카메라·서버 불필요)
"""
import json
import os
import sys
import types
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.makedirs("logs", exist_ok=True)

from ai.live_server import build_live_payload, start_live_server
from ai.stress_index import StressReading

passed = 0


def check(cond, msg):
    global passed
    assert cond, "실패: " + msg
    passed += 1
    print("  ✓", msg)


reading = StressReading(index=7.04, neg=1.0, hr_up=0.95, arousal=0.12, instab=0.3, valence_mean=-0.4,
                        arousal_mean=0.1, mssd=0.01, face_confidence=0.9, label="슬픔", n_frames=300)
state = {"face_detected": True, "face_emotion": "슬픔", "face_confidence": 0.9, "fused_valence": -0.36,
         "fused_arousal": 0.3, "hr_bpm": 108.0, "hr_quality": 0.52, "resting_hr": 89.0, "hr_mode": "fusion",
         "face_baseline_ready": True, "face_method": "rule", "model_label": "슬픔", "stress_level": 8,
         "stt_text": "오늘 너무 힘들다", "last_intervention_id": 12}

print("== 1. 실시간 상태 통로")
payload = build_live_payload(state, reading)
check(payload["emotionKey"] == "sad" and payload["faceEmotion"] == "슬픔",
      "한국어 라벨과 웹페이지용 영어 키를 함께 보냄 (슬픔 → sad)")
check(payload["stressIndex"] == 7.04 and payload["stressHr"] == 0.95 and payload["heartRate"] == 108.0,
      "스트레스 지수·구성·심박을 그대로 보냄")
empty = build_live_payload({}, None)
check(empty["stressIndex"] is None and empty["emotionKey"] == "neutral" and empty["faceDetected"] is False,
      "아직 계산 전이면 빈 값(None)과 무감정으로 보냄")

srv = start_live_server(18765, lambda: build_live_payload(state, reading))
with urllib.request.urlopen("http://127.0.0.1:18765/state", timeout=2) as r:
    body = json.loads(r.read().decode("utf-8"))
    cors = r.headers.get("Access-Control-Allow-Origin")
check(body["heartRate"] == 108.0 and cors == "*", "HTTP로 읽힘, 웹페이지(다른 포트)에서 읽도록 CORS 허용")
try:
    urllib.request.urlopen("http://127.0.0.1:18765/other", timeout=2)
    check(False, "")
except urllib.error.HTTPError as e:
    check(e.code == 404, "/state 외 주소는 404")
srv.shutdown()
check(start_live_server(0, lambda: {}) is None, "LIVE_PORT=0이면 열지 않음")

print("== 2. 백엔드 저장 항목")
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *a, **k: None))
import backend.api_client as api
sent = {}

class Resp:
    def raise_for_status(self): pass
    def json(self): return {"data": {"emotionLogId": 77}}

def fake_post(url, json=None, timeout=None):
    sent["url"], sent["json"] = url, json
    return Resp()

import main as M
fields = M.fusion_fields(state, reading, "periodic")
with mock.patch.object(api.requests, "post", fake_post):
    eid = api.save_emotion_log(face_emotion="슬픔", face_confidence=0.9, speech_text="", vad_stress_score=7.04,
                               final_stress_level=7, extra=fields)
j = sent["json"]
check(eid == 77 and j["primaryEmotion"] == "슬픔" and j["finalStressLevel"] == 7,
      "감정·최종 스트레스를 채워서 저장 (이전에는 최종 스트레스가 비어 웹페이지에 0으로 보임)")
need = {"heartRate", "restingHeartRate", "stressNeg", "stressHr", "stressArousal", "stressInstab",
        "hrMode", "faceMethod", "modelEmotion", "logType"}
check(need <= set(j) and j["heartRate"] == 108.0 and j["logType"] == "periodic",
      "백엔드에 추가한 10개 항목 이름 그대로 보냄 (Java DTO 필드명과 같음)")
fields2 = M.fusion_fields({"hr_mode": "face_only"}, None, "threshold")
with mock.patch.object(api.requests, "post", fake_post):
    api.save_emotion_log(face_emotion="무감정", face_confidence=0.1, speech_text="", vad_stress_score=1.0,
                         final_stress_level=1, extra=fields2)
check("heartRate" not in sent["json"] and sent["json"]["hrMode"] == "face_only",
      "모르는 값(심박 없음 등)은 보내지 않음")

print("== 3. 1분 주기 저장")
calls = []
M.LATEST["reading"] = reading
waits = iter([False, False, True])   # 두 번 저장한 뒤 종료
with mock.patch.object(M, "EMOTION_LOG_EVERY_SEC", 60), \
     mock.patch.object(M.stop_event, "wait", lambda t: next(waits)), \
     mock.patch.object(M, "get_state", lambda: state), \
     mock.patch.object(M, "save_emotion_log", lambda **kw: calls.append(kw)):
    M.run_periodic_emotion_log()
check(len(calls) == 2 and calls[0]["final_stress_level"] == 7 and calls[0]["extra"]["logType"] == "periodic",
      "정해진 간격마다 저장, 최종 스트레스 = 지수 반올림")
calls.clear(); waits = iter([False, True])
with mock.patch.object(M.stop_event, "wait", lambda t: next(waits)), \
     mock.patch.object(M, "get_state", lambda: {**state, "face_detected": False}), \
     mock.patch.object(M, "save_emotion_log", lambda **kw: calls.append(kw)):
    M.run_periodic_emotion_log()
check(calls == [], "얼굴이 없을 때는 저장하지 않음")

print(f"\n모든 검증 통과: {passed}개")
