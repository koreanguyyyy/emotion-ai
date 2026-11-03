# ai/cooldown.py  ← detection 레포에 추가
import time
import os

COOLDOWN_MAP = {
    "urgent": 10 * 60,   # 스트레스 9~10 → 10분
    "high":   int(os.getenv("COOLDOWN_MINUTES", 30)) * 60,   # 스트레스 7~8  → 30분
}

last_intervention_time = 0.0

def is_cooldown_active(stress_level: int) -> bool:
    cooldown = COOLDOWN_MAP["urgent"] if stress_level >= 9 else COOLDOWN_MAP["high"]
    return time.time() - last_intervention_time < cooldown

def update_intervention_time():
    global last_intervention_time
    last_intervention_time = time.time()

def get_remaining_cooldown(stress_level: int) -> int:
    cooldown = COOLDOWN_MAP["urgent"] if stress_level >= 9 else COOLDOWN_MAP["high"]
    remaining = cooldown - (time.time() - last_intervention_time)
    return max(0, int(remaining / 60))