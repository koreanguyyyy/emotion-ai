#!/bin/bash
# 설정 바꾸기: ./use_env.sh present | visitor | exhibit | lab
# 바꾼 뒤 감지 프로그램(python main.py)을 다시 켜야 적용됩니다.
cd "$(dirname "$0")"
case "$1" in
  present|visitor|exhibit|lab) ;;
  *) echo "사용법: ./use_env.sh present(발표) | visitor(관람객) | exhibit(기존 전시) | lab(원래 값)"; exit 1 ;;
esac
[ -f ".env.$1" ] || { echo ".env.$1 파일이 없습니다"; exit 1; }
cp ".env.$1" .env && echo "✅ .env ← .env.$1  (감지 프로그램을 다시 켜 주세요)"
grep -E "^(DETECTION_INTERVAL|FACE_BASELINE_SEC|BASELINE_SEC|HR_CALIB_TIMEOUT_SEC|STRESS_WINDOW_SEC|STRESS_THRESHOLD|CHAT_IDLE_SEC|RESTING_HR)=" .env
