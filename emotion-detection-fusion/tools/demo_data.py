# tools/demo_data.py
"""
시연용 샘플 데이터 — 웹페이지(대시보드·감정 이력·개입/추천)가 비어 보이지 않도록 지난 7일치 기록을 만든다.

⚠️ 이 데이터는 실제 측정이 아니라 만들어 낸 샘플이다. 화면 시연용으로만 쓰고, 논문·발표의 결과로 쓰지 않는다.
   모든 행에 demo 표시가 붙어 있어 --clear로 한 번에 지울 수 있다.
   (emotion_logs.log_type='demo', interventions.intervention_type='demo', 추천은 demo 개입에 딸린 것)

사용 (백엔드를 11/3 수정본으로 한 번 켠 뒤에 — 새 열이 그때 생긴다):
  python tools/demo_data.py                  → logs/demo_seed.sql 생성
  mysql -u capstone -p capstone < logs/demo_seed.sql
  python tools/demo_data.py --clear          → logs/demo_clear.sql 생성 (샘플만 지움, 실제 기록은 그대로)
  mysql -u capstone -p capstone < logs/demo_clear.sql
옵션: --days 7 --user 1 --seed 42

만드는 방식
  - 하루 9~22시, 20~40분 간격 감정 기록(실제 프로그램의 1분 주기 저장을 성기게 흉내).
  - 하루 흐름: 오전 평온 → 오후(과제·마감 시간대) 스트레스 상승 → 저녁 회복. 날마다 강도가 다르다.
  - 스트레스 지수는 실제 계산식과 같게 맞춘다: 10 × (0.35 부정 + 0.35 심박 상승 + 0.10 표정 각성 + 0.20 기복).
  - 지수 6 이상이면 개입 기록 + 추천 1~2개. 개입 후 30분 쉬는 시간은 모두 과거에 끝나게 해서 시연 중 실제 개입을 막지 않는다.
  - 추천 링크는 실제 상품 주소가 아니라 Spotify·알라딘·카카오맵 '검색' 주소다.
"""
import argparse
import os
import random
from datetime import datetime, timedelta
from urllib.parse import quote

MUSIC = [("Clair de Lune", "Claude Debussy"), ("밤편지", "아이유"), ("Paris in the Rain", "Lauv"),
         ("River Flows in You", "이루마"), ("봄날", "방탄소년단"), ("Weightless", "Marconi Union")]
BOOKS = [("죽고 싶지만 떡볶이는 먹고 싶어", "백세희"), ("미움받을 용기", "기시미 이치로, 고가 후미타케"),
         ("아주 작은 습관의 힘", "제임스 클리어"), ("나는 나로 살기로 했다", "김수현")]
PLACES = [("근처 공원 산책로", "공원"), ("조용한 동네 카페", "카페"), ("가까운 도서관", "도서관")]
SPEECH = {
    "슬픔": ["오늘 진짜 아무것도 하기 싫다", "또 떨어졌네… 좀 우울하다", "요즘 너무 지친다"],
    "분노": ["아 진짜 왜 안 되는 거야", "코드가 계속 에러 나서 짜증나", "마감이 내일인데 아직 반도 못 했어"],
    "공포": ["발표 내일인데 너무 떨린다", "시험 망하면 어떡하지"],
}
MESSAGES = {
    "슬픔": "오늘 많이 지쳐 보이네요. 잠깐 하던 일을 내려놓고 숨을 고르는 것도 괜찮아요. "
            "마음을 조금 가라앉혀 줄 {rec}을(를) 준비해 봤어요. 천천히 들어 보세요.",
    "분노": "일이 뜻대로 안 풀려서 답답하시죠. 그럴 땐 잠깐 자리에서 일어나 몸을 움직이는 게 도움이 돼요. "
            "{rec}에서 10분만 걸어 보는 건 어때요?",
    "공포": "걱정이 많아 보여요. 지금 느끼는 긴장은 그만큼 잘하고 싶다는 뜻일 거예요. "
            "{rec}로 잠시 마음을 쉬게 해 보세요.",
}


def q(v):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return repr(round(v, 4)) if isinstance(v, float) else str(v)
    if isinstance(v, datetime):
        return "'" + v.strftime("%Y-%m-%d %H:%M:%S") + "'"
    return "'" + str(v).replace("\\", "\\\\").replace("'", "''") + "'"


def stress_index(neg, hr_up, aro, instab):
    return 10 * (0.35 * neg + 0.35 * hr_up + 0.10 * aro + 0.20 * instab)


def rec_row(category, rng):
    if category == "MUSIC":
        t, a = rng.choice(MUSIC)
        return t, a, f"https://open.spotify.com/search/{quote(t + ' ' + a)}", "긴장을 풀어 주는 잔잔한 곡"
    if category == "BOOK":
        t, a = rng.choice(BOOKS)
        return t, a, f"https://www.aladin.co.kr/search/wsearchresult.aspx?SearchWord={quote(t)}", "지친 마음에 위로가 되는 책"
    t, kw = rng.choice(PLACES)
    return t, None, f"https://map.kakao.com/?q={quote(kw)}", "잠시 환기할 수 있는 가까운 장소"


def generate(days, user_id, seed, now):
    rng = random.Random(seed)
    out = ["-- 시연용 샘플 데이터 (실제 측정 아님). 지우기: python tools/demo_data.py --clear",
           "START TRANSACTION;"]
    resting = 72.0
    n_logs = n_iv = 0
    for d in range(days - 1, -1, -1):
        day = (now - timedelta(days=d)).replace(hour=9, minute=0, second=0, microsecond=0)
        peak_hour = rng.uniform(14, 18)
        intensity = rng.uniform(0.7, 1.0) if rng.random() < 0.75 else rng.uniform(0.2, 0.45)
        if d == 0:   # 오늘: 대시보드의 '오늘' 카드가 채워지도록, 지금보다 1~2시간 전에 스트레스 정점을 둔다
            peak_hour = max(10.5, min(peak_hour, now.hour + now.minute / 60 - 1.5))
            intensity = 0.95
        t = day + timedelta(minutes=rng.randint(0, 20))
        last_iv = None
        prev_val = 0.0
        while t.hour < 22:
            if t > now - timedelta(minutes=40):      # 오늘은 지금보다 40분 전까지만 (쉬는 시간이 끝나 있게)
                break
            h = t.hour + t.minute / 60
            s = intensity * max(0.0, 1 - abs(h - peak_hour) / 3.0)          # 0~1 스트레스 곡선
            s = min(1.0, max(0.0, s + rng.gauss(0, 0.08)))
            if s > 0.55:
                emo = rng.choices(["분노", "슬픔", "공포"], [0.45, 0.35, 0.2])[0]
            elif s < 0.15 and rng.random() < 0.25:
                emo = "행복"
            else:
                emo = "무감정"
            neg = 0.0 if emo in ("행복", "무감정") else min(1.0, s * rng.uniform(0.9, 1.2))
            hr_up = min(1.0, max(0.0, s * rng.uniform(0.8, 1.1)))
            aro = min(1.0, max(0.0, s * rng.uniform(0.2, 0.6)))
            val = 0.6 if emo == "행복" else -0.4 * neg
            instab = min(1.0, abs(val - prev_val) * 2 + rng.uniform(0, 0.15) * s)
            prev_val = val
            idx = round(stress_index(neg, hr_up, aro, instab), 2)
            hr = round(resting + hr_up * 20 + rng.gauss(0, 2), 1)
            threshold = idx >= 6 and (last_iv is None or t - last_iv >= timedelta(minutes=30))
            speech = rng.choice(SPEECH[emo]) if threshold and emo in SPEECH else None
            cols = {
                "user_id": user_id, "primary_emotion": emo, "confidence_score": round(rng.uniform(0.6, 0.95), 4),
                "face_emotion": emo, "face_confidence": round(rng.uniform(0.6, 0.95), 4), "speech_text": speech,
                "vad_stress_score": idx, "final_stress_level": round(idx), "sigh_detected": False,
                "motion_pattern": None, "intervention_triggered": bool(threshold), "detected_at": t,
                "heart_rate": hr, "resting_heart_rate": resting, "stress_neg": round(neg, 3),
                "stress_hr": round(hr_up, 3), "stress_arousal": round(aro, 3), "stress_instab": round(instab, 3),
                "hr_mode": "fusion", "face_method": "rule", "model_emotion": emo, "log_type": "demo",
            }
            out.append(f"INSERT INTO emotion_logs ({', '.join(cols)}) VALUES ({', '.join(q(v) for v in cols.values())});")
            n_logs += 1
            if threshold:
                last_iv = t
                cats = {"분노": ["PLACE", "MUSIC"], "슬픔": ["MUSIC", "BOOK"], "공포": ["MUSIC", "BOOK"]}.get(emo, ["MUSIC"])
                recs = [rec_row(c, rng) for c in cats[: rng.choice([1, 2])]]
                msg = MESSAGES.get(emo, MESSAGES["슬픔"]).format(rec=f"‘{recs[0][0]}’")
                recent = (now - t) < timedelta(hours=26)
                rating = None if recent else rng.choices(["HELPFUL", "NEUTRAL", "UNNECESSARY"], [0.7, 0.2, 0.1])[0]
                iv = {
                    "user_id": user_id, "intervention_type": "demo", "agent_message": msg, "tts_played": True,
                    "cooldown_expires_at": t + timedelta(minutes=30), "intervention_rating": rating,
                    "emotion_accuracy": None if rating is None else "CORRECT",
                    "feedback_at": None if rating is None else t + timedelta(minutes=2), "created_at": t,
                }
                out.append(
                    f"INSERT INTO interventions (emotion_log_id, {', '.join(iv)}) "
                    f"SELECT MAX(emotion_log_id), {', '.join(q(v) for v in iv.values())} "
                    f"FROM emotion_logs WHERE log_type = 'demo';")
                n_iv += 1
                for i, (title, author, url, reason) in enumerate(recs):
                    cat = cats[i]
                    fb = None if rating is None else ("LIKED" if rating == "HELPFUL" else "SKIPPED")
                    rc = {
                        "category": cat, "title": title, "artist_or_author": author, "external_id": None,
                        "external_url": url, "thumbnail_url": None, "reason": reason, "user_feedback": fb,
                        "played": fb == "LIKED", "relevance_rating": None if fb is None else "RELEVANT",
                        "created_at": t,
                    }
                    out.append(
                        f"INSERT INTO recommendations (intervention_id, {', '.join(rc)}) "
                        f"SELECT MAX(intervention_id), {', '.join(q(v) for v in rc.values())} "
                        f"FROM interventions WHERE intervention_type = 'demo';")
            t += timedelta(minutes=rng.randint(20, 40))
    out.append("COMMIT;")
    out.append(f"-- 감정 기록 {n_logs}건, 개입 {n_iv}건")
    return "\n".join(out) + "\n", n_logs, n_iv


CLEAR_SQL = """-- 시연용 샘플만 지운다 (실제 기록은 그대로)
START TRANSACTION;
DELETE FROM recommendations WHERE intervention_id IN (SELECT intervention_id FROM interventions WHERE intervention_type = 'demo');
DELETE FROM interventions WHERE intervention_type = 'demo';
DELETE FROM emotion_logs WHERE log_type = 'demo';
COMMIT;
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--user", type=int, default=int(os.getenv("USER_ID", "1").split("#")[0].strip() or 1))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--now", default=None, help=argparse.SUPPRESS)   # 시험용: 'YYYY-MM-DD HH:MM'
    a = ap.parse_args()
    os.makedirs("logs", exist_ok=True)
    if a.clear:
        path = a.out or "logs/demo_clear.sql"
        open(path, "w", encoding="utf-8").write(CLEAR_SQL)
        print(f"{path} 생성 → mysql -u capstone -p capstone < {path}")
        return
    now = datetime.strptime(a.now, "%Y-%m-%d %H:%M") if a.now else datetime.now()
    sql, n_logs, n_iv = generate(a.days, a.user, a.seed, now)
    path = a.out or "logs/demo_seed.sql"
    open(path, "w", encoding="utf-8").write(sql)
    print(f"{path} 생성: 감정 기록 {n_logs}건, 개입 {n_iv}건 (사용자 {a.user}, 최근 {a.days}일)")
    print(f"넣기: mysql -u capstone -p capstone < {path}")


if __name__ == "__main__":
    main()
