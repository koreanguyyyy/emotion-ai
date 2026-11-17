# tools/compare_face.py
"""
세션 기록에서 표정 계산 두 방식(규칙 기반 vs 학습 모델)을 비교한다. 논문의 비교 표·그래프용.
사용: python tools/compare_face.py                (가장 최근 logs/session_*.csv)
      python tools/compare_face.py logs/session_20261030_160000.csv

전제: FACE_MODEL=compare 또는 use로 측정한 기록 (model_label 열이 채워져 있어야 함).
무표정 기준선 보정이 끝난 뒤(face_baseline_ready=True)의 행만 비교한다.
정답 라벨이 없으므로 '어느 쪽이 맞다'가 아니라 '얼마나 일치하나, 어디서 갈리나'를 본다.
정답과 비교하려면 측정하며 지은 표정을 시간과 함께 메모해 두고 --script로 넘긴다:
      python tools/compare_face.py logs/session_x.csv --script "70-80:행복,100-110:슬픔,130-140:분노"
"""
import csv
import glob
import os
import sys
from collections import Counter

LABELS = ["무감정", "행복", "슬픔", "분노", "공포", "혐오", "놀람"]


def parse_script(text):
    spans = []
    for part in text.split(","):
        rng, lab = part.split(":")
        a, b = rng.split("-")
        spans.append((float(a), float(b), lab.strip()))
    return spans


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    script = None
    if "--script" in sys.argv:
        script = parse_script(sys.argv[sys.argv.index("--script") + 1])
        args = [a for a in args if a != sys.argv[sys.argv.index("--script") + 1]]
    path = args[0] if args else max(glob.glob("logs/session_*.csv"), key=os.path.getmtime)
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    rows = [r for r in rows if r.get("face_baseline_ready") == "True" and r.get("model_label")]
    print(f"기록: {path}\n비교한 행(기준선 보정 후, 모델 결과 있음): {len(rows)}초")
    if not rows:
        print("비교할 행이 없습니다. FACE_MODEL=compare로 측정했는지 확인하세요.")
        return

    agree = sum(r["rule_label"] == r["model_label"] for r in rows)
    print(f"\n라벨 일치율: {agree}/{len(rows)} = {100 * agree / len(rows):.0f}%")
    print("\n라벨별 개수 (초)\n  라벨      규칙  모델")
    rc, mc = Counter(r["rule_label"] for r in rows), Counter(r["model_label"] for r in rows)
    for lab in LABELS:
        if rc[lab] or mc[lab]:
            print(f"  {lab:6} {rc[lab]:6} {mc[lab]:5}")

    print("\n갈린 경우 상위 (규칙 → 모델)")
    for (a, b), n in Counter((r["rule_label"], r["model_label"]) for r in rows
                             if r["rule_label"] != r["model_label"]).most_common(6):
        print(f"  {a} → {b}: {n}초")

    rv = [float(r["rule_valence"]) for r in rows]
    mv = [float(r["model_valence"]) for r in rows]
    if len(rows) >= 3:
        mr, mm = sum(rv) / len(rv), sum(mv) / len(mv)
        cov = sum((a - mr) * (b - mm) for a, b in zip(rv, mv))
        sr = sum((a - mr) ** 2 for a in rv) ** 0.5
        sm = sum((b - mm) ** 2 for b in mv) ** 0.5
        if sr > 0 and sm > 0:
            print(f"\n정서가 상관계수(규칙 vs 모델): {cov / (sr * sm):+.2f}")

    if script:
        print("\n표정 메모와 비교 (메모한 구간만)")
        for name, key in (("규칙", "rule_label"), ("모델", "model_label")):
            hit = tot = 0
            for a, b, lab in script:
                seg = [r for r in rows if a <= float(r["t_sec"]) <= b]
                tot += len(seg)
                hit += sum(r[key] == lab for r in seg)
            print(f"  {name}: {hit}/{tot}초 일치 ({100 * hit / tot if tot else 0:.0f}%)")


if __name__ == "__main__":
    main()
