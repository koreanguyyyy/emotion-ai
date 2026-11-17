# detection/rppg_diag.py
"""
rPPG 진단 기록 — 심박이 안 잡힐 때 원인을 눈으로 확인하기 위한 저장 기능.

DIAG_EVERY_SEC(기본 30초)마다 logs/rppg_diag_<시각>/ 아래에 저장한다.
  - snapshot_<경과초>.npz : 일정 간격으로 맞춘 색 신호(R,G,B), CHROM·POS 맥파, 추정값
  - snapshot_<경과초>.png : ① 색 신호 ② 맥파 ③ 주파수 그래프 (matplotlib가 있을 때)

그래프 보는 법
  ③ 주파수 그래프에서 42~180bpm 구간에 뾰족한 봉우리가 하나 있고, 그 두 배 위치에
  작은 봉우리가 하나 더 있으면 맥박이 잡힌 것이다. 여러 봉우리가 비슷한 높이로 흩어져
  있으면 잡음뿐이다. 워치 심박과 봉우리 위치를 비교하면 된다.
그림은 별도 스레드에서 그려서 카메라 루프를 멈추지 않는다. 원본 영상은 저장하지 않는다.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime

import numpy as np


class RPPGDiagnostics:
    def __init__(self, log_dir: str = "logs", every_sec: float = 30.0, enabled: bool = True):
        self.enabled = enabled and every_sec > 0
        self.every_sec = every_sec
        self._next = every_sec
        self.dir = os.path.join(log_dir, f"rppg_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self._warned = False
        if self.enabled:
            os.makedirs(self.dir, exist_ok=True)

    def maybe_save(self, now: float, est, threshold: float) -> str | None:
        """now: 실행 경과 초. 저장했으면 파일 경로(확장자 제외)를 돌려준다."""
        if not self.enabled or est is None or est.t_uniform is None or now < self._next:
            return None
        self._next = now + self.every_sec
        base = os.path.join(self.dir, f"snapshot_{int(now):04d}s")
        np.savez_compressed(
            base + ".npz",
            t=est.t_uniform, rgb=est.rgb_uniform, chrom=est.chrom_signal, pos=est.pos_signal,
            chrom_bpm=est.chrom_bpm, chrom_quality=est.chrom_quality,
            pos_bpm=est.pos_bpm, pos_quality=est.pos_quality, fps_actual=est.fps_actual,
            threshold=threshold,
        )
        threading.Thread(target=self._plot, args=(base, est, threshold, now), daemon=True).start()
        return base

    def _plot(self, base: str, est, threshold: float, now: float) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            if not self._warned:
                print("[rPPG 진단] matplotlib가 없어 그래프는 건너뛰고 .npz만 저장합니다 (pip install matplotlib)")
                self._warned = True
            return
        try:
            t = est.t_uniform - est.t_uniform[0]
            fs = 1.0 / np.median(np.diff(est.t_uniform))
            _korean_font(plt)   # 글꼴은 그리기 전에 정해야 제목·범례에 적용된다
            fig, ax = plt.subplots(3, 1, figsize=(9, 8))

            rgb = est.rgb_uniform / est.rgb_uniform.mean(axis=0) - 1
            for c, col in enumerate(("tab:red", "tab:green", "tab:blue")):
                ax[0].plot(t, rgb[:, c] * 100, color=col, lw=0.8, label="RGB"[c])
            ax[0].set_title("① 이마+볼 평균 색 (평균 대비 %)")
            ax[0].set_ylabel("%"); ax[0].legend(loc="upper right", fontsize=8)

            for sig, name, col in ((est.chrom_signal, "CHROM", "tab:purple"), (est.pos_signal, "POS", "tab:orange")):
                s = sig / (np.std(sig) + 1e-12)
                ax[1].plot(t, s, color=col, lw=0.8, label=name)
            ax[1].set_title("② 맥파 (표준화)"); ax[1].legend(loc="upper right", fontsize=8)
            ax[1].set_xlabel("초")

            for sig, name, col, bpm, q in ((est.chrom_signal, "CHROM", "tab:purple", est.chrom_bpm, est.chrom_quality),
                                           (est.pos_signal, "POS", "tab:orange", est.pos_bpm, est.pos_quality)):
                f = np.fft.rfftfreq(len(sig), d=1 / fs) * 60
                p = np.abs(np.fft.rfft(sig)) ** 2
                m = (f >= 30) & (f <= 240)
                ax[2].plot(f[m], p[m] / p[m].max(), color=col, lw=1,
                           label=f"{name}: {bpm:.0f} bpm, 품질 {q:.2f}")
                if bpm == bpm:
                    ax[2].axvline(bpm, color=col, ls="--", lw=0.8)
            ax[2].axvspan(42, 180, color="0.92", zorder=0)
            ax[2].set_title(f"③ 주파수 (회색=검색 범위 42~180bpm, 품질 기준 {threshold:.2f})")
            ax[2].set_xlabel("bpm"); ax[2].legend(loc="upper right", fontsize=8)

            ok = est.quality >= threshold
            fig.suptitle(f"{int(now)}초 | 실제 {est.fps_actual:.1f}fps | 채택 {est.method.upper()} "
                         f"{est.bpm:.0f} bpm, 품질 {est.quality:.2f} → {'사용' if ok else '버림'}",
                         color="green" if ok else "red")
            fig.tight_layout()
            fig.savefig(base + ".png", dpi=90)
            plt.close(fig)
        except Exception as e:   # 진단 그림 실패가 본 기능을 멈추면 안 된다
            print(f"[rPPG 진단] 그래프 저장 실패: {e}")


def _korean_font(plt) -> None:
    """맥의 기본 한글 글꼴(Apple SD Gothic Neo)이 있으면 쓴다. 없으면 한글이 네모로 보일 수 있다."""
    from matplotlib import font_manager
    for name in ("Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR", "NanumGothic", "Malgun Gothic"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
