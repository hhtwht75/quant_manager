#!/usr/bin/env python3
"""
전 구간: RMABS-QQQ vs RMABS-QLD(strategy_rsi_ma_based_switching).

1) 두 순자산 ground truth 실행 + 전구간 성과 출력
2) 동일 무작위가 아니라 순자산 **로그비** 경로에서 252 거래일(약 1년) 창을 슬라이드해
   |Δln(NAV_Q / NAV_P)| 최대 비중첩 에피소드 최대 3개 추출
3) 각 에피소드 구간별 수익률 차·교차되는 이벤트 요약

실행:
  python3 01_CODE/rmabs_qqq_vs_qld_episodes.py [--root 2002-10-01]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import (  # noqa: E402
    load_extended_daily,
    strategy_rsi_ma_based_switching,
    strategy_rsi_ma_based_switching_qqq_only,
)
from evaluation_metrics import full_metrics, fmt_metrics_row  # noqa: E402

OUT_DIR = _ROOT / "03_RESULT" / "sensitivity"
CAP = 100_000.0
WIN = 252
MIN_GAP_DAYS = 126


def _events_between(ev: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    if ev.empty or not len(ev.columns):
        return ev
    d = pd.to_datetime(ev["Date"])
    m = (d >= t0) & (d <= t1)
    return ev.loc[m]


def pick_episodes(
    nav_p: pd.Series,
    nav_q: pd.Series,
    *,
    win: int,
    min_gap_days: int,
    k_episodes: int,
) -> list[dict]:
    """Pick up to k non-overlapping windows by magnitude of log-ratio change.

    Prefer QLD outperforming QQ (positive Δlog-ratio) vs underperform for variety.
    """
    idx = nav_p.index
    rp = nav_p.astype(float).values
    rq = nav_q.astype(float).values
    lr = np.log(rq / np.maximum(rp, 1e-12))

    candidates: list[tuple[float, int, float]] = []
    n = len(lr)
    for s in range(0, n - win):
        e = s + win
        dlt = float(lr[e] - lr[s])
        candidates.append((abs(dlt), s, dlt))

    candidates.sort(reverse=True, key=lambda x: x[0])

    spans: list[tuple[int, int, float]] = []
    for _abs_mag, s, dlt in candidates:
        e = s + win
        if any(not (e < es - min_gap_days or s > ee + min_gap_days) for es, ee, _ in spans):
            continue
        spans.append((s, e, dlt))
        if len(spans) >= k_episodes:
            break

    spans.sort(key=lambda x: idx[x[0]])
    episodes: list[dict] = []
    for si, ei, dlt_log in spans:
        t0, t1 = idx[si], idx[ei]
        r_p = float(nav_p.iloc[ei] / nav_p.iloc[si] - 1.0)
        r_q = float(nav_q.iloc[ei] / nav_q.iloc[si] - 1.0)
        episodes.append(
            {
                "start": str(t0.date()),
                "end": str(t1.date()),
                "days": win,
                "delta_log_nav_ratio_qld_minus_qq": dlt_log,
                "period_return_RMQQ_pct": r_p * 100.0,
                "period_return_RMABS_QLD_pct": r_q * 100.0,
                "spread_return_pct": (r_q - r_p) * 100.0,
            }
        )
    return episodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="2002-10-01")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qqq = load_extended_daily("QQQ")
    qld = load_extended_daily("QLD")
    tqq = load_extended_daily("TQQQ")
    c = qqq.index.intersection(qld.index).intersection(tqq.index)
    c = c[c >= pd.Timestamp(args.root.strip())]
    Q, L, T = qqq.loc[c], qld.loc[c], tqq.loc[c]

    nav_qq, ev_qq = strategy_rsi_ma_based_switching_qqq_only(
        Q, L, T, CAP, series_name="RMABS-QQQ"
    )
    nav_qld, ev_qld = strategy_rsi_ma_based_switching(
        Q, L, T, CAP, series_name="RMABS-QLD"
    )

    mq = full_metrics(nav_qq)
    ml = full_metrics(nav_qld)
    MK = ("cagr", "sharpe", "sortino", "mdd", "ulcer")

    print("=" * 90)
    print("RMABS-QQQ vs RMABS-QLD — 전구간 순자산 (시그널: QQQ)")
    print(f"기간 {Q.index[0].date()} ~ {Q.index[-1].date()}  일수 {len(Q)}  초기 {CAP:,.0f}")
    print("-" * 90)
    print(fmt_metrics_row("RMABS-QQQ (방어 QQ)", mq))
    print(fmt_metrics_row("RMABS-QLD (규칙0·청산 후 QQQ 전환 포함)", ml))
    print(
        f"이벤트 수  RMQQ: {len(ev_qq)}  |  RMQLD: {len(ev_qld)} "
        " (포트폴리오 체결 신호 행 개수)"
    )
    print("=" * 90)

    episodes = pick_episodes(
        nav_qq,
        nav_qld,
        win=WIN,
        min_gap_days=MIN_GAP_DAYS,
        k_episodes=3,
    )

    print("\n--- Δ가 큰 에피소드 3건 (약 252 거래일, 비중첩 최소 126거래일 간격) ---")
    print("spread_return_pct = RMABS-QLD 구간수익(%) − RMABS-QQQ 구간수익(%)")
    blob_eps: list[dict] = []

    idx = Q.index
    for rank, ep in enumerate(episodes, start=1):
        t0 = pd.Timestamp(ep["start"])
        t1 = pd.Timestamp(ep["end"])
        print(f"\n[에피소드 {rank}] {ep['start']} ~ {ep['end']} (~{WIN} 거래일)")
        print(
            f"  RMABS-QQQ 구간총수익 {ep['period_return_RMQQ_pct']:+.2f}%  "
            f" RMABS-QLD 구간총수익 {ep['period_return_RMABS_QLD_pct']:+.2f}%  "
            f" 차이(QLD−QQQ) {ep['spread_return_pct']:+.2f}% pp"
        )
        print(f"  구간 로그비 변화(QLD 상대 QQ) Δln≈ {ep['delta_log_nav_ratio_qld_minus_qq']:+.4f}")

        eqq = _events_between(ev_qq, t0, t1)
        eld = _events_between(ev_qld, t0, t1)
        print(f"  이 구간 교차 신호 RMQQ 행수 {len(eqq)} / RMQLD 행수 {len(eld)}")
        if len(eqq) > 0:
            print("  RMQQ (요약 타입 카운트):")
            print(eqq.groupby("type").size().sort_values(ascending=False).head(12).to_string())
        if len(eld) > 0:
            print("  RMQLD (요약 타입 카운트):")
            print(
                eld.groupby("type")
                .size()
                .sort_values(ascending=False)
                .head(12)
                .to_string()
            )
        blob_eps.append(ep | {"rmqq_counts": {}, "rmqld_counts": {}})
        if len(eqq) > 0:
            blob_eps[-1]["rmqq_counts"] = eqq.groupby("type").size().astype(int).to_dict()
        if len(eld) > 0:
            blob_eps[-1]["rmqld_counts"] = eld.groupby("type").size().astype(int).to_dict()

    out_json = OUT_DIR / (
        f"rmabs_qq_vs_qld_episodes_{Q.index[0].strftime('%Y%m%d')}_{Q.index[-1].strftime('%Y%m%d')}.json"
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "root": args.root.strip(),
                "full_metrics": {
                    "RMABS-QQQ": {k: float(mq[k]) for k in MK},
                    "RMABS-QLD": {k: float(ml[k]) for k in MK},
                },
                "episode_window_trading_days": WIN,
                "min_gap_between_episodes": MIN_GAP_DAYS,
                "episodes": blob_eps,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nJSON: {out_json}")


if __name__ == "__main__":
    main()
