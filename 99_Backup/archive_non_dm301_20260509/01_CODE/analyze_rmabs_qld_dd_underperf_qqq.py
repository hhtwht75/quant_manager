#!/usr/bin/env python3
"""
전구간 1회: RMABS-QLD vs QQQ B&H
- RMABS-QLD 순자산 ATH-구간 MDD 에피소드(깊은 순 상위 N)
- QQQ 매수후보유 대비 RMABS-QLD 순수익이 상대적으로 가장 불리했던 고정 거래일 창

실행:
  python3 01_CODE/analyze_rmabs_qld_dd_underperf_qqq.py [--root 2002-10-01]
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
    strategy_buy_and_hold,
    strategy_rsi_ma_based_switching,
)

OUT_DIR = _ROOT / "03_RESULT" / "sensitivity"
CAP = 100_000.0


def ath_episodes(port: pd.Series) -> list[dict]:
    idx = port.index
    w = port.astype(float).values
    ath_i: list[int] = []
    cur_m = -np.inf
    for i, v in enumerate(w):
        if v > cur_m:
            cur_m = v
            ath_i.append(i)

    episodes: list[dict] = []
    for j in range(len(ath_i) - 1):
        p, pend = ath_i[j], ath_i[j + 1]
        peak_val = w[p]
        seg = w[p : pend + 1]
        if len(seg) < 2:
            continue
        tl = int(seg.argmin())
        t_i = p + tl
        trough_val = w[t_i]
        mdd = float(trough_val / peak_val - 1.0)
        episodes.append(
            {
                "peak_date": str(idx[p].date()),
                "trough_date": str(idx[t_i].date()),
                "recovery_to_new_ath_date": str(idx[pend].date()),
                "dd_days_peak_to_trough": int(t_i - p),
                "mdd_pct": mdd * 100.0,
            }
        )

    if ath_i:
        p = ath_i[-1]
        seg = w[p:]
        peak_val = w[p]
        tl = int(seg.argmin())
        t_i = p + tl
        trough_val = w[t_i]
        mdd = float(trough_val / peak_val - 1.0)
        last_dt = idx[-1]
        if mdd < -0.005 and trough_val < peak_val * 0.995:
            episodes.append(
                {
                    "peak_date": str(idx[p].date()),
                    "trough_date": str(idx[t_i].date()),
                    "recovery_to_new_ath_date": None,
                    "dd_days_peak_to_trough": int(t_i - p),
                    "mdd_pct": mdd * 100.0,
                    "as_of_note": str(last_dt.date()),
                }
            )

    return episodes


def worst_excess_bh_over_strat(nav_bh: pd.Series, nav_ld: pd.Series, win: int, top_k: int) -> list[dict]:
    idx = nav_bh.index
    n = len(idx)
    if n <= win:
        return []
    bh = nav_bh.astype(float).values
    ld = nav_ld.astype(float).values
    rows: list[tuple[float, int, float, float]] = []
    for s in range(0, n - win):
        e = s + win
        rb = bh[e] / bh[s] - 1.0
        rl = ld[e] / ld[s] - 1.0
        rows.append((rb - rl, s, rb, rl))
    rows.sort(reverse=True, key=lambda x: x[0])
    chosen: list[dict] = []
    used_ranges: list[tuple[int, int]] = []
    min_gap_days = win // 2
    for ex, s, rb, rl in rows:
        e = s + win
        if len(chosen) >= top_k:
            break
        if any(not (e < us - min_gap_days or s > ue + min_gap_days) for us, ue in used_ranges):
            continue
        used_ranges.append((s, e))
        chosen.append(
            {
                "start_date": str(idx[s].date()),
                "end_date": str(idx[e].date()),
                "window_trading_days": win,
                "qqq_buyhold_period_return_pct": float(rb * 100.0),
                "rmabs_qld_period_return_pct": float(rl * 100.0),
                "excess_qq_over_rmabs_qld_pp": float(ex * 100.0),
            }
        )
    return chosen


def cumulative_relative(nav_bh: pd.Series, nav_ld: pd.Series) -> pd.Series:
    r = nav_ld.astype(float) / nav_bh.astype(float).replace(0.0, np.nan)
    return r / r.iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="2002-10-01")
    ap.add_argument("--top-mdd", type=int, default=8)
    ap.add_argument("--top-underperf", type=int, default=8)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qqq = load_extended_daily("QQQ")
    qld_df = load_extended_daily("QLD")
    tq = load_extended_daily("TQQQ")
    c = qqq.index.intersection(qld_df.index).intersection(tq.index)
    c = c[c >= pd.Timestamp(args.root.strip())]
    Q, L, T = qqq.loc[c], qld_df.loc[c], tq.loc[c]

    nav_bh = strategy_buy_and_hold(Q, CAP, "QQQ BH")
    nav_rm, events = strategy_rsi_ma_based_switching(Q, L, T, CAP, series_name="RMABS-QLD")

    eps = sorted(ath_episodes(nav_rm), key=lambda x: x["mdd_pct"])[: args.top_mdd]

    uw252 = worst_excess_bh_over_strat(nav_bh, nav_rm, 252, args.top_underperf)
    uw126 = worst_excess_bh_over_strat(nav_bh, nav_rm, 126, args.top_underperf)

    rel = cumulative_relative(nav_bh, nav_rm)
    rel_eps = sorted(ath_episodes(rel.rename("ratio")), key=lambda x: x["mdd_pct"])[:5]

    outp = OUT_DIR / (
        f"rmabs_qld_dd_underperf_vs_qqq_{c[0].strftime('%Y%m%d')}_{c[-1].strftime('%Y%m%d')}.json"
    )

    dd_full = float((nav_rm / nav_rm.cummax() - 1.0).min() * 100.0)
    rel_dd_full = float(((rel / rel.cummax()) - 1.0).min() * 100.0)

    blob = {
        "benchmark": "QQQ 100% buy-and-hold",
        "strategy": "RMABS-QLD strategy_rsi_ma_based_switching",
        "period": [str(c[0].date()), str(c[-1].date())],
        "mdd_nav_full_sample_pct": dd_full,
        "mdd_normalized_ratio_vs_bh_full_pct": rel_dd_full,
        "notes": (
            "mdd_normalized_ratio_vs_bh: 시작일 기준 NAV_LD/NAV_QQ 를 1로 맞춘 뒤, "
            "그 비율 시계열의 ATH 대비 깊은 하락 = 장기 들고만 있던 QQQ 대비 전략 순자산이 많이 졌던 시기."
        ),
        "nav_rmabs_events_count": int(len(events)),
        "deepest_absolute_nav_drawdown_segments": eps,
        "worst_buyhold_minus_strategy_windows_tr252d_nonoverlap_greedy": uw252,
        "worst_buyhold_minus_strategy_windows_tr126d_nonoverlap_greedy": uw126,
        "deepest_underperformance_ratio_LD_over_Q_segments": [
            {"interpretation_ratio": "(NAV_LD/NAV_Q) / 시작비", **e}
            for e in rel_eps
        ],
    }

    with open(outp, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, ensure_ascii=False)

    print(f"저장 JSON: {outp}\n")
    print(f"전구간 순자산 MDD(RMABS-QLD 단돈) ≈ {dd_full:.2f}%")
    print(f"정규화 비 NAV_LD÷NAV_QQQ 의 ATH대비 깊은 하락 ≈ {rel_dd_full:.2f}% (QQQ 우세 범위)\n")

    print("▶ RMABS-QLD 절대 NAV — ATH 에피소드 바닥(깊은 순):")
    for i, e in enumerate(eps, 1):
        rec = e.get("recovery_to_new_ath_date")
        tail = f"회복→새 ATH {rec}" if rec else f"(미회종가 {e.get('as_of_note', '')})"
        print(
            f"  {i}. {e['peak_date']} ATH → 최저 {e['trough_date']}  "
            f"MDD **{e['mdd_pct']:.1f}%**  피크~저점 영업일 {e['dd_days_peak_to_trough']}  {tail}"
        )

    print("\n▶ 같은 구간 순수익에서 QQQ B&H가 RMABS-QLD를 가장 많이 이긴 252영업일 창:")
    for i, r in enumerate(uw252, 1):
        print(
            f"  {i}. `{r['start_date']}`~`{r['end_date']}`  "
            f"QQQ **{r['qqq_buyhold_period_return_pct']:+.1f}%** vs 전략 **{r['rmabs_qld_period_return_pct']:+.1f}%**  "
            f"→ 초과 **{r['excess_qq_over_rmabs_qld_pp']:+.1f}%p**"
        )

    print("\n▶ 동형 · 126영업일:")
    for i, r in enumerate(uw126, 1):
        print(
            f"  {i}. `{r['start_date']}`~`{r['end_date']}`  초과 **{r['excess_qq_over_rmabs_qld_pp']:+.1f}%p**"
        )

    print("\n▶ 비 NAV_LD÷NAV_QQQ (시작일=1) 시계열의 ATH 에피소드 하락 — QQQ 속도 우위가 컸던 장기 패치:")
    for i, e in enumerate(rel_eps, 1):
        rec = e.get("recovery_to_new_ath_date")
        tail = f"회복 {rec}" if rec else str(e.get("as_of_note", ""))
        print(
            f"  {i}. 비율고점 ~ 비율바닥  {e['peak_date']} → {e['trough_date']}  "
            f"상대저점 **{e['mdd_pct']:.1f}%**  {tail}"
        )


if __name__ == "__main__":
    main()
