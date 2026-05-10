#!/usr/bin/env python3
"""
RMABS-QLD(방어 QQ) vs RMABS-GOLD: **QLD쪽 구간수익이 더 큰** 252거래일 창(비중첩)을 고르고
각 창 거래일마다 daily_audit 행(QLD쪽 / GOLD쪽)을 출력·저장.

전제: rmabs_gold_simulation 과 동일( GC=F 우선·warmup_hold_cash=True ).

실행:
  python3 01_CODE/rmabs_gold_qq_superior_daily_audit.py [--top N] [--root 2002-10-01]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from rmabs_gold_simulation import (  # noqa: E402
    CAP,
    MIN_GAP_TRADING,
    WIN_TRADING,
    load_gold_series,
)
from backtest_switching import (  # noqa: E402
    load_extended_daily,
    strategy_rsi_ma_based_switching,
    strategy_rsi_ma_based_switching_gold,
)

OUT_TXT = _ROOT / "03_RESULT" / "sensitivity" / "rmabs_gold_qq_superior_daily_audit.txt"


def pick_qq_superior_episodes(
    nav_qld_def: pd.Series,
    nav_gold: pd.Series,
    *,
    win: int,
    min_gap_days: int,
    k_episodes: int,
) -> list[dict]:
    """spread = (구간 GOLD% − 구간 QQQ방%) 가 음수인 창 중, QQQ 초과폭 최대 순."""
    idx = nav_qld_def.index
    n = len(idx)
    cands: list[tuple[float, int, float, float, float]] = []
    for s in range(0, max(0, n - win)):
        e = s + win
        ret_q = float(nav_qld_def.iloc[e] / nav_qld_def.iloc[s] - 1.0)
        ret_g = float(nav_gold.iloc[e] / nav_gold.iloc[s] - 1.0)
        spread_pp = (ret_g - ret_q) * 100.0
        if spread_pp >= 0.0:
            continue
        edge_pp = (ret_q - ret_g) * 100.0
        cands.append((edge_pp, s, ret_q * 100.0, ret_g * 100.0, spread_pp))

    cands.sort(reverse=True, key=lambda x: x[0])

    spans: list[tuple[int, int, float, float, float, float]] = []
    for edge_pp, s, rq, rg, sp in cands:
        e = s + win
        if any(
            not (e < es - min_gap_days or s > ee + min_gap_days)
            for es, ee, *_ in spans
        ):
            continue
        spans.append((s, e, edge_pp, rq, rg, sp))
        if len(spans) >= k_episodes:
            break

    spans.sort(key=lambda x: idx[x[0]])
    episodes: list[dict] = []
    for si, ei, edge_pp, rq, rg, sp in spans:
        t0, t1 = idx[si], idx[ei]
        episodes.append(
            {
                "start": str(t0.date()),
                "end": str(t1.date()),
                "days": win,
                "period_return_RM_abs_QLD_defqqq_pct": rq,
                "period_return_RM_abs_GOLD_pct": rg,
                "spread_gold_minus_qld_pp": sp,
                "qq_advantage_pp": edge_pp,
            }
        )
    return episodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="2002-10-01")
    ap.add_argument("--top", type=int, default=4, help="비중첩 에피소드 개수")
    args = ap.parse_args()

    q_raw = load_extended_daily("QQQ")
    ql_raw = load_extended_daily("QLD")
    t_raw = load_extended_daily("TQQQ")
    ix0 = q_raw.index.intersection(ql_raw.index).intersection(t_raw.index).sort_values()
    root_ts = pd.Timestamp(args.root.strip())
    ix0 = ix0[ix0 >= root_ts]
    Q0, L0, T0 = q_raw.loc[ix0], ql_raw.loc[ix0], t_raw.loc[ix0]

    gold_df, gold_note = load_gold_series(ix0[0], ix0[-1])
    common = ix0.intersection(gold_df.index).sort_values()
    if len(common) < 400:
        raise SystemExit("공통 교집합 거래일이 너무 짧음.")
    Q, L, T, G = Q0.loc[common], L0.loc[common], T0.loc[common], gold_df.loc[common]

    aud_q: list[dict] = []
    aud_g: list[dict] = []
    nav_q, _ = strategy_rsi_ma_based_switching(
        Q,
        L,
        T,
        CAP,
        series_name="RMABS-QLD(ref)",
        warmup_hold_cash=True,
        daily_audit=aud_q,
    )
    nav_g, _ = strategy_rsi_ma_based_switching_gold(Q, G, L, T, CAP, warmup_hold_cash=True, daily_audit=aud_g)

    df_q = pd.DataFrame(aud_q)
    df_g = pd.DataFrame(aud_g)
    df_q["Date"] = pd.to_datetime(df_q["Date"]).dt.normalize()
    df_g["Date"] = pd.to_datetime(df_g["Date"]).dt.normalize()
    df_q = df_q.set_index("Date")
    df_g = df_g.set_index("Date")

    eps = pick_qq_superior_episodes(
        nav_q,
        nav_g,
        win=WIN_TRADING,
        min_gap_days=MIN_GAP_TRADING,
        k_episodes=max(1, args.top),
    )

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("RMABS-QLD(방어=QQQ) 우월 에피소드 — 거래일별 audit")
    lines.append(f"warmup_hold_cash=True | 금: {gold_note}")
    lines.append(f"공통기간 {common[0].date()} ~ {common[-1].date()} | 에피소드 수 {len(eps)}")
    lines.append("스프레드 = 구간(GOLD%) − 구간(QLD방%)  (음수 → QQQ방 우세)")
    lines.append("")

    for i, ep in enumerate(eps, start=1):
        t0 = pd.Timestamp(ep["start"])
        t1 = pd.Timestamp(ep["end"])
        m = (df_q.index >= t0) & (df_q.index <= t1)
        sub_ix = df_q.index[m]

        hdr = (
            f"=== 에피소드 {i}: {ep['start']} → {ep['end']} (~{WIN_TRADING} 거래일) ===\n"
            f"  구간수익%  QQQ방={ep['period_return_RM_abs_QLD_defqqq_pct']:+.2f}  "
            f"GOLD={ep['period_return_RM_abs_GOLD_pct']:+.2f}\n"
            f"  스프레드(G−Q)= {ep['spread_gold_minus_qld_pp']:+.2f} pp "
            f" (QQ 우위 +{ep['qq_advantage_pp']:.2f} pp)"
        )
        lines.append(hdr)
        lines.append(
            f"{'date':12s}{'QQ_hold':10s}{'QQ_trade':<38s}"
            f"{'QQ_NAV':>14s}{'':3s}"
            f"{'G_hold':10s}{'G_trade':<38s}"
            f"{'G_NAV':>14s}"
        )
        lines.append("-" * 160)

        for dt in sub_ix:
            rq = df_q.loc[dt]
            rg = df_g.loc[dt]
            tq = (rq["trade"] or "").replace("\n", " ")[:36]
            tg = (rg["trade"] or "").replace("\n", " ")[:36]
            lines.append(
                f"{str(dt.date()):12s}"
                f"{str(rq['hold']):10s}{(tq+'…' if len(str(rq['trade'] or ''))>36 else tq):<38s}"
                f"{rq['nav']:>14,.2f}   "
                f"{str(rg['hold']):10s}{(tg+'…' if len(str(rg['trade'] or ''))>36 else tg):<38s}"
                f"{rg['nav']:>14,.2f}"
            )
        lines.append("")

    text = "\n".join(lines)
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n저장: {OUT_TXT}")


if __name__ == "__main__":
    main()
