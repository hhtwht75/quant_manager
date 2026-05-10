"""
S4-2 (strategy_tiered) 기본 bounce_l3=-10% vs 완화 bounce_l3=-11% 비교.

실행:
  python3 01_CODE/compare_s42_bounce_l3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import load_extended_daily  # noqa: E402
from backtest_tiered import strategy_tiered  # noqa: E402
from evaluation_metrics import full_metrics  # noqa: E402

ROOT = "2002-10-01"
CAP = 100_000


def _load_bundle():
    qqq = load_extended_daily("QQQ")
    qld = load_extended_daily("QLD")
    tqq = load_extended_daily("TQQQ")
    c = qqq.index.intersection(qld.index).intersection(tqq.index)
    Q, L, T = qqq.loc[c], qld.loc[c], tqq.loc[c]
    m = Q.index >= pd.Timestamp(ROOT)
    return Q.loc[m], L.loc[m], T.loc[m]


def main():
    Q, L, T = _load_bundle()
    p0, ev0 = strategy_tiered(Q, L, T, CAP, bounce_l3=-0.10, series_name="S42_b10")
    p1, ev1 = strategy_tiered(Q, L, T, CAP, bounce_l3=-0.11, series_name="S42_b11")

    m0 = full_metrics(p0)
    m1 = full_metrics(p1)

    def _fmt(d: dict) -> None:
        print(
            f"    CAGR: {d['cagr']:.2%}  Sharpe: {d['sharpe']:.2f}  Sortino: {d['sortino']:.2f}  "
            f"MDD: {d['mdd']:.2%}  Ulcer: {d['ulcer']:.2f}"
        )

    print("=== 전체 구간 (root", ROOT, ") ===")
    print("  bounce_l3 = -10% (기본)")
    _fmt(m0)
    print(f"    최종 자산: ${p0.iloc[-1]:,.0f}")
    print("  bounce_l3 = -11%")
    _fmt(m1)
    print(f"    최종 자산: ${p1.iloc[-1]:,.0f}")

    print(
        f"\n  최종 자산 비율 (b11 vs b10): {p1.iloc[-1] / p0.iloc[-1] - 1:+.2%}  "
        f"(${p1.iloc[-1]:,.0f} / ${p0.iloc[-1]:,.0f})"
    )

    # 위기 부분창
    d0, d1 = pd.Timestamp("2007-10-31"), pd.Timestamp("2008-10-27")
    print(f"\n=== 부분창 {d0.date()} ~ {d1.date()} (이전 극단 승패 구간) ===")
    s0 = p0.loc[d0:d1]
    s1 = p1.loc[d0:d1]
    r0, r1 = s0.iloc[-1] / s0.iloc[0] - 1, s1.iloc[-1] / s1.iloc[0] - 1
    print(f"  b10 구간 수익률: {r0:.2%}")
    print(f"  b11 구간 수익률: {r1:.2%}")
    print(f"  차이 (b11 - b10):   {(r1 - r0) * 100:+.2f} pt")

    # 2007-12-07 이벤트
    day = pd.Timestamp("2007-12-07")
    for label, ev in ("b10", ev0), ("b11", ev1):
        sub = ev[pd.to_datetime(ev["Date"]) == day] if not ev.empty else pd.DataFrame()
        print(f"\n=== {label} on {day.date()} ===")
        if sub.empty:
            print("  (이 날짜 이벤트 없음)")
        else:
            print(sub.to_string(index=False))

    # b11만 12/7 L3 신규 — 12월 전후 이벤트 몇 줄
    ev1d = ev1.copy()
    ev1d["Date"] = pd.to_datetime(ev1d["Date"])
    w = (ev1d["Date"] >= "2007-11-01") & (ev1d["Date"] <= "2008-03-01")
    print("\n=== bounce_l3=-11% 이벤트 2007-11~2008-02 ===")
    print(ev1d.loc[w].to_string(index=False))


if __name__ == "__main__":
    main()
