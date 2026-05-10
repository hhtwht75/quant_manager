"""
S4-2 (strategy_tiered): L1/L2 **attack_stops** on vs off 동일 기간 비교.

• 손절 on  = 기본(L1 @ QLD ≤ -15%, L2 @ QLD < -20%)
• 손절 off = attack_stops=False (TRAILING의 TRAIL_FLOOR/EXIT 는 그대로)

실행: python3 01_CODE/s4_2_attack_stop_compare.py
      python3 01_CODE/s4_2_attack_stop_compare.py --start 2010-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import load_extended_daily  # noqa: E402
from backtest_tiered import strategy_tiered  # noqa: E402
from evaluation_metrics import full_metrics, oos_metric_bundle  # noqa: E402

OUT_DIR = _ROOT / "03_RESULT" / "sensitivity"
DEFAULT_START = pd.Timestamp("2002-10-01")
CAP = 100_000


def parse_args():
    p = argparse.ArgumentParser(description="S4-2 attack_stops on/off 비교")
    p.add_argument("--start", type=str, default=str(DEFAULT_START.date()))
    p.add_argument("--end", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end) if args.end.strip() else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qqq = load_extended_daily("QQQ")
    qld = load_extended_daily("QLD")
    tqqq = load_extended_daily("TQQQ")
    c = qqq.index.intersection(qld.index).intersection(tqqq.index)
    Q, L, T = qqq.loc[c], qld.loc[c], tqqq.loc[c]
    m = Q.index >= start_ts
    if end_ts is not None:
        m &= Q.index <= end_ts
    Q, L, T = Q.loc[m], L.loc[m], T.loc[m]

    rows = []
    for label, attack_stops in (
        ("S4-2 손절 ON (L1≤-15%, L2<-20%)", True),
        ("S4-2 손절 OFF (attack_stops=False)", False),
    ):
        ser, ev = strategy_tiered(
            Q, L, T, CAP,
            attack_stops=attack_stops,
            series_name=label,
        )
        fm = full_metrics(ser)
        omb = oos_metric_bundle(ser)
        n_l1 = len(ev[ev["type"] == "L1_STOP"]) if len(ev) else 0
        n_l2 = len(ev[ev["type"] == "L2_STOP"]) if len(ev) else 0
        rows.append({
            "label": label,
            "attack_stops": attack_stops,
            "total_return_pct": round(omb["total_return"] * 100, 4),
            "cagr_pct": round(omb["cagr"] * 100, 4),
            "mdd_pct": round(omb["mdd"] * 100, 4),
            "sharpe": round(omb["sharpe"], 4),
            "sortino": round(omb["sortino"], 4),
            "ulcer": round(omb["ulcer"], 4),
            "calmar": round(fm["calmar"], 4),
            "l1_stop_events": int(n_l1),
            "l2_stop_events": int(n_l2),
            "end_value": float(ser.iloc[-1]),
        })

    tag = f"{Q.index[0].strftime('%Y%m%d')}_{Q.index[-1].strftime('%Y%m%d')}"
    out_json = OUT_DIR / f"s4_2_attack_stops_compare_{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"period": tag, "rows": rows}, f, indent=2, ensure_ascii=False)

    print("=" * 92)
    print(f"  S4-2 손절 비교  |  ${CAP:,.0f} 시작  |  {Q.index[0].date()} ~ {Q.index[-1].date()}  ({len(Q):,}일)")
    print("=" * 92)
    print(
        f"  {'설정':<36} {'누적%':>10} {'CAGR%':>8} {'MDD%':>8} "
        f"{'Sharpe':>8} {'Sortino':>8} {'L1_STOP':>7} {'L2_STOP':>7}"
    )
    print("-" * 92)
    for r in rows:
        print(
            f"  {r['label']:<36} {r['total_return_pct']:>9.2f}% {r['cagr_pct']:>7.2f}% "
            f"{r['mdd_pct']:>7.2f}% {r['sharpe']:>8.3f} {r['sortino']:>8.3f} "
            f"{r['l1_stop_events']:>7} {r['l2_stop_events']:>7}"
        )
    print("=" * 92)
    print(f"  JSON: {out_json}")


if __name__ == "__main__":
    main()
