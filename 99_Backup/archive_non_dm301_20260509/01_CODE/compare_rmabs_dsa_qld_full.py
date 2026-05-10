"""
전 구간 1회: QLD 매수후보유 vs DSA(strategy_s5) vs RMABS(strategy_rsi_ma_based_switching).

실행:
  python3 01_CODE/compare_rmabs_dsa_qld_full.py [--root 2002-10-01]
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

from backtest_switching import (  # noqa: E402
    load_extended_daily,
    strategy_buy_and_hold,
    strategy_rsi_ma_based_switching,
    strategy_s5,
)
from evaluation_metrics import fmt_metrics_row, full_metrics  # noqa: E402

OUT_DIR = _ROOT / "03_RESULT" / "sensitivity"
CAP = 100_000.0
S5_DSA = dict(
    beta=0.5,
    min_drop=0.10,
    max_drop=0.20,
    trailing_stop_pct=-0.15,
    use_stop_loss=True,
    position_mode="exp",
    exp_frac_lo=0.25,
    exp_frac_hi=1.00,
    exp_base=2.0,
    stop_factor=0.75,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="2002-10-01")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qqq = load_extended_daily("QQQ")
    qld = load_extended_daily("QLD")
    tqqq = load_extended_daily("TQQQ")
    common = qqq.index.intersection(qld.index).intersection(tqqq.index)
    root_ts = pd.Timestamp(args.root.strip())
    common = common[common >= root_ts]
    Q, L, T = qqq.loc[common], qld.loc[common], tqqq.loc[common]

    bh = strategy_buy_and_hold(L, CAP, "QLD B&H")
    dsa, ev_dsa = strategy_s5(Q, L, T, CAP, series_name="DSA", **S5_DSA)
    rmabs, ev_rm = strategy_rsi_ma_based_switching(
        Q, L, T, CAP, series_name="RMABS"
    )

    m_bh = full_metrics(bh)
    m_d = full_metrics(dsa)
    m_r = full_metrics(rmabs)

    print("=" * 88)
    print("RSI_and_Moving_Average_Based_Switching (RMABS) 전 구간 vs QLD 보유 vs DSA")
    print(f"기간 {common[0].date()} ~ {common[-1].date()}  거래일 {len(common)}  초기자본 ${CAP:,.0f}")
    print("-" * 88)
    print("RMABS (QQQ 시그널): MA 최초일 규칙0(QLD vs QQQ) → RSI(30↑↓) 후 종가>MA×1.03→TQQQ;")
    print("       TQQQ 청산→QLD(진입가↓·트레일-15%); 이후 종가<MA×0.97까지 QQQ 플래그 시 QQQ 전환")
    print("=" * 88)
    print(fmt_metrics_row("QLD 100% B&H", m_bh))
    print(fmt_metrics_row("DSA (S5 anchor)", m_d))
    print(fmt_metrics_row("RMABS", m_r))
    print("-" * 88)
    print(f"DSA 이벤트 수: {len(ev_dsa)}  |  RMABS 거래 신호 수: {len(ev_rm)}")
    if len(ev_rm) > 0 and "type" in ev_rm.columns:
        print(ev_rm["type"].value_counts().to_string())

    stem = (
        f"compare_rmabs_dsa_qld_{common[0].strftime('%Y%m%d')}_{common[-1].strftime('%Y%m%d')}"
    )
    outp = OUT_DIR / f"{stem}.json"
    MK = ("cagr", "sharpe", "sortino", "mdd", "ulcer")
    blob = {
        "strategy_defined_as": "RSI_and_Moving_Average_Based_Switching (RMABS)",
        "period": [str(common[0].date()), str(common[-1].date())],
        "rows": [
            {"name": "QLD B&H", **{k: float(m_bh[k]) for k in MK}},
            {"name": "DSA", **{k: float(m_d[k]) for k in MK}},
            {"name": "RMABS", **{k: float(m_r[k]) for k in MK}},
        ],
        "rma_events_count": len(ev_rm),
        "dsa_events_count": len(ev_dsa),
    }
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, ensure_ascii=False)
    print("\n저장:", outp)


if __name__ == "__main__":
    main()
