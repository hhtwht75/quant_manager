"""
전구간 1회: 벤치 4종(B1~B4) + DSA + RMABS-QQQ + RMABS-QLD(기본 파라).

벤치 정의는 random_window_dsa_rmabs_bench 의 BENCH_META 와 동일.
  B1 QQ 100% | B2 QQ/TQQ 50:50 드리프트 | B3 QLD 100% | B4 TQQ 100%

RMABS-QLD = ``strategy_rsi_ma_based_switching`` (기본 ma_breakdown_multiplier=0.97 등).

실행: python3 01_CODE/full_period_bench_dsa_rmabs_once.py [--root 2002-10-01]
결과: 03_RESULT/full_period_bench_dsa_rmabs_<start>_<end>.json + 콘솔 표
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

from benchmark_s2_s5_anchor_table import S5_ANCHOR_COMMON, half_half_drift  # noqa: E402
from backtest_switching import (  # noqa: E402
    load_extended_daily,
    strategy_buy_and_hold,
    strategy_rsi_ma_based_switching,
    strategy_rsi_ma_based_switching_qqq_only,
    strategy_s5,
)
from evaluation_metrics import full_metrics  # noqa: E402
from strategy_result_tables import df_to_wide_console_block, full_period_metrics_table  # noqa: E402

SERIES_META = (
    ("B1", "QQQ 100%"),
    ("B2", "QQQ/TQQQ 50:50 드리프트"),
    ("B3", "QLD 100%"),
    ("B4", "TQQQ 100%"),
    ("DSA", "DSA (S5 앵커)"),
    ("RMQQ", "RMABS-QQQ"),
    ("RQLD", "RMABS-QLD"),
)

OUT_DIR = _ROOT / "03_RESULT"
CAP = 100_000.0
DSA_KW = {**S5_ANCHOR_COMMON, "stop_factor": 0.75}
MK = ("cagr", "sharpe", "sortino", "mdd", "ulcer", "calmar", "pain_ratio")


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

    codes = tuple(c for c, _ in SERIES_META)

    b1 = strategy_buy_and_hold(Q, CAP, "B1 QQ")
    b2 = half_half_drift(Q, T, CAP).rename("B2 50:50")
    b3 = strategy_buy_and_hold(L, CAP, "B3 QLD")
    b4 = strategy_buy_and_hold(T, CAP, "B4 TQQ")
    dsa, ev_dsa = strategy_s5(Q, L, T, CAP, series_name="DSA", **DSA_KW)
    rmqq, ev_qq = strategy_rsi_ma_based_switching_qqq_only(Q, L, T, CAP, series_name="RMABS-QQQ")
    rm_qld, ev_qld = strategy_rsi_ma_based_switching(Q, L, T, CAP, series_name="RMABS-QLD")

    nav = {"B1": b1, "B2": b2, "B3": b3, "B4": b4, "DSA": dsa, "RMQQ": rmqq, "RQLD": rm_qld}
    evt: dict[str, int | None] = {
        "B1": None,
        "B2": None,
        "B3": None,
        "B4": None,
        "DSA": len(ev_dsa),
        "RMQQ": len(ev_qq),
        "RQLD": len(ev_qld),
    }

    print("=" * 92)
    print("전구간 1회 백테스트 (공통 교집합 일봉)")
    print(f"  기간: {common[0].date()} ~ {common[-1].date()}  거래일 {len(common)}  시작 ${CAP:,.0f}")
    print("=" * 92)

    df_full = full_period_metrics_table(
        codes=codes,
        series_meta=SERIES_META,
        nav=nav,
        event_counts=evt,
    )
    print()
    print(df_to_wide_console_block("전구간 — 전략별 절대 지표 한 표", df_full))

    rows_out = []
    for code in codes:
        m = full_metrics(nav[code])
        nm = dict(SERIES_META)[code]
        rows_out.append(
            {
                "code": code,
                "label": nm,
                **{k: float(m[k]) for k in MK},
                "switch_or_event_count": evt[code],
            }
        )

    blob = {
        "period": [str(common[0].date()), str(common[-1].date())],
        "n_days": int(len(common)),
        "initial_capital": CAP,
        "dsa_params": {"note": "S5_ANCHOR_COMMON + stop_factor 0.75", **DSA_KW},
        "rma_bs_qld_defaults": {"strategy": "strategy_rsi_ma_based_switching (defaults incl. md=0.97)"},
        "table_wide": df_full.to_dict(orient="records"),
        "rows": rows_out,
    }
    stem = f"full_period_bench_dsa_rmabs_{common[0].strftime('%Y%m%d')}_{common[-1].strftime('%Y%m%d')}"
    outp = OUT_DIR / f"{stem}.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, ensure_ascii=False)
    print("\n저장:", outp)


if __name__ == "__main__":
    main()
