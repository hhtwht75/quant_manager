"""
벤치마크(B1~B4)와 S2~S5 앵커 및 **S4-2**(3티어 전략)를 동일 기간·동일 초기자본으로 비교.

• 기간: --start / --end (기본 2002-10-01 ~ 데이터 끝)
• B4: QLD 100% 매수후보유
• **S4-2**: QLD -10%→-5% 반등 시 TQQQ 25%, -15%→-7.5% → 50%, -20%→-10% → 100% (backtest_tiered.strategy_tiered)
• **S5-2**: S4-2 티어·손절·트레일 + NORMAL 진입만 S5 게이트 (β=0.5, swing 깊이>11%)
• DSA (Dynamic Switching Algorithm, strategy_s5 앵커): β=0.5, min_drop=10%, max_drop=20%, trail=-15%,
           **지수 사이징** (min→25%, max→100%), **손절 on**: ATTACK 중 스윙 `drop`이 진입 시보다 깊어지면 청산.

출력: 콘솔 표 + CSV/JSON (03_RESULT/sensitivity/, 파일명에 기간 태그)

실행 예:
  python3 01_CODE/benchmark_s2_s5_anchor_table.py
  python3 01_CODE/benchmark_s2_s5_anchor_table.py --start 2019-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import (  # noqa: E402
    load_extended_daily,
    strategy_buy_and_hold,
    strategy_switching,
    strategy_switching_rsi,
    strategy_s4_trailing,
    strategy_s5,
    strategy_qld_touch_bounce_full,
    strategy_s5_2,
)
from backtest_tiered import strategy_tiered  # noqa: E402
from evaluation_metrics import oos_metric_bundle  # noqa: E402

OUT_DIR = _ROOT / "03_RESULT" / "sensitivity"
DEFAULT_START = pd.Timestamp("2002-10-01")
CAP = 100_000
S5_MAX_DROP = 0.20
S5_ANCHOR_COMMON = dict(
    beta=0.5,
    min_drop=0.10,
    max_drop=S5_MAX_DROP,
    trailing_stop_pct=-0.15,
    use_stop_loss=True,
    position_mode="exp",
    exp_frac_lo=0.25,
    exp_frac_hi=1.00,
    exp_base=2.0,
)

# S4 앵커 = stage4 S4 Anchor (b×2 shallow, b×4 deep in drawdown space)
S4_ANCHOR = dict(
    shallow_drop=-0.10,
    deep_drop=-0.20,
    shallow_bounce=-0.05,
    deep_bounce=-0.10,
    half_stop=-0.15,
    trailing_stop_pct=-0.15,
    half_frac=0.50,
    full_frac=1.00,
)


def half_half_drift(a: pd.DataFrame, b: pd.DataFrame, cap: float) -> pd.Series:
    sha = (cap * 0.5) / a["Close"].iloc[0]
    shb = (cap * 0.5) / b["Close"].iloc[0]
    return pd.Series(
        sha * a["Close"].values + shb * b["Close"].values,
        index=a.index,
    )


def parse_args():
    p = argparse.ArgumentParser(description="벤치마크(B1–B4) & S2–S5 앵커 & DSA & S4-2 비교 테이블")
    p.add_argument("--start", type=str, default=str(DEFAULT_START.date()),
                   help="시작일 YYYY-MM-DD (포함)")
    p.add_argument("--end", type=str, default="",
                   help="종료일 YYYY-MM-DD (포함). 비우면 데이터 마지막까지")
    return p.parse_args()


def main():
    args = parse_args()
    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end) if args.end.strip() else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qqq = load_extended_daily("QQQ")
    qld = load_extended_daily("QLD")
    tqqq = load_extended_daily("TQQQ")
    common = qqq.index.intersection(qld.index).intersection(tqqq.index)
    qqq, qld, tqqq = qqq.loc[common], qld.loc[common], tqqq.loc[common]
    if end_ts is not None:
        m = (qqq.index >= start_ts) & (qqq.index <= end_ts)
    else:
        m = qqq.index >= start_ts
    Q, L, T = qqq.loc[m], qld.loc[m], tqqq.loc[m]
    if len(Q) < 5:
        raise SystemExit(f"데이터 구간이 너무 짧습니다: {len(Q)}일 ({start_ts} ~ {end_ts})")

    runners: list[tuple[str, pd.Series]] = [
        ("B1 QQQ 100%", strategy_buy_and_hold(Q, CAP, "B1")),
        ("B2 QQQ/TQQQ 50/50 drift", half_half_drift(Q, T, CAP)),
        ("B3 TQQQ 100%", strategy_buy_and_hold(T, CAP, "B3")),
        ("B4 QLD 100%", strategy_buy_and_hold(L, CAP, "B4")),
        ("S2 앵커 (2단 스위칭)", strategy_switching(Q, L, T, CAP)[0]),
        ("S3 앵커 (2단+RSI)", strategy_switching_rsi(Q, L, T, CAP)[0]),
        (
            "S4 앵커 (트레일링)",
            strategy_s4_trailing(Q, L, T, CAP, **S4_ANCHOR)[0],
        ),
        (
            "S4-2 (3티어 10/15/20%→포트 25/50/100%)",
            strategy_tiered(Q, L, T, CAP)[0],
        ),
        (
            "S5-2 (S4-2 + NORMAL S5게이트 β=0.5 min_sw>11%)",
            strategy_s5_2(Q, L, T, CAP)[0],
        ),
        (
            "DSA (exp b=2, 25%→100%, drop SL, β=0.5, md=10/20%)",
            strategy_s5(
                Q, L, T, CAP,
                stop_factor=0.75,
                **S5_ANCHOR_COMMON,
            )[0],
        ),
        (
            "DSA (손절 ATH dd ≤ −1.5×진입스윙깊이, drop_deepen 대체)",
            strategy_s5(
                Q, L, T, CAP,
                stop_factor=0.75,
                entry_depth_stop_mult=1.5,
                **S5_ANCHOR_COMMON,
            )[0],
        ),
        (
            "DSA + S4-2식 ATH dd SL (25%→l1 -15%, 중간→l2<-20%, 100%→없음)",
            strategy_s5(
                Q, L, T, CAP,
                stop_factor=0.75,
                attack_stop_mode="s4_dd",
                **S5_ANCHOR_COMMON,
            )[0],
        ),
        (
            "QLD-14/7 Full (100% TQQQ, -14% 재하락 SL, ATH 트레일)",
            strategy_qld_touch_bounce_full(Q, L, T, CAP)[0],
        ),
    ]

    rows = []
    for name, port in runners:
        met = oos_metric_bundle(port)
        rows.append({
            "전략": name,
            "누적수익": met["total_return"],
            "CAGR": met["cagr"],
            "MDD": met["mdd"],
            "Sharpe": met["sharpe"],
            "Sortino": met["sortino"],
            "Ulcer": met["ulcer"],
            "연수": met["n_years"],
        })

    df = pd.DataFrame(rows)
    df_fmt = df.copy()
    df_fmt["누적수익"] = df_fmt["누적수익"].map(lambda x: f"{x:.2%}")
    df_fmt["CAGR"] = df_fmt["CAGR"].map(lambda x: f"{x:.2%}")
    df_fmt["MDD"] = df_fmt["MDD"].map(lambda x: f"{x:.2%}")
    df_fmt["Sharpe"] = df_fmt["Sharpe"].map(lambda x: f"{x:.3f}")
    df_fmt["Sortino"] = df_fmt["Sortino"].map(lambda x: f"{x:.3f}")
    df_fmt["Ulcer"] = df_fmt["Ulcer"].map(lambda x: f"{x:.2f}")
    df_fmt["연수"] = df_fmt["연수"].map(lambda x: f"{x:.2f}")

    period = f"{Q.index[0].date()} ~ {Q.index[-1].date()}  ({len(Q):,}일)"
    print("=" * 100)
    print(f"벤치마크(B1–B4) & S2–S5 앵커 & DSA/S4-2 비교  |  초기자본 ${CAP:,.0f}  |  {period}")
    print("DSA: Dynamic Switching Algorithm (= strategy_s5 앵커); exp_base=2, min→25% max→100%, SL=drop deepen(기본) 또는 attack_stop_mode=s4_dd, β=0.5, min=10% max=20%, trail=-15%")
    print("=" * 100)
    print(df_fmt.to_string(index=False))
    print("=" * 100)

    tag = f"{Q.index[0].strftime('%Y%m%d')}_{Q.index[-1].strftime('%Y%m%d')}"
    csv_path = OUT_DIR / f"benchmark_s2_s5_anchor_table_{tag}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    json_path = OUT_DIR / f"benchmark_s2_s5_anchor_table_{tag}.json"
    payload = {
        "period": [str(Q.index[0].date()), str(Q.index[-1].date())],
        "n_days": int(len(Q)),
        "initial_capital": CAP,
        "dsa_anchor": {
            "display_name": "Dynamic Switching Algorithm (DSA)",
            "note": "exp: frac=lo+(hi-lo)*(b^t-1)/(b-1), b=exp_base(=2), t=(drop-min)/(max-min); SL drop deepen",
            **S5_ANCHOR_COMMON,
        },
        "s4_anchor": S4_ANCHOR,
        "s4_2_note": "QLD 3-tier: -10→-5% (25%), -15→-7.5% (50%), -20→-10% (100%) TQQQ of portfolio; backtest_tiered.strategy_tiered",
        "s5_2_note": "S4-2 + S5 gate on NORMAL-only entries: swing_drop>0.11 (default), rebound<beta*swing_drop",
        "qld_14_7_full": "touch -14%, bounce to >=-7% → 100% TQQQ; if dd<=-14% again while holding → QQQ; dd>=0 → TRAIL -15%",
        "rows": df.to_dict(orient="records"),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\n저장: {csv_path}")
    print(f"저장: {json_path}")


if __name__ == "__main__":
    main()
