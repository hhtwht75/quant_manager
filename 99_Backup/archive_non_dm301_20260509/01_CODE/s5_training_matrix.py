"""
s5_training_matrix.py
=======================
IS/OOS: 2002-10-01 ~ 2019-11-01 (학습), 2019-11-02 ~ (OOS).

동일 슬라이스에서 벤치마크(B1~B3), S2/S3, S4 트레일링(라벨별), S4 StopLoss(plot 레거시),
S5 앵커, S5 DE 최적(JSON 있을 때) 평가 + IS rolling 리스크 + paired bootstrap vs S1.

실행: 저장소 루트에서  python 01_CODE/s5_training_matrix.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import (
    load_extended_daily,
    strategy_buy_and_hold,
    strategy_switching,
    strategy_switching_rsi,
    strategy_switching_stoploss,
    strategy_s4_trailing,
    strategy_s5,
)
from evaluation_metrics import (
    annualized_sharpe,
    full_metrics,
    paired_bootstrap_compare,
)

IS_START = pd.Timestamp("2002-10-01")
IS_END = pd.Timestamp("2019-11-01")
OOS_START = pd.Timestamp("2019-11-02")
INIT = 100_000
OUT_DIR = _ROOT / "03_RESULT/sensitivity"
ROLL_WIN = 756  # ~3y trading days

# S4 라벨 고정 표: (표시명, param_kind, 파라미터 dict | None)
# param_kind: anchor | de_trained | bayesian_posterior
S4_TRAILING_ROWS: list[tuple[str, str, dict]] = [
    ("S4 Anchor (설계 앵커)", "anchor", {
        "b": -0.05, "r1": 2.0, "r2": 2.0, "trail": -0.15, "hf": 0.50, "ff": 1.0,
    }),
    ("S4 DE-old (IS=2010-19)", "de_trained", {
        "b": -0.10404, "r1": 1.0561, "r2": 2.7402, "trail": -0.0894, "hf": 0.8997, "ff": 1.0,
    }),
    ("S4 DE-extended", "de_trained", {
        "b": -0.10422, "r1": 1.0542, "r2": 1.733, "trail": -0.12906, "hf": 0.8988, "ff": 1.0,
    }),
    ("S4 DE-robust", "de_trained", {
        "b": -0.05578, "r1": 2.0, "r2": 1.7571, "trail": -0.06283, "hf": 0.9810, "ff": 1.0,
    }),
    ("S4 Posterior (Bayesian)", "bayesian_posterior", {
        "b": -0.0469, "r1": 2.0, "r2": 1.978, "trail": -0.0836, "hf": 0.514, "ff": 1.0,
    }),
]


def slice_rng(
    qqq: pd.DataFrame, qld: pd.DataFrame, tqqq: pd.DataFrame,
    start: pd.Timestamp, end: pd.Timestamp | None,
):
    if end is None:
        m = qqq.index >= start
    else:
        m = (qqq.index >= start) & (qqq.index <= end)
    return qqq.loc[m], qld.loc[m], tqqq.loc[m]


def buy_and_hold_close(df: pd.DataFrame, name: str, cap: float = INIT) -> pd.Series:
    sh = cap / df["Close"].iloc[0]
    return pd.Series(sh * df["Close"].values, index=df.index, name=name)


def half_half_drift(
    a: pd.DataFrame, b: pd.DataFrame, name: str, cap: float = INIT,
) -> pd.Series:
    sha = (cap * 0.5) / a["Close"].iloc[0]
    shb = (cap * 0.5) / b["Close"].iloc[0]
    return pd.Series(
        sha * a["Close"].values + shb * b["Close"].values,
        index=a.index,
        name=name,
    )


def run_s4_trailing_portfolio(Q, L, T, p: dict) -> pd.Series:
    r1 = p.get("r1", 2.0)
    sd = p["b"] * r1
    db = p["b"] * p["r2"]
    dd = p["b"] * r1 * p["r2"]
    hs = (sd + dd) / 2
    port, _ = strategy_s4_trailing(
        Q, L, T, INIT,
        shallow_drop=sd, deep_drop=dd,
        shallow_bounce=p["b"], deep_bounce=db,
        half_stop=hs, trailing_stop_pct=p["trail"],
        half_frac=p["hf"], full_frac=p["ff"],
        series_name="S4",
    )
    return port


def rolling_sharpe_stats(series: pd.Series, window: int = ROLL_WIN) -> dict | None:
    r = series.pct_change().dropna()
    if len(r) < window + 5:
        return None
    sharpes = []
    for i in range(len(r) - window + 1):
        seg = r.iloc[i : i + window].values
        sharpes.append(annualized_sharpe(seg))
    arr = np.array(sharpes)
    return {
        "window_days": window,
        "n_windows": int(len(arr)),
        "min_rolling_sharpe": float(arr.min()),
        "p10_rolling_sharpe": float(np.percentile(arr, 10)),
        "median_rolling_sharpe": float(np.median(arr)),
    }


def min_rolling_cagr(wealth: pd.Series, window: int = ROLL_WIN) -> dict | None:
    r = wealth.pct_change().dropna()
    if len(r) < window + 5:
        return None
    ny = window / 252.0
    cagrs = []
    for i in range(len(r) - window + 1):
        seg = r.iloc[i : i + window]
        cum = float((1 + seg).prod())
        cagrs.append(cum ** (1.0 / ny) - 1.0 if cum > 0 else -1.0)
    arr = np.array(cagrs)
    return {
        "window_days": window,
        "min_rolling_cagr": float(arr.min()),
        "median_rolling_cagr": float(np.median(arr)),
    }


def bootstrap_summarize(strat: pd.Series, bench: pd.Series, n_iter: int = 500) -> dict:
    out = paired_bootstrap_compare(
        strat.pct_change(), bench.pct_change(),
        block_len=60, n_iter=n_iter, seed=42,
    )
    if "raw" in out:
        del out["raw"]
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("S5 training matrix — extended daily")
    print(f"  IS : {IS_START.date()} ~ {IS_END.date()}")
    print(f"  OOS: {OOS_START.date()} ~ (끝)")
    print("=" * 72)

    qqq = load_extended_daily("QQQ")
    qld = load_extended_daily("QLD")
    tqqq = load_extended_daily("TQQQ")
    common = qqq.index.intersection(qld.index).intersection(tqqq.index)
    qqq, qld, tqqq = qqq.loc[common], qld.loc[common], tqqq.loc[common]

    Q_is, L_is, T_is = slice_rng(qqq, qld, tqqq, IS_START, IS_END)
    Q_oos, L_oos, T_oos = slice_rng(qqq, qld, tqqq, OOS_START, None)
    Q_all, L_all, T_all = slice_rng(qqq, qld, tqqq, IS_START, None)

    s1_is = strategy_buy_and_hold(Q_is, INIT, "S1")
    s1_oos = strategy_buy_and_hold(Q_oos, INIT, "S1")
    s1_all = strategy_buy_and_hold(Q_all, INIT, "S1")

    runners: list[tuple[str, str, dict | None, callable]] = []

    # 벤치마크
    runners.append(("B1 QQQ 100%", "benchmark", None, lambda Q, L, T: buy_and_hold_close(Q, "B1")))
    runners.append(("B2 QQQ/TQQQ 50/50 drift", "benchmark", None, lambda Q, L, T: half_half_drift(Q, T, "B2")))
    runners.append(("B3 TQQQ 100%", "benchmark", None, lambda Q, L, T: buy_and_hold_close(T, "B3")))

    # S2 / S3
    runners.append(("S2 Switching (2단)", "s2", None, lambda Q, L, T: strategy_switching(Q, L, T, INIT)[0]))
    runners.append(("S3 Switching+RSI", "s3", None, lambda Q, L, T: strategy_switching_rsi(Q, L, T, INIT)[0]))

    # S4 트레일링 (표별)
    for label, kind, params in S4_TRAILING_ROWS:
        runners.append((
            label, "s4_trailing", {"kind": kind, **params},
            lambda Q, L, T, p=params: run_s4_trailing_portfolio(Q, L, T, p),
        ))

    # S4 StopLoss — plot_results 레거시 명명과 동일 계열
    runners.append((
        "S4 Switching+StopLoss (레거시 차트 명칭)", "s4_stoploss", None,
        lambda Q, L, T: strategy_switching_stoploss(Q, L, T, INIT)[0],
    ))

    # S5 앵커：지수 사이징 + 스윙 재악화 시 손절
    runners.append((
        "S5 앵커 (exp b=2, SL=drop deepen, β=0.5, 10/40%, trail=-15%)", "s5",
        {"beta": 0.5, "max_drop": 0.40, "min_drop": 0.10, "trail": -0.15,
         "use_stop_loss": True, "position_mode": "exp", "exp_base": 2.0},
        lambda Q, L, T: strategy_s5(
            Q, L, T, INIT,
            beta=0.5, max_drop=0.40, min_drop=0.10,
            trailing_stop_pct=-0.15, use_stop_loss=True,
            position_mode="exp", exp_base=2.0,
        )[0],
    ))

    de_path = OUT_DIR / "s5_optimization_result.json"
    if de_path.exists():
        with open(de_path) as f:
            dej = json.load(f)
        op = dej.get("optimal_params") or {}
        if all(k in op for k in ("beta", "max_drop", "min_drop", "stop_factor", "trail")):
            b, md, mind, sf, tr = (
                op["beta"], op["max_drop"], op["min_drop"],
                op["stop_factor"], op["trail"],
            )
            runners.append((
                "S5 DE optimal (JSON, use_stop_loss=True)", "s5_de",
                op,
                lambda Q, L, T, _b=b, _md=md, _mind=mind, _sf=sf, _tr=tr: strategy_s5(
                    Q, L, T, INIT,
                    beta=_b, max_drop=_md, min_drop=_mind,
                    stop_factor=_sf, trailing_stop_pct=_tr, use_stop_loss=True,
                    position_mode="linear",
                )[0],
            ))
            print(f"  (로드) DE 파라미터 from {de_path.name}")
        else:
            print(f"  (건너뜀) {de_path.name}에 min_drop 등 5파라미터 없음 — backtest_optimize_s5.py 실행 후 재실행")

    table = []
    s4_meta = []

    for label, family, meta, fn in runners:
        try:
            p_is = fn(Q_is, L_is, T_is)
            p_oos = fn(Q_oos, L_oos, T_oos)
            p_all = fn(Q_all, L_all, T_all)
        except Exception as e:
            print(f"  ✗ {label}: {e}")
            continue

        m_is = full_metrics(p_is)
        m_oos = full_metrics(p_oos)
        m_all = full_metrics(p_all)

        row = {
            "label": label,
            "family": family,
            "meta": meta,
            "is": m_is,
            "oos": m_oos,
            "full_from_is_start": m_all,
        }
        if family == "s4_trailing" and isinstance(meta, dict):
            s4_meta.append({
                "label": label,
                "param_kind": meta.get("kind"),
                "params": {k: v for k, v in meta.items() if k != "kind"},
            })

        # IS robustness vs S1 only for non-benchmark (benchmarks still get rolling stats)
        row["is_rolling_sharpe"] = rolling_sharpe_stats(p_is)
        row["is_min_rolling_cagr"] = min_rolling_cagr(p_is)
        if label != "B1 QQQ 100%":
            row["is_bootstrap_vs_s1"] = bootstrap_summarize(p_is, s1_is)
        table.append(row)

        sig = ""
        if row.get("is_bootstrap_vs_s1"):
            pval = row["is_bootstrap_vs_s1"]["delta_sharpe"]["p_value"]
            sig = "★" if pval < 0.05 else ("·" if pval < 0.2 else "")
        print(
            f"  {label[:50]:50}  IS Sh={m_is['sharpe']:.2f}  OOS Sh={m_oos['sharpe']:.2f}  {sig}"
        )

    out = {
        "is_range": [str(IS_START.date()), str(IS_END.date())],
        "oos_start": str(OOS_START.date()),
        "s4_trailing_definitions": s4_meta,
        "results": table,
    }
    out_path = OUT_DIR / "s5_training_matrix.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n저장:", out_path)


if __name__ == "__main__":
    main()
