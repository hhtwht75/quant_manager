"""
S5 연속 사이징 전략 — 베이지안 스타일 파라미터 최적화 (Optuna TPE)

IS (튜닝) : 2002-10-01 ~ 2019-11-01
OOS (평가): 2019-11-02 ~ 데이터 끝

5차원: beta, max_drop, min_drop, stop_factor, trail  (use_stop_loss=True,
      position_mode=\"linear\" 고정 — 레거시 연속 사이징+손절 튜닝용)

앵커 비교: ANCHOR_S5 는 tiered + SL off (현행 S5 앵커 정의).

목적함수: DE와 동일 스타일 — IS에서 Sharpe×√ΔCAGR 최대화,
          S1 대비 CAGR/Sharpe 미달 시 패널티 (Optuna는 minimize → 음의 점수 최소화)

의존: pip install optuna

출력: 03_RESULT/sensitivity/s5_bayesian_optuna_result.{json,png}

실행: 저장소 루트에서  python3 01_CODE/s5_bayesian_optimize.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Apple SD Gothic Neo"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import (  # noqa: E402
    load_extended_daily,
    strategy_s5,
    strategy_buy_and_hold,
)
from evaluation_metrics import full_metrics, oos_metric_bundle  # noqa: E402

OUT_DIR = _ROOT / "03_RESULT" / "sensitivity"
IS_START = pd.Timestamp("2002-10-01")
IS_END = pd.Timestamp("2019-11-01")
OOS_START = pd.Timestamp("2019-11-02")
CAP = 100_000
N_TRIALS = 200
ANCHOR_S5 = dict(
    beta=0.5, max_drop=0.40, min_drop=0.10, stop_factor=0.97,
    trailing_stop_pct=-0.15, use_stop_loss=True, position_mode="exp", exp_base=2.0,
)


def eval_s5_portfolio(Q, L, T, beta, md, mind, sf, tr, sl=False, position_mode="linear"):
    try:
        port, _ = strategy_s5(
            Q, L, T, CAP,
            beta=float(beta), max_drop=float(md), min_drop=float(mind),
            stop_factor=float(sf), trailing_stop_pct=float(tr),
            use_stop_loss=bool(sl),
            position_mode=str(position_mode),
        )
        return port
    except Exception:
        return None


def s1_bench(Q):
    return strategy_buy_and_hold(Q, CAP, "S1")


def main():
    try:
        import optuna
        from optuna.samplers import TPESampler

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("Optuna 필요:  python3 -m pip install optuna")
        raise SystemExit(1) from None

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("S5 Bayesian-style opt (Optuna TPE)")
    print(f"  IS : {IS_START.date()} ~ {IS_END.date()}")
    print(f"  OOS: {OOS_START.date()} ~")
    print(f"  Trials: {N_TRIALS}")
    print("=" * 72)

    qqq = load_extended_daily("QQQ")
    qld = load_extended_daily("QLD")
    tqqq = load_extended_daily("TQQQ")
    common = qqq.index.intersection(qld.index).intersection(tqqq.index)
    qqq, qld, tqqq = qqq.loc[common], qld.loc[common], tqqq.loc[common]

    m_is = (qqq.index >= IS_START) & (qqq.index <= IS_END)
    Q_is, L_is, T_is = qqq.loc[m_is], qld.loc[m_is], tqqq.loc[m_is]
    Q_oos = qqq[qqq.index >= OOS_START]
    L_oos = qld[qld.index >= OOS_START]
    T_oos = tqqq[tqqq.index >= OOS_START]

    s1_is = full_metrics(s1_bench(Q_is))

    def objective(trial: "optuna.Trial") -> float:
        beta = trial.suggest_float("beta", 0.10, 0.95)
        max_drop = trial.suggest_float("max_drop", 0.03, 0.60)
        min_drop = trial.suggest_float("min_drop", 0.03, 0.22)
        stop_factor = trial.suggest_float("stop_factor", 0.20, 2.0)
        trail = trial.suggest_float("trail", -0.50, -0.02)

        if min_drop >= max_drop * 0.98:
            return 1e6
        if stop_factor <= beta + 0.02:
            return 1e6

        port = eval_s5_portfolio(
            Q_is, L_is, T_is, beta, max_drop, min_drop, stop_factor, trail, True, "linear",
        )
        if port is None or len(port) < 10:
            return 1e6

        r = full_metrics(port)
        cagr_viol = max(0.0, s1_is["cagr"] - r["cagr"])
        sharpe_viol = max(0.0, s1_is["sharpe"] - r["sharpe"])
        if cagr_viol > 0 or sharpe_viol > 0:
            return 100.0 + cagr_viol * 200 + sharpe_viol * 50

        delta_cagr = max(r["cagr"] - s1_is["cagr"], 1e-8)
        score = r["sharpe"] * (delta_cagr ** 0.5)
        return -float(score)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42, multivariate=True),
    )
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    elapsed = time.time() - t0

    bp = study.best_params
    best = {
        "beta": bp["beta"],
        "max_drop": bp["max_drop"],
        "min_drop": bp["min_drop"],
        "stop_factor": bp["stop_factor"],
        "trail": bp["trail"],
        "use_stop_loss": True,
        "position_mode": "linear",
    }

    print("\n[IS 최적 파라미터 (TPE)]")
    for k, v in best.items():
        if k == "use_stop_loss":
            print(f"  {k} = {v}")
        elif k == "trail":
            print(f"  {k} = {v:.5f}  ({v*100:.2f}%)")
        else:
            print(f"  {k} = {v:.5f}")

    port_is_best = eval_s5_portfolio(
        Q_is, L_is, T_is,
        best["beta"], best["max_drop"], best["min_drop"],
        best["stop_factor"], best["trail"], True, "linear",
    )
    port_oos_best = eval_s5_portfolio(
        Q_oos, L_oos, T_oos,
        best["beta"], best["max_drop"], best["min_drop"],
        best["stop_factor"], best["trail"], True, "linear",
    )
    port_oos_anchor = eval_s5_portfolio(
        Q_oos, L_oos, T_oos,
        ANCHOR_S5["beta"], ANCHOR_S5["max_drop"], ANCHOR_S5["min_drop"],
        ANCHOR_S5["stop_factor"], ANCHOR_S5["trailing_stop_pct"],
        ANCHOR_S5["use_stop_loss"], ANCHOR_S5["position_mode"],
    )

    met_is_s5 = full_metrics(port_is_best) if port_is_best is not None else {}

    print("\n[IS 성과 vs S1]")
    print(
        f"  S1   CAGR={s1_is['cagr']:.2%}  Sharpe={s1_is['sharpe']:.3f}  MDD={s1_is['mdd']:.2%}"
    )
    print(
        f"  S5★  CAGR={met_is_s5.get('cagr', 0):.2%}  Sharpe={met_is_s5.get('sharpe', 0):.3f}  "
        f"MDD={met_is_s5.get('mdd', 0):.2%}"
    )

    oos_rows = []
    for lab, p in [
        ("S1 QQQ B&H", s1_bench(Q_oos)),
        ("S5 앵커 (exp, SL=drop deepen)", port_oos_anchor),
        ("S5 Bayes-TPE (IS튜닝)", port_oos_best),
    ]:
        o = oos_metric_bundle(p)
        oos_rows.append({"전략": lab, **{k: o[k] for k in o}})

    print("\n" + "=" * 100)
    print(f"OOS 평가  ({Q_oos.index[0].date()} ~ {Q_oos.index[-1].date()})")
    print("=" * 100)
    hdr = f"{'전략':<22} {'누적수익':>10} {'CAGR':>8} {'MDD':>8} {'Sharpe':>7} {'Sortino':>8} {'Ulcer':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in oos_rows:
        print(
            f"{r['전략']:<22} {r['total_return']:>9.2%} {r['cagr']:>7.2%} {r['mdd']:>7.2%} "
            f"{r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['ulcer']:>7.2f}"
        )
    print("=" * 100)

    payload = {
        "method": "Optuna TPE (Tree-structured Parzen Estimator)",
        "is_period": [str(IS_START.date()), str(IS_END.date())],
        "oos_period": [str(Q_oos.index[0].date()), str(Q_oos.index[-1].date())],
        "n_trials": N_TRIALS,
        "elapsed_sec": round(elapsed, 1),
        "best_params": {
            k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v
            for k, v in best.items()
        },
        "study_best_value": float(study.best_value),
        "is_metrics_s1": {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in s1_is.items()
        },
        "is_metrics_s5_best": {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in met_is_s5.items()
        },
        "oos_comparison": oos_rows,
        "anchor_reference": {k: float(v) if isinstance(v, float) else v for k, v in ANCHOR_S5.items()},
    }
    out_json = OUT_DIR / "s5_bayesian_optuna_result.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    labs = [r["전략"] for r in oos_rows]
    cagr = [r["cagr"] * 100 for r in oos_rows]
    x = np.arange(len(labs))
    ax.bar(x, cagr, color=["#9CA3AF", "#2563EB", "#DC2626"], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=15, ha="right")
    ax.set_ylabel("OOS CAGR (%)")
    ax.set_title("S5 Bayesian (TPE) 튜닝 — OOS CAGR 비교\nIS 2002-10 ~ 2019-11")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_png = OUT_DIR / "s5_bayesian_optuna_result.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n저장: {out_json}")
    print(f"저장: {out_png}")


if __name__ == "__main__":
    main()
