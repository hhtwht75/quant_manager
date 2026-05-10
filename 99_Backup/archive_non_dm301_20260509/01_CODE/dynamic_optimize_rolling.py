"""
dynamic_optimize_rolling.py
===========================
Dynamic-A 워크포워드 + OOS 평가 (사용자 사양)

• **롤링 학습(보정 단계)**  
  - 데이터 시작 **2002**부터 사용.  
  - 각 **캘린더 연도 y**에 대해 **학습구간 [y−4, y−1]** 에서 DE 최적화.  
  - **보정 출력**: 같은 규칙으로 **2006년~2016년** OOS 패널(각 연도는 직전 4년으로 학습한 파라미터 적용).  

• **평가(리포트 초점)**  
  - 같은 스케줄 규칙을 **2017년~2026년**까지 연속 적용하여 **평가 구간**(기본 종료 `2026-04-30`) 성과 출력.  

• 워밍업: **2002~2005** 는 ATH/포지션 연속만 위해 **베이스 파라미터**(0.5 / 40 / 0.75).

참고: 직전 채팅의 2016–2022 스펙은 **오타 교정**(2002 시작, 2006–2016 보정 후 2017–2026 평가).

출력: 03_RESULT/sensitivity/dynamic_wf_roll_2002_cal_2017_eval.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_dynamic import strategy_dynamic, strategy_dynamic_schedule
from backtest_switching import load_extended_daily
from backtest_tiered import strategy_tiered
from evaluation_metrics import full_metrics

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_JSON = BASE_DIR / "03_RESULT" / "sensitivity" / "dynamic_wf_roll_2002_cal_2017_eval.json"

DATA_START  = "2002-01-02"
DATA_END    = "2026-04-30"
WARMUP_END_YEAR = 2005          # inclusive: 2002-2005 uses baseline params
FIRST_WF_YEAR   = 2006          # rolling params from this calendar year onward
LAST_WF_YEAR    = 2026          # inclusive
CAL_LAST_YEAR   = 2016          # calibration panel ends here (printing / JSON section)

BASE_K_ENT = 0.5
BASE_DIV   = 40.0
BASE_STOP  = 0.75

EVAL_FIRST_YEAR = 2017

DE_SEED   = 101
POPSIZE   = 11
MAXITER   = 32
ATOL_FTOL = 1e-4

W_SORTINO = 0.35
W_ULCER   = 0.10
MDD_HARD  = 0.52
LAMB_MDD  = 12.0


def load_slice(start: str, end: str):
    qqq = load_extended_daily("QQQ")
    qld = load_extended_daily("QLD")
    tqq = load_extended_daily("TQQQ")
    common = qqq.index.intersection(qld.index.intersection(tqq.index))
    common = common[(common >= start) & (common <= end)]
    return qqq.loc[common], qld.loc[common], tqq.loc[common]


def composite_score(metrics: dict) -> float:
    sharpe = float(metrics["sharpe"])
    sor = float(metrics["sortino"])
    ulcer = float(metrics["ulcer"])
    mdd = abs(float(metrics["mdd"]))
    penalty = max(0.0, mdd - MDD_HARD)
    return sharpe + W_SORTINO * sor - W_ULCER * ulcer - LAMB_MDD * (penalty ** 2)


def eval_params(x: np.ndarray, Q, L, T, init_cap: float) -> float:
    k_ent, div, k_stop = float(x[0]), float(x[1]), float(x[2])
    div = max(div, 1e-6)
    port, _ = strategy_dynamic(
        Q, L, T, init_cap,
        entry_retrace_k=k_ent,
        invest_divisor=div,
        stop_dd_k=k_stop,
    )
    m = full_metrics(port)
    if not np.isfinite(m["sharpe"]) or m["cagr"] < -0.999:
        return -1e6
    return composite_score(m)


def optimize_fold(Q, L, T, init_cap: int, seed: int) -> dict:
    bounds = [
        (0.30, 0.72),
        (22.0, 95.0),
        (0.55, 0.92),
    ]

    def objective(vec):
        return -eval_params(vec, Q, L, T, init_cap)

    rng = np.random.default_rng(seed)
    res = differential_evolution(
        objective,
        bounds,
        strategy="best1bin",
        maxiter=MAXITER,
        popsize=POPSIZE,
        tol=ATOL_FTOL,
        atol=ATOL_FTOL,
        seed=rng,
        workers=1,
        polish=False,
        updating="immediate",
    )
    k_ent, div, k_stop = [float(z) for z in res.x]
    return {
        "entry_retrace_k": k_ent,
        "invest_divisor": div,
        "stop_dd_k": k_stop,
        "de_fun": float(res.fun),
        "de_nit": int(res.nit),
        "de_success": bool(res.success),
    }


def train_window_for_year(y: int) -> tuple[str, str]:
    """Calendar year y 를 OOS 에 쓸 때 학습 시작/끝 (그 이전까지)."""
    return f"{y - 4}-01-02", f"{y - 1}-12-31"


def build_param_fn(year_params: dict[int, tuple[float, float, float]]) -> callable:
    def param_fn(ts: pd.Timestamp) -> tuple[float, float, float]:
        y = int(ts.year)
        if y <= WARMUP_END_YEAR:
            return (BASE_K_ENT, BASE_DIV, BASE_STOP)
        if FIRST_WF_YEAR <= y <= LAST_WF_YEAR and y in year_params:
            return year_params[y]
        return (BASE_K_ENT, BASE_DIV, BASE_STOP)

    return param_fn


def fmt_metrics(m: dict) -> str:
    return (f"CAGR={m['cagr']*100:+6.2f}%  Sharpe={m['sharpe']:.3f}  "
            f"Sortino={m['sortino']:.3f}  MDD={m['mdd']*100:+7.2f}%  Ulcer={m['ulcer']:.2f}")


def main():
    init = 100_000
    year_params: dict[int, tuple[float, float, float]] = {}
    folds_log = []

    print("=" * 84)
    print(f"  Dynamic-A 워크포워드  |  학습창 길이: 4년  |  데이터 {DATA_START} ~ {DATA_END}")
    print(f"  워밍업(베이스): {DATA_START.split('-')[0]}–{WARMUP_END_YEAR}  |  "
          f"롤링 파라미터: {FIRST_WF_YEAR}–{LAST_WF_YEAR}")
    print(f"  보정 패널(참고): {FIRST_WF_YEAR}–{CAL_LAST_YEAR}  |  **평가 리포트: {EVAL_FIRST_YEAR}–{LAST_WF_YEAR}**")
    print("=" * 84)

    for y in range(FIRST_WF_YEAR, LAST_WF_YEAR + 1):
        tr_s, tr_e = train_window_for_year(y)
        Qt, Lt, Tt = load_slice(tr_s, tr_e)
        if len(Qt) < 480:
            raise RuntimeError(f"학습 부족 y={y} ({len(Qt)}일)")
        seed = DE_SEED + y * 131
        opt = optimize_fold(Qt, Lt, Tt, init, seed)
        k_ent = opt["entry_retrace_k"]
        div = opt["invest_divisor"]
        k_stop = opt["stop_dd_k"]
        year_params[y] = (k_ent, div, k_stop)

        port_is, _ = strategy_dynamic(
            Qt, Lt, Tt, init,
            entry_retrace_k=k_ent,
            invest_divisor=div,
            stop_dd_k=k_stop,
        )
        m_is = full_metrics(port_is)

        fold = {
            "oos_calendar_year_for_params": y,
            "train_start": tr_s,
            "train_end": tr_e,
            "train_days": int(len(Qt)),
            "params": {"entry_retrace_k": k_ent, "invest_divisor": div, "stop_dd_k": k_stop},
            "de": {k: opt[k] for k in ("de_fun", "de_nit", "de_success")},
            "train_is_metrics": {k: float(m_is[k]) for k in m_is},
            "composite_is": float(composite_score(m_is)),
        }
        folds_log.append(fold)

        flag = "*" if y <= CAL_LAST_YEAR else " "
        print(f"{flag} [{y}] 학습 {tr_s} ~ {tr_e}  "
              f"k={k_ent:.4f} div={div:.2f} stop={k_stop:.4f}  "
              f"IS CAGR={m_is['cagr']*100:+5.1f}% Sh={m_is['sharpe']:.3f}")

    Qf, Lf, Tf = load_slice(DATA_START, DATA_END)
    param_fn = build_param_fn(year_params)

    port_wf, ev_wf = strategy_dynamic_schedule(
        Qf, Lf, Tf, init, param_fn, series_name="Dynamic WF-roll",
    )
    baseline, _ = strategy_dynamic(Qf, Lf, Tf, init)
    tier, _ = strategy_tiered(Qf, Lf, Tf, init)

    ix_eval = port_wf.index[
        (port_wf.index >= pd.Timestamp(f"{EVAL_FIRST_YEAR}-01-01"))
        & (port_wf.index <= pd.Timestamp(DATA_END))
    ]

    wf_e = port_wf.loc[ix_eval]
    bs_e = baseline.loc[ix_eval]
    t3_e = tier.loc[ix_eval]

    bm_qq = Qf["Close"] / Qf["Close"].iloc[0] * init
    q_e = bm_qq.loc[ix_eval]

    mw = full_metrics(wf_e)
    mb = full_metrics(bs_e)
    mt3 = full_metrics(t3_e)
    mq = full_metrics(q_e)

    qqq_off = wf_e.iloc[0] / q_e.iloc[0]
    mq_ex = full_metrics(q_e * qqq_off)

    print(f"\n{'=' * 84}")
    print(f"  평가 구간 단독 재기준 성과 ({EVAL_FIRST_YEAR} ~ {DATA_END}, 일수 {len(ix_eval):,})")
    print(f"{'=' * 84}\n")
    print(f"  {'전략':30s}  {fmt_metrics(mw)}")
    print(f"  {'Baseline Dynamic-A (디폴트 고정)':30s}  {fmt_metrics(mb)}")
    print(f"  {'3-Tier':30s}  {fmt_metrics(mt3)}")
    print(f"  {'QQQ (동일 시작가로 스케일)':30s}  {fmt_metrics(mq_ex)}")

    print(f"\n  연간 수익률 WF vs 베이스 ({EVAL_FIRST_YEAR}–2026 또는 데이터 끝):")
    print(f"  {'연도':>5}   {'WF-roll':>8}   {'BaseDyn':>8}")
    print("  " + "-" * 28)
    for yr in range(EVAL_FIRST_YEAR, int(pd.Timestamp(DATA_END).year) + 1):
        mk = wf_e.index.year == yr
        if mk.sum() < 5:
            continue
        rwf = (wf_e[mk].iloc[-1] / wf_e[mk].iloc[0] - 1) * 100
        rbs = (bs_e[mk].iloc[-1] / bs_e[mk].iloc[0] - 1) * 100
        mark = "★" if rwf > rbs + 0.8 else ("▼" if rwf < rbs - 0.8 else " ")
        print(f"  {yr:>5}   {rwf:>+7.1f}%{mark}   {rbs:>+7.1f}%")

    bundle = {
        "data_start": DATA_START,
        "data_end": DATA_END,
        "warmup_baseline_through_year": WARMUP_END_YEAR,
        "wf_param_years": f"{FIRST_WF_YEAR}-{LAST_WF_YEAR}",
        "train_window_calendar_years": 4,
        "calibration_years_logged": f"{FIRST_WF_YEAR}-{CAL_LAST_YEAR}",
        "eval_focus_years": f"{EVAL_FIRST_YEAR}-{LAST_WF_YEAR}",
        "year_params": {
            str(k): {"entry_retrace_k": v[0], "invest_divisor": v[1], "stop_dd_k": v[2]}
            for k, v in sorted(year_params.items())
        },
        "folds": folds_log,
        "eval_slice_metrics": {
            "wf_schedule": {k: float(mw[k]) for k in mw},
            "dynamic_baseline_fixed": {k: float(mb[k]) for k in mb},
            "three_tier": {k: float(mt3[k]) for k in mt3},
            "qqq_scaled": {k: float(mq_ex[k]) for k in mq_ex},
        },
        "weights": {"W_SORTINO": W_SORTINO, "W_ULCER": W_ULCER,
                    "MDD_HARD": MDD_HARD, "LAMB_MDD": LAMB_MDD},
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)

    print(f"\n  JSON → {OUT_JSON}")
    print("\n[완료]")


if __name__ == "__main__":
    main()
