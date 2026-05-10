"""
dynamic_optimize_train.py
==========================

Dynamic-A 전략 3가지 연속형 파라미터를 학습(In-Sample, 2004-2011)으로 추정합니다.

파라미터
--------
1. entry_retrace_k ("저점/진입" 비율)
   ATH 대비 낙폭이 A%(%)일 때, QLD가 ATH에서 A×k까지 회복했다고 보면
   가격 트리거 = ATH × (1 − k × A / 100).
   k=0.5 → 과거 디폴트와 동일(저점과 ATH 중간까지 반등).

2. invest_divisor (투입 규모)
   TQQQ 비중 = min(A / divisor, 1). divisor가 작을수록 동일 A에서 더 많이 진입.

3. stop_dd_k (손절선 위치)
   ATH 기준 추가 하방 한계 = ATH × (1 − k_stop × A / 100).
   k_stop=0.75 → 과거 디폴트와 동일(ATH 대비 낙폭 3A/4까지 허용 후 청산).

최적화 방법론 (요약)
--------------------
• **Differential Evolution (DE)** : 연속 변수·다봉우 함수에서 국소해에 빠지기 쉬운 한계를 완화.
• **목표함수(IS 복합 점수)** : Sharpe + w_s·Sortino − w_u·Ulcer − w_dd·(MDD 과도 시 패널티)
  과적합을 완만히 줄이기 위해 CAGR 단독 최대화는 지양하고, 분산하방(Ulcer)과 MDD에 가중 패널티.
• 결과는 학습구간 성과만 “최적”이므로 반드시 OOS 재검증 권장 (본 스크립트는 참고용 OOS도 출력).

출력
----
03_RESULT/sensitivity/dynamic_train_2004_2011.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_switching import load_extended_daily
from backtest_dynamic import strategy_dynamic
from evaluation_metrics import full_metrics

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_JSON = BASE_DIR / "03_RESULT" / "sensitivity" / "dynamic_train_2004_2011.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# 학습 구간(IS)
TRAIN_START = "2004-01-02"
TRAIN_END   = "2011-12-31"

DE_SEED      = 17
POPSIZE      = 14
MAXITER      = 45
ATOL_FTOL    = 1e-4

# 복합 점수 가중치
W_SORTINO = 0.35
W_ULCER   = 0.10       # ulcer 스케일 ~10–30
MDD_HARD  = 0.52       # 학습구간에서 -52% 깊어지면 강패널티
LAMB_MDD  = 12.0


def load_slice(start: str, end: str):
    qqq = load_extended_daily("QQQ")
    qld = load_extended_daily("QLD")
    tqq = load_extended_daily("TQQQ")
    common = qqq.index.intersection(qld.index.intersection(tqq.index))
    common = common[(common >= start) & (common <= end)]
    return qqq.loc[common], qld.loc[common], tqq.loc[common]


def composite_score(metrics: dict) -> float:
    """높을수록 우수."""
    sharpe = float(metrics["sharpe"])
    sor    = float(metrics["sortino"])
    ulcer = float(metrics["ulcer"])
    mdd    = abs(float(metrics["mdd"]))  # 양수로
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


def fmt_row(label: str, m: dict) -> str:
    return (f"{label:22s}  CAGR={m['cagr']*100:+6.2f}%  Sharpe={m['sharpe']:.3f}  "
            f"Sortino={m['sortino']:.3f}  MDD={m['mdd']*100:+7.2f}%  Ulcer={m['ulcer']:.2f}")


def main():
    init = 100_000

    Qt, Lt, Tt = load_slice(TRAIN_START, TRAIN_END)
    port_b, _ = strategy_dynamic(Qt, Lt, Tt, init)
    mb = full_metrics(port_b)
    baseline_score = composite_score(mb)

    bounds = [
        (0.30, 0.72),      # entry_retrace_k — 너무 낮으면 너무 이른 진입
        (22.0, 95.0),      # invest_divisor — 기본 40 중심
        (0.55, 0.92),      # stop_dd_k — 기본 0.75 중심
    ]

    def objective(vec):
        return -eval_params(vec, Qt, Lt, Tt, init)

    rng = np.random.default_rng(DE_SEED)

    print("=" * 80)
    print("  Dynamic-A 3-parameter training (DE)")
    print(f"  IS: {TRAIN_START} ~ {TRAIN_END}  ({len(Qt):,} 일)")
    print("=" * 80)
    print(f"\n  Baseline(k=0.5, div=40, stop=0.75)  composite={baseline_score:.4f}")
    print(f"    {fmt_row('', mb)}\n")

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

    k_ent_opt, div_opt, k_stop_opt = [float(z) for z in res.x]
    port_opt, ev_opt = strategy_dynamic(
        Qt, Lt, Tt, init,
        entry_retrace_k=k_ent_opt,
        invest_divisor=div_opt,
        stop_dd_k=k_stop_opt,
        series_name="Dynamic-A OPT",
    )
    mo = full_metrics(port_opt)
    score_opt = composite_score(mo)

    print("  ── 최적 결과 (DE) ──")
    print(f"    entry_retrace_k  = {k_ent_opt:.4f}")
    print(f"    invest_divisor   = {div_opt:.2f}")
    print(f"    stop_dd_k        = {k_stop_opt:.4f}")
    print(f"    composite_score  = {score_opt:.4f} (baseline {baseline_score:+.4f})")
    print(f"    {fmt_row('', mo)}")
    print(f"    n_iter={res.nit}  success={res.success}")

    # OOS 간단 검증창들
    oos_slices = [
        ("OOS 2012-2020", "2012-01-01", "2020-12-31"),
        ("OOS 2021-2025", "2021-01-01", "2025-12-31"),
        ("FULL 2004-2026", TRAIN_START, "2026-04-30"),
    ]

    rows = [{"period": TRAIN_START + "~" + TRAIN_END + " IS",
             "baseline": mb, "opt": mo, "params": None}]
    bundle = {
        "train_period": {"start": TRAIN_START, "end": TRAIN_END},
        "baseline_params": {"entry_retrace_k": 0.5, "invest_divisor": 40, "stop_dd_k": 0.75},
        "optimized_params": {
            "entry_retrace_k": k_ent_opt,
            "invest_divisor": div_opt,
            "stop_dd_k": k_stop_opt,
        },
        "de_result": {"fun": float(res.fun), "nit": int(res.nit),
                      "success": bool(res.success), "message": str(res.message)},
        "objective": {"W_SORTINO": W_SORTINO, "W_ULCER": W_ULCER,
                      "MDD_HARD": MDD_HARD, "LAMB_MDD": LAMB_MDD},
        "oos_metrics": [],
    }

    print(f"\n{'=' * 80}")
    print("  OOS 간단 검증 (참고)")
    print(f"{'=' * 80}\n")

    for label, st, ed in oos_slices:
        Q_, L_, T_ = load_slice(st, ed)
        pb, _ = strategy_dynamic(Q_, L_, T_, init)
        po, _ = strategy_dynamic(
            Q_, L_, T_, init,
            entry_retrace_k=k_ent_opt,
            invest_divisor=div_opt,
            stop_dd_k=k_stop_opt,
        )
        mb_o, mo_o = full_metrics(pb), full_metrics(po)
        bundle["oos_metrics"].append({"label": label, "start": st, "end": ed,
                                      "baseline": {k: float(mb_o[k]) for k in mb_o},
                                      "optimized": {k: float(mo_o[k]) for k in mo_o}})
        print(f"  [{label}]  {len(Q_)}일")
        print(f"    Base   {fmt_row('', mb_o)}")
        print(f"    OPT    {fmt_row('', mo_o)}\n")

    bundle["baseline_is"] = {k: float(mb[k]) for k in mb}
    bundle["optimized_is"] = {k: float(mo[k]) for k in mo}

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)

    print(f"  JSON 저장 → {OUT_JSON}")
    print("\n[완료]")


if __name__ == "__main__":
    main()
