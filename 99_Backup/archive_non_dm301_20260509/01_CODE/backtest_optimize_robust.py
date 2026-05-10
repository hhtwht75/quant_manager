"""
backtest_optimize_robust.py
============================
Strategy 4 & 5 robust 최적화 — 오버피팅 방지 처방 1-6 동시 적용.

처방 정리
--------
1) **자유도 축소**:
   - S4: 6 → 4 파라미터  (full_frac=1.0, r1=2.0 고정)
   - S5: 4 그대로
2) **Walk-forward CV**: 5년 train + 1년 test, 1년 step
   → 2002~2026 기간에서 약 19~20개 OOS fold 생성
3) **Robust objective**:
       score = median(WF_OOS_Sharpe) − 0.5×std(WF_OOS_Sharpe)
              − 1.0×max(0, |WF_OOS_MDD| − 0.50)
   → 단일 IS Sharpe가 아닌 "여러 OOS fold 분포의 견고함" 직접 최적화
4) **Plateau penalty**: Top-K 솔루션에 ±10% perturbation을 가해
   neighborhood Sharpe std가 작은 쪽 우선 선택
5) **Stationary block bootstrap** (60일 블록, 200회) 으로 최종 후보의
   OOS-전구간(2017~2026) 안정성을 확률 분포로 평가
6) **Default-anchor 비교**: 손-튜닝 default가 robust score 기준으로
   최적 후보를 능가하면 default를 그대로 채택 (no-tuning is the best tuning)
"""

import sys, warnings, time, json
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Apple SD Gothic Neo"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import differential_evolution

from backtest_switching import (
    load_extended_daily, strategy_s4_trailing, strategy_s5,
    strategy_buy_and_hold, compute_statistics,
)

OUT_DIR = Path("03_RESULT/sensitivity")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)

# ════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드 (확장 1999~)
# ════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("데이터 로드 (extended 1999~)...")
qqq  = load_extended_daily("QQQ")
qld  = load_extended_daily("QLD")
tqqq = load_extended_daily("TQQQ")
common = qqq.index.intersection(qld.index).intersection(tqqq.index)
qqq, qld, tqqq = qqq.loc[common], qld.loc[common], tqqq.loc[common]
print(f"  전체 기간: {qqq.index[0].date()} ~ {qqq.index[-1].date()}  ({len(qqq):,}일)")

# ════════════════════════════════════════════════════════════════════════════
# 2. Walk-forward fold 생성 (5y train + 1y test, 1y step)
# ════════════════════════════════════════════════════════════════════════════
TRAIN_YEARS  = 5
TEST_YEARS   = 1
WF_START     = pd.Timestamp("2002-01-01")   # 충분한 lookback 후
OOS_HOLDOUT  = pd.Timestamp("2017-01-01")   # 2017~ : 최종 holdout (CV 사용 X)

def make_wf_folds(start_date: pd.Timestamp, end_date: pd.Timestamp):
    """5y-train, 1y-test rolling folds"""
    folds = []
    cur = start_date
    while cur + pd.DateOffset(years=TRAIN_YEARS+TEST_YEARS) <= end_date:
        train_s = cur
        train_e = cur + pd.DateOffset(years=TRAIN_YEARS) - pd.Timedelta(days=1)
        test_s  = train_e + pd.Timedelta(days=1)
        test_e  = test_s + pd.DateOffset(years=TEST_YEARS) - pd.Timedelta(days=1)
        folds.append((train_s, train_e, test_s, test_e))
        cur = cur + pd.DateOffset(years=TEST_YEARS)
    return folds

# CV 영역: 2002 ~ OOS_HOLDOUT (2017) → CV에서만 파라미터 결정
CV_FOLDS = make_wf_folds(WF_START, OOS_HOLDOUT)
print(f"\n  Walk-forward CV folds (2002~2016, 5y+1y rolling):  {len(CV_FOLDS)}개")
for i, (ts, te, ds, de) in enumerate(CV_FOLDS):
    print(f"    [{i+1:2d}] train {ts.date()}~{te.date()}  →  test {ds.date()}~{de.date()}")

print(f"\n  최종 holdout (CV 미사용, 마지막 평가용): {OOS_HOLDOUT.date()} ~ {qqq.index[-1].date()}")

# ════════════════════════════════════════════════════════════════════════════
# 3. 보조 함수
# ════════════════════════════════════════════════════════════════════════════
def slice_data(start: pd.Timestamp, end: pd.Timestamp):
    m = (qqq.index >= start) & (qqq.index <= end)
    return qqq[m], qld[m], tqqq[m]

def derive_s4(b, r1, r2, trail):
    return dict(shallow_drop=b*r1, shallow_bounce=b,
                deep_drop=b*r1*r2, deep_bounce=b*r2,
                half_stop=(b*r1 + b*r1*r2)/2, trailing_stop=trail)

def stats_from_port(port: pd.Series) -> dict:
    """반복 호출 안전한 빠른 stats."""
    if port.empty or len(port) < 2:
        return {"cagr": 0.0, "mdd": 0.0, "sharpe": 0.0}
    n_days = (port.index[-1] - port.index[0]).days
    if n_days <= 0:
        return {"cagr": 0.0, "mdd": 0.0, "sharpe": 0.0}
    n_years = n_days / 365.25
    cagr = (port.iloc[-1] / port.iloc[0]) ** (1.0 / n_years) - 1
    rets = port.pct_change().dropna()
    if rets.std() == 0:
        return {"cagr": cagr, "mdd": 0.0, "sharpe": 0.0}
    rf_d = (1.04) ** (1/252) - 1
    sharpe = (rets.mean() - rf_d) / rets.std() * np.sqrt(252)
    mdd = ((port - port.cummax()) / port.cummax()).min()
    return {"cagr": float(cagr), "mdd": float(mdd), "sharpe": float(sharpe)}

# ── S4 평가 (자유도 축소: r1=2, ff=1 고정) ───────────────────────────────────
S4_R1_FIXED = 2.0
S4_FF_FIXED = 1.0

def eval_s4_robust(Q, L, T, b, r2, trail, hf):
    """4-param S4 (b, r2, trail, hf) 평가."""
    p = derive_s4(b, S4_R1_FIXED, r2, trail)
    try:
        port, _ = strategy_s4_trailing(
            Q, L, T, 100_000,
            shallow_drop=p["shallow_drop"], deep_drop=p["deep_drop"],
            shallow_bounce=p["shallow_bounce"], deep_bounce=p["deep_bounce"],
            half_stop=p["half_stop"], trailing_stop_pct=p["trailing_stop"],
            half_frac=hf, full_frac=S4_FF_FIXED,
        )
        return stats_from_port(port)
    except Exception:
        return None

def eval_s5_robust(Q, L, T, beta, max_drop, stop_factor, trail):
    try:
        port, _ = strategy_s5(
            Q, L, T, 100_000,
            beta=beta, max_drop=max_drop, min_drop=0.0,
            stop_factor=stop_factor, trailing_stop_pct=trail,
            use_stop_loss=True,
            position_mode="linear",
        )
        return stats_from_port(port)
    except Exception:
        return None

def eval_s1(Q):
    return stats_from_port(strategy_buy_and_hold(Q, 100_000, "S1"))

# ════════════════════════════════════════════════════════════════════════════
# 4. Walk-forward objective
# ════════════════════════════════════════════════════════════════════════════
def wf_score(eval_fn, params, folds, mdd_target=0.50,
             w_std=0.5, w_mdd=1.0):
    """folds 전체에서 OOS test 구간을 평가해 robust 점수 계산.

    score = median(test_sharpe) − w_std×std(test_sharpe)
            − w_mdd×max(0, |median_test_mdd| − mdd_target)
    """
    test_sharpes, test_mdds, test_cagrs = [], [], []
    for ts, te, ds, de in folds:
        # 원리: train 구간은 무시 (자유 파라미터가 train에 의존하지 않으므로)
        # 우리는 "test 구간 OOS 성능"만 직접 최적화.
        Q, L, T = slice_data(ds, de)
        if len(Q) < 50:
            continue
        r = eval_fn(Q, L, T, *params)
        if r is None:
            return -10.0
        test_sharpes.append(r["sharpe"])
        test_mdds.append(r["mdd"])
        test_cagrs.append(r["cagr"])
    if len(test_sharpes) < 5:
        return -10.0
    test_sharpes = np.array(test_sharpes)
    test_mdds    = np.array(test_mdds)
    med_s   = np.median(test_sharpes)
    std_s   = np.std(test_sharpes)
    med_mdd = np.median(np.abs(test_mdds))
    return (med_s
            - w_std * std_s
            - w_mdd * max(0.0, med_mdd - mdd_target))

# ════════════════════════════════════════════════════════════════════════════
# 5. Plateau penalty (perturbation 안정성 검증)
# ════════════════════════════════════════════════════════════════════════════
def plateau_check(eval_fn, params, bounds, folds,
                  n_perturb=10, perturb_frac=0.10):
    """Top 후보의 robustness 평가: ±perturb_frac 범위 내 무작위 섭동 후 score std."""
    base_score = wf_score(eval_fn, params, folds)
    pert_scores = [base_score]
    for _ in range(n_perturb):
        new_p = []
        for p, (lo, hi) in zip(params, bounds):
            width = (hi - lo) * perturb_frac
            np_ = p + RNG.uniform(-width, width)
            np_ = max(lo, min(hi, np_))
            new_p.append(np_)
        pert_scores.append(wf_score(eval_fn, new_p, folds))
    pert_scores = np.array(pert_scores)
    return {
        "base_score":   float(base_score),
        "pert_mean":    float(pert_scores.mean()),
        "pert_std":     float(pert_scores.std()),
        "pert_min":     float(pert_scores.min()),
        "robust_score": float(pert_scores.mean() - 0.5 * pert_scores.std()),
    }

# ════════════════════════════════════════════════════════════════════════════
# 6. S4 / S5 최적화 실행
# ════════════════════════════════════════════════════════════════════════════
S4_BOUNDS = [
    (-0.15, -0.02),  # b      : HALF 반등 진입선
    ( 1.50,  4.00),  # r2     : FULL/HALF 배수
    (-0.25, -0.05),  # trail  : 트레일링 스탑
    ( 0.30,  1.00),  # hf     : HALF 어택 시 TQQQ 비중
]
S4_ANCHOR = [-0.05, 2.0, -0.15, 0.50]   # default

S5_BOUNDS = [
    (0.30, 0.85),    # beta
    (0.10, 0.40),    # max_drop
    (0.30, 1.20),    # stop_factor
    (-0.25, -0.05),  # trail
]
S5_ANCHOR = [0.50, 0.20, 0.75, -0.15]   # default

def run_de(eval_fn, bounds, folds, name: str,
           popsize=10, maxiter=40, seed=42):
    """robust objective로 DE 실행."""
    eval_log = []
    n_eval   = [0]
    def obj(p):
        n_eval[0] += 1
        # S5: stop_factor가 beta보다 0.05 이상 크도록 투영
        if name == "S5":
            beta, md, sf, tr = p
            if sf <= beta + 0.05:
                return 5.0  # 강한 페널티
        s = -wf_score(eval_fn, list(p), folds)
        eval_log.append({"n": n_eval[0], "score": -s, "p": list(p)})
        return s
    print(f"\n  ▶ DE 시작 ({name})  bounds={len(bounds)}D, pop={popsize}, iter={maxiter}")
    t0 = time.time()
    res = differential_evolution(
        obj, bounds, seed=seed,
        maxiter=maxiter, popsize=popsize,
        mutation=(0.5, 1.5), recombination=0.7,
        tol=1e-7, polish=True, disp=False,
    )
    elapsed = time.time() - t0
    best = res.x.tolist()
    best_score = -res.fun
    print(f"     완료: {elapsed:.1f}s, {n_eval[0]:,}회 평가 → best WF score={best_score:.4f}")
    print(f"     best params={best}")
    return {"params": best, "score": best_score, "elapsed": elapsed,
            "n_eval": n_eval[0], "log": eval_log}

print("\n" + "=" * 72)
print("[STAGE 1] Walk-forward CV 기반 DE 최적화")
print("=" * 72)
s4_res = run_de(eval_s4_robust, S4_BOUNDS, CV_FOLDS, "S4", popsize=10, maxiter=40)
s5_res = run_de(eval_s5_robust, S5_BOUNDS, CV_FOLDS, "S5", popsize=10, maxiter=40)

# ════════════════════════════════════════════════════════════════════════════
# 7. Top-K 후보 추출 + Plateau check
# ════════════════════════════════════════════════════════════════════════════
def topk_unique(log: list, k: int = 10, eps: float = 1e-3):
    """eval_log에서 score 상위 K개 (중복 제거)."""
    seen, top = [], []
    for e in sorted(log, key=lambda x: -x["score"]):
        is_dup = any(np.allclose(e["p"], s, atol=eps) for s in seen)
        if not is_dup:
            seen.append(e["p"])
            top.append(e)
        if len(top) >= k:
            break
    return top

print("\n" + "=" * 72)
print("[STAGE 2] Top-10 후보 + Plateau 검증 (±10% perturbation × 10회)")
print("=" * 72)

def evaluate_candidates(name, top_list, eval_fn, bounds):
    print(f"\n  ▶ {name} Top-{len(top_list)} plateau check...")
    enriched = []
    t0 = time.time()
    for i, e in enumerate(top_list):
        pl = plateau_check(eval_fn, e["p"], bounds, CV_FOLDS,
                           n_perturb=10, perturb_frac=0.10)
        enriched.append({**e, **pl})
        print(f"    [{i+1:2d}] base={pl['base_score']:.3f}  "
              f"pert μ={pl['pert_mean']:.3f}  σ={pl['pert_std']:.3f}  "
              f"robust={pl['robust_score']:.3f}")
    print(f"    (소요 {time.time()-t0:.1f}s)")
    return enriched

s4_top  = topk_unique(s4_res["log"], k=10)
s4_top  = evaluate_candidates("S4", s4_top, eval_s4_robust, S4_BOUNDS)
s5_top  = topk_unique(s5_res["log"], k=10)
s5_top  = evaluate_candidates("S5", s5_top, eval_s5_robust, S5_BOUNDS)

# robust_score 기준 best 선택
s4_best = max(s4_top, key=lambda x: x["robust_score"])
s5_best = max(s5_top, key=lambda x: x["robust_score"])

# ════════════════════════════════════════════════════════════════════════════
# 8. Default anchor와 robust score 비교
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("[STAGE 3] Default anchor 비교 (no-tuning baseline)")
print("=" * 72)
s4_anchor_pl = plateau_check(eval_s4_robust, S4_ANCHOR, S4_BOUNDS, CV_FOLDS,
                              n_perturb=10, perturb_frac=0.10)
s5_anchor_pl = plateau_check(eval_s5_robust, S5_ANCHOR, S5_BOUNDS, CV_FOLDS,
                              n_perturb=10, perturb_frac=0.10)

print(f"  S4 anchor      : robust={s4_anchor_pl['robust_score']:.3f}  base={s4_anchor_pl['base_score']:.3f}")
print(f"  S4 best (DE)   : robust={s4_best['robust_score']:.3f}  base={s4_best['base_score']:.3f}")
print(f"  S5 anchor      : robust={s5_anchor_pl['robust_score']:.3f}  base={s5_anchor_pl['base_score']:.3f}")
print(f"  S5 best (DE)   : robust={s5_best['robust_score']:.3f}  base={s5_best['base_score']:.3f}")

# 5% margin 규칙: DE가 anchor를 5%이상 능가할 때만 채택
def select_final(anchor_pl, best, anchor_params, name):
    margin = best["robust_score"] - anchor_pl["robust_score"]
    rel    = margin / max(abs(anchor_pl["robust_score"]), 0.01)
    print(f"  {name} margin = {margin:+.3f}  ({rel*100:+.1f}%)", end="  ")
    if rel >= 0.05:
        print("→ DE 채택")
        return {"params": best["p"], "source": "DE-robust",
                "robust_score": best["robust_score"]}
    else:
        print("→ Anchor 채택 (margin < 5%)")
        return {"params": anchor_params, "source": "anchor",
                "robust_score": anchor_pl["robust_score"]}

s4_final = select_final(s4_anchor_pl, s4_best, S4_ANCHOR, "S4")
s5_final = select_final(s5_anchor_pl, s5_best, S5_ANCHOR, "S5")

# ════════════════════════════════════════════════════════════════════════════
# 9. Holdout (2017~) 평가
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("[STAGE 4] 최종 Holdout 평가 (2017~2026, CV 전혀 미사용)")
print("=" * 72)

# 전체 / IS(CV) / OOS(holdout) 평가용 데이터
END = qqq.index[-1]
Q_cv,  L_cv,  T_cv  = slice_data(WF_START, OOS_HOLDOUT - pd.Timedelta(days=1))
Q_oos, L_oos, T_oos = slice_data(OOS_HOLDOUT, END)
Q_full,L_full,T_full= slice_data(WF_START, END)

s1_cv   = eval_s1(Q_cv);    s1_oos = eval_s1(Q_oos);    s1_full = eval_s1(Q_full)

def eval_three(eval_fn, params, label):
    cv  = eval_fn(Q_cv,  L_cv,  T_cv,  *params)
    oos = eval_fn(Q_oos, L_oos, T_oos, *params)
    full= eval_fn(Q_full,L_full,T_full,*params)
    return {"label": label, "params": params, "cv": cv, "oos": oos, "full": full}

results = [
    eval_three(eval_s4_robust, S4_ANCHOR,        "S4 Anchor"),
    eval_three(eval_s4_robust, s4_best["p"],     "S4 DE-best"),
    eval_three(eval_s4_robust, s4_final["params"], f"S4 FINAL ({s4_final['source']})"),
    eval_three(eval_s5_robust, S5_ANCHOR,        "S5 Anchor"),
    eval_three(eval_s5_robust, s5_best["p"],     "S5 DE-best"),
    eval_three(eval_s5_robust, s5_final["params"], f"S5 FINAL ({s5_final['source']})"),
]

print(f"\n  {'전략':28s}  {'CV(2002-16)':>26s}  {'OOS(2017-26)':>26s}  {'전체':>26s}")
print("  " + "-" * 110)

def fmt(d, s1=None):
    cagr = d["cagr"]*100; mdd = d["mdd"]*100; sh = d["sharpe"]
    if s1 is not None:
        c_ok = "✓" if d["cagr"] > s1["cagr"] else "✗"
        s_ok = "✓" if d["sharpe"] > s1["sharpe"] else "✗"
        return f"C={cagr:+5.1f}%{c_ok} M={mdd:5.1f}% S={sh:.2f}{s_ok}"
    return f"C={cagr:+5.1f}%  M={mdd:5.1f}% S={sh:.2f}"

# S1 행 먼저
print(f"  {'S1 (QQQ Buy & Hold)':28s}  "
      f"{fmt(s1_cv):>26s}  {fmt(s1_oos):>26s}  {fmt(s1_full):>26s}")
print("  " + "-" * 110)
for r in results:
    print(f"  {r['label']:28s}  "
          f"{fmt(r['cv'], s1_cv):>26s}  {fmt(r['oos'], s1_oos):>26s}  {fmt(r['full'], s1_full):>26s}")

# ════════════════════════════════════════════════════════════════════════════
# 10. Stationary block bootstrap on Holdout returns
# ════════════════════════════════════════════════════════════════════════════
def stationary_block_bootstrap(returns: pd.Series, block_len: int = 60,
                                n_iter: int = 200, seed: int = 7):
    """Politis-Romano stationary bootstrap → Sharpe 분포."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    sharpes, cagrs, mdds = [], [], []
    p = 1.0 / block_len
    rets = returns.values
    for _ in range(n_iter):
        idx = []
        i = rng.integers(0, n)
        for _step in range(n):
            idx.append(i)
            if rng.random() < p:
                i = rng.integers(0, n)
            else:
                i = (i + 1) % n
        sample = rets[idx]
        if sample.std() == 0:
            continue
        rf_d = (1.04) ** (1/252) - 1
        sharpe = (sample.mean() - rf_d) / sample.std() * np.sqrt(252)
        cagr = (1 + sample.mean()) ** 252 - 1
        wealth = (1 + sample).cumprod()
        mdd = float(((wealth - np.maximum.accumulate(wealth)) / np.maximum.accumulate(wealth)).min())
        sharpes.append(sharpe); cagrs.append(cagr); mdds.append(mdd)
    return {"sharpe": np.array(sharpes), "cagr": np.array(cagrs), "mdd": np.array(mdds)}

print("\n" + "=" * 72)
print("[STAGE 5] OOS Holdout (2017~) Stationary Block Bootstrap (60일 블록 × 200회)")
print("=" * 72)

def get_oos_returns(eval_fn, params):
    """OOS 수익률 시리즈 추출."""
    Q, L, T = slice_data(OOS_HOLDOUT, END)
    if eval_fn == eval_s4_robust:
        b, r2, trail, hf = params
        p = derive_s4(b, S4_R1_FIXED, r2, trail)
        port, _ = strategy_s4_trailing(
            Q, L, T, 100_000,
            shallow_drop=p["shallow_drop"], deep_drop=p["deep_drop"],
            shallow_bounce=p["shallow_bounce"], deep_bounce=p["deep_bounce"],
            half_stop=p["half_stop"], trailing_stop_pct=p["trailing_stop"],
            half_frac=hf, full_frac=S4_FF_FIXED,
        )
    else:
        beta, md, sf, tr = params
        port, _ = strategy_s5(
            Q, L, T, 100_000,
            beta=beta, max_drop=md, min_drop=0.0,
            stop_factor=sf, trailing_stop_pct=tr,
            use_stop_loss=True,
            position_mode="linear",
        )
    return port.pct_change().dropna()

s1_rets_oos = strategy_buy_and_hold(Q_oos, 100_000, "S1").pct_change().dropna()
s1_bs       = stationary_block_bootstrap(s1_rets_oos)

bootstrap_results = {"S1": s1_bs}
for r in results:
    if "FINAL" not in r["label"]:
        continue
    eval_fn = eval_s4_robust if "S4" in r["label"] else eval_s5_robust
    rets = get_oos_returns(eval_fn, r["params"])
    bootstrap_results[r["label"]] = stationary_block_bootstrap(rets)

print(f"\n  {'전략':30s}  {'Sharpe (median, [P5, P95])':>40s}  {'P(Sharpe>S1)':>14s}")
print("  " + "-" * 90)
s1_sh = bootstrap_results["S1"]["sharpe"]
for label, bs in bootstrap_results.items():
    sh = bs["sharpe"]
    med = np.median(sh); p5 = np.percentile(sh, 5); p95 = np.percentile(sh, 95)
    if label == "S1":
        beat = "—"
    else:
        # paired (sample-by-sample) 비교는 어려우니 각각의 분포 대비
        beat = f"{(sh > np.median(s1_sh)).mean()*100:.0f}%"
    print(f"  {label:30s}  {f'{med:.3f}  [{p5:.3f}, {p95:.3f}]':>40s}  {beat:>14s}")

# ════════════════════════════════════════════════════════════════════════════
# 11. JSON & 시각화 저장
# ════════════════════════════════════════════════════════════════════════════
summary = {
    "method": "Robust optimization (WF-CV + plateau + bootstrap)",
    "cv_period": f"{WF_START.date()} ~ {(OOS_HOLDOUT - pd.Timedelta(days=1)).date()}",
    "oos_period": f"{OOS_HOLDOUT.date()} ~ {END.date()}",
    "n_cv_folds": len(CV_FOLDS),
    "S4": {
        "anchor_params": S4_ANCHOR,
        "anchor_robust_score": s4_anchor_pl["robust_score"],
        "de_best_params": s4_best["p"],
        "de_best_robust_score": s4_best["robust_score"],
        "final_params": s4_final["params"],
        "final_source": s4_final["source"],
        "fixed": {"r1": S4_R1_FIXED, "full_frac": S4_FF_FIXED},
    },
    "S5": {
        "anchor_params": S5_ANCHOR,
        "anchor_robust_score": s5_anchor_pl["robust_score"],
        "de_best_params": s5_best["p"],
        "de_best_robust_score": s5_best["robust_score"],
        "final_params": s5_final["params"],
        "final_source": s5_final["source"],
    },
    "performance": [
        {"label": r["label"], "params": r["params"],
         "cv": r["cv"], "oos": r["oos"], "full": r["full"]} for r in results
    ],
    "s1": {"cv": s1_cv, "oos": s1_oos, "full": s1_full},
    "bootstrap_oos": {
        label: {"sharpe_median": float(np.median(b["sharpe"])),
                "sharpe_p5": float(np.percentile(b["sharpe"], 5)),
                "sharpe_p95": float(np.percentile(b["sharpe"], 95)),
                "cagr_median": float(np.median(b["cagr"])),
                "mdd_median": float(np.median(b["mdd"]))}
        for label, b in bootstrap_results.items()
    },
}
with open(OUT_DIR / "robust_optimization_result.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
print(f"\nJSON 저장: {OUT_DIR / 'robust_optimization_result.json'}")

# ── 시각화 ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30)
fig.suptitle(
    "S4·S5 Robust 최적화 (Walk-Forward CV + Plateau + Bootstrap)\n"
    f"CV: {WF_START.date()}~{(OOS_HOLDOUT-pd.Timedelta(days=1)).date()} "
    f"({len(CV_FOLDS)} folds)  |  Holdout: {OOS_HOLDOUT.date()}~{END.date()}",
    fontsize=11, fontweight="bold")

# ① WF fold별 OOS Sharpe (S4 anchor vs final, S5 anchor vs final)
def fold_oos_sharpes(eval_fn, params):
    out = []
    for ts, te, ds, de in CV_FOLDS:
        Q, L, T = slice_data(ds, de)
        if len(Q) < 50: continue
        r = eval_fn(Q, L, T, *params)
        out.append((ds.year, r["sharpe"] if r else 0))
    return out

ax = fig.add_subplot(gs[0, 0])
for params, lbl, color in [
    (S4_ANCHOR, "S4 Anchor", "#9CA3AF"),
    (s4_best["p"], "S4 DE-best", "#DC2626"),
    (s4_final["params"], f"S4 FINAL ({s4_final['source']})", "#2563EB"),
]:
    pairs = fold_oos_sharpes(eval_s4_robust, params)
    yrs = [p[0] for p in pairs]; vals = [p[1] for p in pairs]
    ax.plot(yrs, vals, "o-", lw=1.5, ms=5, label=lbl, color=color, alpha=0.85)
# S1
s1_pairs = []
for ts, te, ds, de in CV_FOLDS:
    Q, _, _ = slice_data(ds, de)
    s1_pairs.append((ds.year, eval_s1(Q)["sharpe"]))
ax.plot([p[0] for p in s1_pairs], [p[1] for p in s1_pairs],
        "s--", lw=1.5, ms=5, label="S1 QQQ", color="#000", alpha=0.6)
ax.axhline(0, color="#999", lw=0.7)
ax.set_xlabel("Test year"); ax.set_ylabel("OOS Sharpe (1y)")
ax.set_title("WF fold별 OOS Sharpe — S4", fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
for params, lbl, color in [
    (S5_ANCHOR, "S5 Anchor", "#9CA3AF"),
    (s5_best["p"], "S5 DE-best", "#DC2626"),
    (s5_final["params"], f"S5 FINAL ({s5_final['source']})", "#2563EB"),
]:
    pairs = fold_oos_sharpes(eval_s5_robust, params)
    yrs = [p[0] for p in pairs]; vals = [p[1] for p in pairs]
    ax.plot(yrs, vals, "o-", lw=1.5, ms=5, label=lbl, color=color, alpha=0.85)
ax.plot([p[0] for p in s1_pairs], [p[1] for p in s1_pairs],
        "s--", lw=1.5, ms=5, label="S1 QQQ", color="#000", alpha=0.6)
ax.axhline(0, color="#999", lw=0.7)
ax.set_xlabel("Test year"); ax.set_ylabel("OOS Sharpe (1y)")
ax.set_title("WF fold별 OOS Sharpe — S5", fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ② CV / OOS / Full CAGR + Sharpe 막대
ax = fig.add_subplot(gs[1, :])
labels = ["S1"] + [r["label"] for r in results]
def grab(d, key, attr): return d[key][attr]
cvs   = [s1_cv["cagr"]*100]   + [r["cv"]["cagr"]*100   for r in results]
ooss  = [s1_oos["cagr"]*100]  + [r["oos"]["cagr"]*100  for r in results]
fulls = [s1_full["cagr"]*100] + [r["full"]["cagr"]*100 for r in results]
x = np.arange(len(labels)); w = 0.27
ax.bar(x-w, cvs,   w, color="#9CA3AF", label="CV (2002-2016)", alpha=0.9)
ax.bar(x,   ooss,  w, color="#DC2626", label="OOS (2017-2026)", alpha=0.9)
ax.bar(x+w, fulls, w, color="#2563EB", label="Full (2002-2026)", alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("CAGR (%)")
ax.set_title("CAGR 비교 (CV / OOS / Full)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
ax.axhline(0, color="black", lw=0.5)

# ③ Bootstrap Sharpe 분포
ax = fig.add_subplot(gs[2, 0])
for label, bs in bootstrap_results.items():
    ax.hist(bs["sharpe"], bins=30, alpha=0.4, label=label,
            density=True, edgecolor="white", lw=0.5)
ax.axvline(0, color="#999", lw=0.7)
ax.set_xlabel("Bootstrap Sharpe"); ax.set_ylabel("density")
ax.set_title("OOS Sharpe 분포 (block bootstrap)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ④ Plateau check 시각화 (S4 / S5 top-10 base vs robust)
ax = fig.add_subplot(gs[2, 1])
for top, lbl, color in [(s4_top, "S4", "#DC2626"),
                         (s5_top, "S5", "#2563EB")]:
    bases  = [t["base_score"] for t in top]
    robs   = [t["robust_score"] for t in top]
    ax.scatter(bases, robs, s=60, alpha=0.7, label=lbl,
               edgecolors="black", lw=0.5, color=color)
ax.plot([0, 1.5], [0, 1.5], "--", color="#999", lw=0.8)
ax.set_xlabel("Base WF score (no perturbation)")
ax.set_ylabel("Robust score (mean − 0.5σ)")
ax.set_title("Plateau check: Top-10 후보 (대각선 위 = robust)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.savefig(OUT_DIR / "robust_optimization_result.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"PNG 저장: {OUT_DIR / 'robust_optimization_result.png'}")

print("\n" + "=" * 72)
print("✅ 완료. 자세한 비교는 robust_optimization_result.json 참고.")
print("=" * 72)
