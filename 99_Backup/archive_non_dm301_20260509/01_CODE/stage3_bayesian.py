"""
stage3_bayesian.py
==================
[단계 3] Bayesian parameter update — anchor를 prior로, 데이터로 weak posterior.

수학적 프레임워크
-----------------
각 파라미터 θ_i 에 대해:
    prior:     θ_i ~ N(anchor_i, σ_prior_i²)        (강한 prior, σ 작음)
    likelihood: WF Sharpe(θ) = local Gaussian fit around peak
    posterior: θ_post_i = (σ_data²·anchor_i + σ_prior_i²·θ_MLE) / (σ_data² + σ_prior_i²)

핵심
----
• prior를 데이터보다 5-10× 강하게 → 극단값으로 가지 못함
• MLE는 "각 fold의 best score 파라미터" 평균으로 근사
• posterior는 자동으로 anchor 근처 ±20% 안에 머무름
• 시간이 갈수록 데이터 weight 미세 증가 가능

비교 대상
---------
• Anchor (no update)
• MLE (data only — 기존 DE 결과)
• Bayesian Posterior (anchor + data, prior dominant)
"""

import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Apple SD Gothic Neo"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from backtest_switching import (
    load_extended_daily, strategy_buy_and_hold, strategy_s4_trailing,
)
from evaluation_metrics import full_metrics, paired_bootstrap_compare

OUT_DIR = Path("03_RESULT/sensitivity")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(7)

# ── 데이터 ───────────────────────────────────────────────────────────────────
print("=" * 78)
print("[STAGE 3] Bayesian Parameter Update")
print("=" * 78)

qqq  = load_extended_daily("QQQ")
qld  = load_extended_daily("QLD")
tqqq = load_extended_daily("TQQQ")
common = qqq.index.intersection(qld.index).intersection(tqqq.index)
qqq, qld, tqqq = qqq.loc[common], qld.loc[common], tqqq.loc[common]

CV_START  = pd.Timestamp("2002-01-01")
OOS_START = pd.Timestamp("2017-01-01")
END = qqq.index[-1]

def slice_data(start, end):
    m = (qqq.index >= start) & (qqq.index <= end)
    return qqq[m], qld[m], tqqq[m]

# ── S4 평가 ─────────────────────────────────────────────────────────────────
S4_R1 = 2.0
S4_FF = 1.0

def eval_s4(Q, L, T, b, r2, trail, hf):
    sd, db = b*S4_R1, b*r2
    dd, hs = b*S4_R1*r2, (sd + b*S4_R1*r2)/2
    try:
        port, _ = strategy_s4_trailing(Q, L, T, 100_000,
            shallow_drop=sd, deep_drop=dd, shallow_bounce=b, deep_bounce=db,
            half_stop=hs, trailing_stop_pct=trail,
            half_frac=hf, full_frac=S4_FF)
        return full_metrics(port)
    except Exception:
        return None

# ── Anchor (prior 중심) ──────────────────────────────────────────────────────
ANCHOR = {"b": -0.05, "r2": 2.0, "trail": -0.15, "hf": 0.50}

# Prior 표준편차 (anchor 폭의 약 25%)
PRIOR_SIGMA = {
    "b":     0.015,   # ±1.5%p around -5%
    "r2":    0.5,     # ±0.5 around 2.0
    "trail": 0.025,   # ±2.5%p around -15%
    "hf":    0.15,    # ±0.15 around 0.50
}

# Bound (hard limit; posterior 가 이 밖이면 clip)
BOUNDS = {
    "b":     (-0.15, -0.02),
    "r2":    (1.2,    4.0),
    "trail": (-0.25, -0.04),
    "hf":    (0.20,   1.00),
}

# ── Step 1: Walk-forward fold별 local MLE 추정 ────────────────────────────────
print("\n[Step 1] WF fold별 local MLE 파라미터 추정 (각 1년 test 윈도우)")

# CV 기간 안에서 walk-forward fold (5y train + 1y test, 1y step)
# 단, train window는 사용하지 않고 test 1y에서 직접 best 파라미터 찾음
TRAIN_YEARS = 5
TEST_YEARS  = 1
folds = []
cur = CV_START
while cur + pd.DateOffset(years=TRAIN_YEARS+TEST_YEARS) <= OOS_START:
    train_e = cur + pd.DateOffset(years=TRAIN_YEARS) - pd.Timedelta(days=1)
    test_s = train_e + pd.Timedelta(days=1)
    test_e = test_s + pd.DateOffset(years=TEST_YEARS) - pd.Timedelta(days=1)
    folds.append((cur, train_e, test_s, test_e))
    cur += pd.DateOffset(years=TEST_YEARS)
print(f"  {len(folds)} folds  ({folds[0][2].year} ~ {folds[-1][3].year})")

# 각 fold에서 anchor 주변 random search → best 파라미터 (local MLE)
def local_mle(test_start, test_end, n_search=80):
    """Anchor 주변 ±2σ에서 random search, Sharpe 최대화 파라미터."""
    Q, L, T = slice_data(test_start, test_end)
    if len(Q) < 50:
        return None
    best_sh = -np.inf
    best_p = None
    for _ in range(n_search):
        b   = ANCHOR["b"]   + RNG.normal(0, 2*PRIOR_SIGMA["b"])
        r2  = ANCHOR["r2"]  + RNG.normal(0, 2*PRIOR_SIGMA["r2"])
        tr  = ANCHOR["trail"] + RNG.normal(0, 2*PRIOR_SIGMA["trail"])
        hf  = ANCHOR["hf"]  + RNG.normal(0, 2*PRIOR_SIGMA["hf"])
        # bound clip
        b   = float(np.clip(b,  BOUNDS["b"][0],  BOUNDS["b"][1]))
        r2  = float(np.clip(r2, BOUNDS["r2"][0], BOUNDS["r2"][1]))
        tr  = float(np.clip(tr, BOUNDS["trail"][0], BOUNDS["trail"][1]))
        hf  = float(np.clip(hf, BOUNDS["hf"][0], BOUNDS["hf"][1]))
        r = eval_s4(Q, L, T, b, r2, tr, hf)
        if r and r["sharpe"] > best_sh:
            best_sh = r["sharpe"]
            best_p = {"b": b, "r2": r2, "trail": tr, "hf": hf, "sharpe": r["sharpe"]}
    return best_p

t0 = time.time()
mle_per_fold = []
for tr_s, tr_e, te_s, te_e in folds:
    mle = local_mle(te_s, te_e, n_search=60)
    if mle:
        mle_per_fold.append({"test_year": te_s.year, **mle})
        print(f"  [test {te_s.year}]  MLE: b={mle['b']*100:>+5.1f}%  r2={mle['r2']:.2f}  "
              f"trail={mle['trail']*100:>+5.1f}%  hf={mle['hf']:.2f}  Sh={mle['sharpe']:.2f}")
print(f"  ({time.time()-t0:.1f}s)")

# ── Step 2: Bayesian posterior ──────────────────────────────────────────────
print("\n[Step 2] Bayesian posterior 계산 (Conjugate normal)")
# MLE 평균과 분산 (cross-fold)
mle_df = pd.DataFrame(mle_per_fold)

# σ_data: cross-fold std / sqrt(n_folds)  → MLE 평균의 추정 오차
posterior = {}
for k in ["b", "r2", "trail", "hf"]:
    mle_mean = mle_df[k].mean()
    mle_std  = mle_df[k].std()
    sigma_data = mle_std / np.sqrt(len(mle_df))   # SE of mean
    sigma_prior = PRIOR_SIGMA[k]
    # Conjugate posterior mean (precision-weighted)
    prec_data  = 1.0 / max(sigma_data**2, 1e-9)
    prec_prior = 1.0 / sigma_prior**2
    post_mean  = (prec_prior * ANCHOR[k] + prec_data * mle_mean) / (prec_prior + prec_data)
    post_var   = 1.0 / (prec_prior + prec_data)
    # Hard bound clip
    post_mean = float(np.clip(post_mean, BOUNDS[k][0], BOUNDS[k][1]))
    posterior[k] = {
        "anchor": ANCHOR[k],
        "mle_mean": float(mle_mean),
        "mle_std": float(mle_std),
        "sigma_prior": sigma_prior,
        "sigma_data_of_mean": float(sigma_data),
        "post_mean": post_mean,
        "post_std": float(np.sqrt(post_var)),
        "shrinkage": float(prec_prior / (prec_prior + prec_data)),  # 1=full anchor, 0=full MLE
    }

print(f"\n  {'param':>8s}  {'anchor':>9s}  {'MLE mean':>9s}  {'σ_prior':>8s}  {'σ_data':>8s}  "
      f"{'shrinkage':>10s}  {'POSTERIOR':>10s}")
print("  " + "-" * 80)
for k, p in posterior.items():
    fmt = "{:>8.3f}"
    print(f"  {k:>8s}  {fmt.format(p['anchor']):>9s}  {fmt.format(p['mle_mean']):>9s}  "
          f"{fmt.format(p['sigma_prior']):>8s}  {fmt.format(p['sigma_data_of_mean']):>8s}  "
          f"{p['shrinkage']*100:>9.1f}%  {fmt.format(p['post_mean']):>10s}")
print("  ※ shrinkage = prior precision share. 1.0 = anchor 그대로, 0.0 = MLE 그대로")

POSTERIOR_PARAMS = {k: posterior[k]["post_mean"] for k in ["b", "r2", "trail", "hf"]}
MLE_AVG_PARAMS   = {k: posterior[k]["mle_mean"]  for k in ["b", "r2", "trail", "hf"]}

# ── Step 3: Anchor / MLE / Posterior 비교 백테스트 ─────────────────────────────
print("\n[Step 3] Anchor / MLE-avg / Posterior backtest 비교")
Q_cv, L_cv, T_cv = slice_data(CV_START, OOS_START - pd.Timedelta(days=1))
Q_oos, L_oos, T_oos = slice_data(OOS_START, END)
Q_full, L_full, T_full = slice_data(CV_START, END)

def run_s4(Q, L, T, p):
    sd, db = p["b"]*S4_R1, p["b"]*p["r2"]
    dd, hs = p["b"]*S4_R1*p["r2"], (sd + p["b"]*S4_R1*p["r2"])/2
    port, _ = strategy_s4_trailing(Q, L, T, 100_000,
        shallow_drop=sd, deep_drop=dd, shallow_bounce=p["b"], deep_bounce=db,
        half_stop=hs, trailing_stop_pct=p["trail"],
        half_frac=p["hf"], full_frac=S4_FF)
    return port

s1_cv   = strategy_buy_and_hold(Q_cv,   100_000)
s1_oos  = strategy_buy_and_hold(Q_oos,  100_000)
s1_full = strategy_buy_and_hold(Q_full, 100_000)

candidates = [
    ("Anchor",     ANCHOR),
    ("MLE-avg",    MLE_AVG_PARAMS),
    ("Posterior",  POSTERIOR_PARAMS),
]
backtests = {}
for name, p in candidates:
    backtests[name] = {
        "params": p,
        "cv":   run_s4(Q_cv,   L_cv,   T_cv,   p),
        "oos":  run_s4(Q_oos,  L_oos,  T_oos,  p),
        "full": run_s4(Q_full, L_full, T_full, p),
    }

print(f"\n  {'전략':14s}  {'b':>7s}  {'r2':>5s}  {'trail':>7s}  {'hf':>5s}  "
      f"{'CV CAGR':>9s}  {'OOS CAGR':>9s}  {'OOS Sh':>7s}  {'OOS MDD':>9s}  {'OOS Ulcer':>10s}")
print("  " + "-" * 110)
print(f"  {'S1 (QQQ)':14s}  {'-':>7s}  {'-':>5s}  {'-':>7s}  {'-':>5s}  "
      f"{full_metrics(s1_cv)['cagr']*100:>+8.2f}%  "
      f"{full_metrics(s1_oos)['cagr']*100:>+8.2f}%  "
      f"{full_metrics(s1_oos)['sharpe']:>7.2f}  "
      f"{full_metrics(s1_oos)['mdd']*100:>+8.2f}%  "
      f"{full_metrics(s1_oos)['ulcer']:>10.2f}")
print("  " + "-" * 110)
for name, _ in candidates:
    p = backtests[name]["params"]
    m_cv  = full_metrics(backtests[name]["cv"])
    m_oos = full_metrics(backtests[name]["oos"])
    print(f"  {name:14s}  {p['b']*100:>+6.2f}%  {p['r2']:>5.2f}  {p['trail']*100:>+6.2f}%  {p['hf']:>5.2f}  "
          f"{m_cv['cagr']*100:>+8.2f}%  "
          f"{m_oos['cagr']*100:>+8.2f}%  "
          f"{m_oos['sharpe']:>7.2f}  "
          f"{m_oos['mdd']*100:>+8.2f}%  "
          f"{m_oos['ulcer']:>10.2f}")

# ── Bootstrap vs S1 ──────────────────────────────────────────────────────────
print("\n[Step 4] Paired Bootstrap (OOS vs S1, 60-day blocks × 500)")
s1_rets = s1_oos.pct_change().dropna()
bs_results = {}
for name, _ in candidates:
    rets = backtests[name]["oos"].pct_change().dropna()
    bs_results[name] = paired_bootstrap_compare(rets, s1_rets, 60, 500)

def show_bs(label, bs):
    def cell(key, fmt="{:+.3f}"):
        x = bs[key]; sig = "★" if x["p_value"]<0.05 else ("·" if x["p_value"]<0.20 else " ")
        return f"{fmt.format(x['median'])}{sig} ({x['prob_better']*100:>3.0f}%)"
    print(f"  {label:14s}  ΔSharpe={cell('delta_sharpe'):>20s}  "
          f"ΔCAGR={cell('delta_cagr','{:+.1%}'):>22s}  "
          f"ΔSortino={cell('delta_sortino'):>20s}  "
          f"ΔUlcer={cell('delta_ulcer'):>20s}")

for name, _ in candidates:
    show_bs(name, bs_results[name])

# ── 시각화 ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 12))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.30)
fig.suptitle("S4 Bayesian Parameter Update (anchor prior + WF MLE)",
             fontsize=12, fontweight="bold")

# ① Cross-fold MLE 추적 (각 파라미터 추이)
ax = fig.add_subplot(gs[0, :])
years = mle_df["test_year"]
for k, color in [("b", "#DC2626"), ("r2", "#2563EB"),
                  ("trail", "#10B981"), ("hf", "#F59E0B")]:
    # normalise so 0=anchor, 1 σ_prior
    norm_vals = (mle_df[k] - ANCHOR[k]) / PRIOR_SIGMA[k]
    ax.plot(years, norm_vals, "o-", lw=1.6, ms=6, label=k, color=color, alpha=0.85)
ax.axhline(0, color="black", ls="--", lw=0.8, label="anchor")
ax.axhline(1, color="#999", ls=":", lw=0.6); ax.axhline(-1, color="#999", ls=":", lw=0.6)
ax.set_xlabel("Test year"); ax.set_ylabel("Normalised offset (in σ_prior)")
ax.set_title("Fold별 local MLE — anchor 대비 정규화 거리 (±1σ_prior 음영)",
             fontsize=10, fontweight="bold")
ax.fill_between(years, -1, 1, color="#10B981", alpha=0.07)
ax.legend(fontsize=8, ncol=5); ax.grid(True, alpha=0.3)

# ② Posterior shrinkage bar
ax = fig.add_subplot(gs[1, 0])
keys = list(posterior.keys())
shrinks = [posterior[k]["shrinkage"]*100 for k in keys]
colors = ["#DC2626" if s>50 else "#2563EB" for s in shrinks]
ax.barh(keys, shrinks, color=colors, edgecolor="black", lw=0.5)
ax.axvline(50, color="#999", ls="--", lw=0.8)
for i, s in enumerate(shrinks):
    ax.text(s+1, i, f"{s:.1f}%", va="center", fontsize=10)
ax.set_xlabel("Prior precision share (%)  — 100% = anchor 그대로")
ax.set_title("Bayesian shrinkage (각 파라미터)", fontsize=10, fontweight="bold")
ax.set_xlim(0, 110); ax.grid(True, alpha=0.3, axis="x")

# ③ Anchor vs Posterior vs MLE-avg 시각화
ax = fig.add_subplot(gs[1, 1])
keys = list(ANCHOR.keys())
x = np.arange(len(keys)); w = 0.27
def norm(p):
    return [(p[k] - ANCHOR[k]) / PRIOR_SIGMA[k] for k in keys]
ax.bar(x - w, norm(ANCHOR),       w, label="Anchor",    color="#9CA3AF", alpha=0.9)
ax.bar(x,     norm(MLE_AVG_PARAMS), w, label="MLE-avg",   color="#DC2626", alpha=0.9)
ax.bar(x + w, norm(POSTERIOR_PARAMS), w, label="Posterior", color="#2563EB", alpha=0.9)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=10)
ax.set_ylabel("Offset from anchor (in σ_prior)")
ax.set_title("Anchor vs MLE-avg vs Posterior", fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

# ④ OOS 자산곡선 비교
ax = fig.add_subplot(gs[2, :])
ax.plot(s1_oos.index, s1_oos.values / s1_oos.iloc[0] * 100, label="S1 QQQ",
        color="#000", lw=1.4, alpha=0.7)
for name, color in [("Anchor", "#9CA3AF"), ("MLE-avg", "#DC2626"), ("Posterior", "#2563EB")]:
    p = backtests[name]["oos"]
    ax.plot(p.index, p.values / p.iloc[0] * 100, label=f"S4 {name}", color=color,
            lw=1.7 if name == "Posterior" else 1.4)
ax.set_yscale("log"); ax.set_ylabel("Index (start=100, log)")
ax.set_title(f"OOS 자산곡선  ({s1_oos.index[0].date()}~{s1_oos.index[-1].date()})",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.savefig(OUT_DIR / "stage3_bayesian.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── JSON 저장 ────────────────────────────────────────────────────────────────
out = {
    "anchor": ANCHOR,
    "prior_sigma": PRIOR_SIGMA,
    "bounds": {k: list(v) for k, v in BOUNDS.items()},
    "mle_per_fold": mle_per_fold,
    "posterior": posterior,
    "anchor_params": ANCHOR,
    "mle_avg_params": MLE_AVG_PARAMS,
    "posterior_params": POSTERIOR_PARAMS,
    "metrics": {
        name: {
            "cv":   full_metrics(backtests[name]["cv"]),
            "oos":  full_metrics(backtests[name]["oos"]),
            "full": full_metrics(backtests[name]["full"]),
        } for name, _ in candidates
    },
    "s1": {
        "cv":   full_metrics(s1_cv),
        "oos":  full_metrics(s1_oos),
        "full": full_metrics(s1_full),
    },
    "bootstrap_oos_vs_s1": {
        name: {k: v for k, v in bs_results[name].items() if k != "raw"}
        for name in [c[0] for c in candidates]
    },
}

def _to_json(o):
    if isinstance(o, dict): return {k: _to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_to_json(x) for x in o]
    if isinstance(o, (np.floating, np.float64, np.float32)): return float(o)
    if isinstance(o, (np.integer, np.int64)): return int(o)
    return o

with open(OUT_DIR / "stage3_bayesian.json", "w") as f:
    json.dump(_to_json(out), f, indent=2)

print(f"\n  PNG → {OUT_DIR / 'stage3_bayesian.png'}")
print(f"  JSON → {OUT_DIR / 'stage3_bayesian.json'}")
print("\n[STAGE 3 완료]")
