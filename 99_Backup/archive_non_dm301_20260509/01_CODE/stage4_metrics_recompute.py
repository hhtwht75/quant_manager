"""
stage4_metrics_recompute.py
============================
[단계 4] 모든 기존 anchor / DE-best / S1 후보를 새 메트릭으로 재계산.

평가 메트릭 (Sharpe 외)
  • Sortino, Ulcer, Calmar, Pain ratio, UPI
  • Paired bootstrap ΔSharpe / ΔCAGR / ΔSortino / ΔUlcer (vs S1)
  • 통계적 유의성 (★ = p<0.05, · = p<0.20)
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
    load_extended_daily, strategy_buy_and_hold,
    strategy_s4_trailing, strategy_s5,
)
from evaluation_metrics import (
    full_metrics, paired_bootstrap_compare, fmt_metrics_row,
)

OUT_DIR = Path("03_RESULT/sensitivity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 데이터 ───────────────────────────────────────────────────────────────────
print("=" * 78)
print("[STAGE 4] 새 메트릭으로 모든 후보 재계산")
print("=" * 78)
print("데이터 로드...")
qqq  = load_extended_daily("QQQ")
qld  = load_extended_daily("QLD")
tqqq = load_extended_daily("TQQQ")
common = qqq.index.intersection(qld.index).intersection(tqqq.index)
qqq, qld, tqqq = qqq.loc[common], qld.loc[common], tqqq.loc[common]

CV_START = pd.Timestamp("2002-01-01")
OOS_START = pd.Timestamp("2017-01-01")
END = qqq.index[-1]

def slice_data(start, end):
    m = (qqq.index >= start) & (qqq.index <= end)
    return qqq[m], qld[m], tqqq[m]

Q_cv, L_cv, T_cv = slice_data(CV_START, OOS_START - pd.Timedelta(days=1))
Q_oos, L_oos, T_oos = slice_data(OOS_START, END)
Q_full, L_full, T_full = slice_data(CV_START, END)

print(f"  CV  : {Q_cv.index[0].date()} ~ {Q_cv.index[-1].date()}  ({len(Q_cv):,}일)")
print(f"  OOS : {Q_oos.index[0].date()} ~ {Q_oos.index[-1].date()}  ({len(Q_oos):,}일)")
print(f"  Full: {Q_full.index[0].date()} ~ {Q_full.index[-1].date()}  ({len(Q_full):,}일)")

# ── 후보 정의 ────────────────────────────────────────────────────────────────
def make_s4(b, r1, r2, trail, hf, ff):
    return dict(b=b, r1=r1, r2=r2, trail=trail, hf=hf, ff=ff)

def run_s4(Q, L, T, p):
    sd, db = p["b"]*p["r1"], p["b"]*p["r2"]
    dd, hs = p["b"]*p["r1"]*p["r2"], (sd + p["b"]*p["r1"]*p["r2"])/2
    port, _ = strategy_s4_trailing(Q, L, T, 100_000,
        shallow_drop=sd, deep_drop=dd, shallow_bounce=p["b"], deep_bounce=db,
        half_stop=hs, trailing_stop_pct=p["trail"],
        half_frac=p["hf"], full_frac=p["ff"])
    return port

def make_s5(beta, md, sf, trail):
    return dict(beta=beta, md=md, sf=sf, trail=trail)

def run_s5(Q, L, T, p):
    port, _ = strategy_s5(
        Q, L, T, 100_000,
        beta=p["beta"], max_drop=p["md"],
        stop_factor=p["sf"], trailing_stop_pct=p["trail"],
        min_drop=0.0,
        use_stop_loss=True,
        position_mode="linear",
    )
    return port

# 모든 후보
candidates = [
    ("S1 (QQQ B&H)", "S1", None, lambda Q, L, T, p:
        strategy_buy_and_hold(Q, 100_000, "S1")),
    # S4 후보
    ("S4 Anchor",       "S4", make_s4(-0.05,  2.00,  2.00, -0.15,  0.50, 1.00), run_s4),
    ("S4 DE-old (IS=2010-19)", "S4",
        make_s4(-0.10404, 1.0561, 2.7402, -0.0894, 0.8997, 1.0), run_s4),
    ("S4 DE-extended",  "S4",
        make_s4(-0.10422, 1.0542, 1.733, -0.12906, 0.8988, 1.0), run_s4),
    ("S4 DE-robust",    "S4",
        make_s4(-0.05578, 2.0, 1.7571, -0.06283, 0.9810, 1.0), run_s4),
    # S5 후보
    ("S5 Anchor",       "S5", make_s5(0.50, 0.20, 0.75, -0.15), run_s5),
    ("S5 DE (IS=2002-16)", "S5",
        make_s5(0.83289, 0.28823, 0.8836, -0.02278), run_s5),
    ("S5 DE-robust",    "S5", make_s5(0.4345, 0.2096, 0.4890, -0.0795), run_s5),
]

# ── 메트릭 계산 ──────────────────────────────────────────────────────────────
print(f"\n  포트폴리오 구축 + 메트릭 계산 ({len(candidates)} 후보 × 3 기간)...")
t0 = time.time()
results = []
ports_oos = {}  # bootstrap 용
for label, family, params, runner in candidates:
    try:
        p_cv   = runner(Q_cv,   L_cv,   T_cv,   params)
        p_oos  = runner(Q_oos,  L_oos,  T_oos,  params)
        p_full = runner(Q_full, L_full, T_full, params)
    except Exception as e:
        print(f"  ✗ {label}: {e}")
        continue
    ports_oos[label] = p_oos
    results.append({
        "label": label, "family": family, "params": params,
        "cv":   full_metrics(p_cv),
        "oos":  full_metrics(p_oos),
        "full": full_metrics(p_full),
    })
print(f"  완료 ({time.time()-t0:.1f}s)")

# ── 출력: CV / OOS / Full ────────────────────────────────────────────────────
def print_section(title, key):
    print("\n" + "=" * 130)
    print(f"  [{title}]")
    print("=" * 130)
    print(f"  {'전략':30s}  {'CAGR':>8s}  {'Sharpe':>7s}  {'Sortino':>7s}  "
          f"{'MDD':>8s}  {'Ulcer':>6s}  {'Calmar':>7s}  {'Pain':>6s}  {'UPI':>6s}")
    print("  " + "-" * 128)
    for r in results:
        m = r[key]
        print(
            f"  {r['label']:30s}  "
            f"{m['cagr']*100:>+7.2f}%  "
            f"{m['sharpe']:>7.2f}  "
            f"{m['sortino']:>7.2f}  "
            f"{m['mdd']*100:>7.2f}%  "
            f"{m['ulcer']:>6.2f}  "
            f"{m['calmar']:>7.2f}  "
            f"{m['pain_ratio']:>6.2f}  "
            f"{m['upi']:>6.2f}"
        )

print_section("CV 기간 (2002-2016)",  "cv")
print_section("OOS 기간 (2017-2026)", "oos")
print_section("전체 (2002-2026)",     "full")

# ── Paired bootstrap vs S1 (OOS only) ────────────────────────────────────────
print("\n" + "=" * 130)
print("  [Paired Bootstrap Difference Test  vs S1 (OOS, 2017-2026)]")
print("  (★ = p<0.05 통계적으로 유의,  · = p<0.20,  () 안은 P(strategy>S1))")
print("=" * 130)

s1_rets_oos = ports_oos["S1 (QQQ B&H)"].pct_change().dropna()
print(f"  Bootstrap: 60일 stationary block × 500회 (paired)")

bs_results = {}
print(f"\n  {'전략':30s}  {'ΔSharpe':>22s}  {'ΔCAGR':>22s}  "
      f"{'ΔSortino':>22s}  {'ΔUlcer':>22s}")
print("  " + "-" * 128)
for r in results:
    if r["label"] == "S1 (QQQ B&H)":
        continue
    strat_rets = ports_oos[r["label"]].pct_change().dropna()
    bs = paired_bootstrap_compare(strat_rets, s1_rets_oos,
                                   block_len=60, n_iter=500)
    bs_results[r["label"]] = bs

    def cell(key, fmt="{:+.3f}"):
        x = bs[key]
        sig = "★" if x["p_value"] < 0.05 else ("·" if x["p_value"] < 0.20 else " ")
        return f"{fmt.format(x['median'])}{sig} ({x['prob_better']*100:>3.0f}%)"

    print(f"  {r['label']:30s}  "
          f"{cell('delta_sharpe'):>22s}  "
          f"{cell('delta_cagr', '{:+.1%}'):>22s}  "
          f"{cell('delta_sortino'):>22s}  "
          f"{cell('delta_ulcer'):>22s}")

# ── 시각화 ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 13))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.30)
fig.suptitle(
    "Stage 4: 새 메트릭으로 모든 후보 재평가 (CV/OOS/Full + Paired Bootstrap vs S1)",
    fontsize=12, fontweight="bold")

# ① Sortino vs Sharpe (OOS)
ax = fig.add_subplot(gs[0, 0])
for r in results:
    color = "#000" if r["family"] == "S1" else ("#DC2626" if r["family"]=="S4" else "#2563EB")
    marker = "s" if "Anchor" in r["label"] else ("D" if "robust" in r["label"] else "o")
    ax.scatter(r["oos"]["sharpe"], r["oos"]["sortino"], s=120, color=color,
               marker=marker, alpha=0.85, edgecolors="black", lw=0.6)
    ax.annotate(r["label"].replace("S4 ", "").replace("S5 ", ""),
                (r["oos"]["sharpe"], r["oos"]["sortino"]),
                fontsize=7, xytext=(5,2), textcoords="offset points")
s1m = next(r for r in results if r["family"]=="S1")
ax.axvline(s1m["oos"]["sharpe"], color="#999", ls="--", lw=0.8)
ax.axhline(s1m["oos"]["sortino"], color="#999", ls="--", lw=0.8)
ax.set_xlabel("Sharpe (OOS)"); ax.set_ylabel("Sortino (OOS)")
ax.set_title("OOS Sharpe vs Sortino (적색=S4, 청색=S5, ▢=anchor, ◆=DE-robust, ○=other DE)",
             fontsize=9, fontweight="bold")
ax.grid(True, alpha=0.3)

# ② Calmar vs Pain (OOS)
ax = fig.add_subplot(gs[0, 1])
for r in results:
    color = "#000" if r["family"] == "S1" else ("#DC2626" if r["family"]=="S4" else "#2563EB")
    marker = "s" if "Anchor" in r["label"] else ("D" if "robust" in r["label"] else "o")
    ax.scatter(r["oos"]["calmar"], r["oos"]["pain_ratio"], s=120, color=color,
               marker=marker, alpha=0.85, edgecolors="black", lw=0.6)
    ax.annotate(r["label"].replace("S4 ", "").replace("S5 ", ""),
                (r["oos"]["calmar"], r["oos"]["pain_ratio"]),
                fontsize=7, xytext=(5,2), textcoords="offset points")
ax.axvline(s1m["oos"]["calmar"], color="#999", ls="--", lw=0.8)
ax.axhline(s1m["oos"]["pain_ratio"], color="#999", ls="--", lw=0.8)
ax.set_xlabel("Calmar (OOS)"); ax.set_ylabel("Pain Ratio (OOS)")
ax.set_title("OOS Calmar (CAGR/MDD) vs Pain (CAGR/avg-DD)",
             fontsize=9, fontweight="bold")
ax.grid(True, alpha=0.3)

# ③ Bootstrap ΔSharpe 분포
ax = fig.add_subplot(gs[1, 0])
for label, bs in bs_results.items():
    if "Anchor" in label or "robust" in label:
        sty = "-" if "Anchor" in label else "--"
        col = "#DC2626" if "S4" in label else "#2563EB"
        diffs = np.array(bs["raw"]["delta_sharpe"])
        ax.hist(diffs, bins=30, alpha=0.4, label=label, density=True,
                edgecolor="white", lw=0.5)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Δ Sharpe (vs S1)"); ax.set_ylabel("density")
ax.set_title("Paired bootstrap ΔSharpe 분포 (anchor / robust 비교)",
             fontsize=9, fontweight="bold")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# ④ Bootstrap ΔCAGR 분포
ax = fig.add_subplot(gs[1, 1])
for label, bs in bs_results.items():
    if "Anchor" in label or "robust" in label:
        diffs = np.array(bs["raw"]["delta_cagr"]) * 100
        ax.hist(diffs, bins=30, alpha=0.4, label=label, density=True,
                edgecolor="white", lw=0.5)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Δ CAGR (vs S1, %)"); ax.set_ylabel("density")
ax.set_title("Paired bootstrap ΔCAGR 분포",
             fontsize=9, fontweight="bold")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# ⑤ Win prob bar
ax = fig.add_subplot(gs[2, :])
labels_short, sh_p, ca_p, so_p, ul_p = [], [], [], [], []
for label, bs in bs_results.items():
    labels_short.append(label.replace(" (QQQ B&H)", "").replace("S4 ", "").replace("S5 ", ""))
    sh_p.append(bs["delta_sharpe"]["prob_better"] * 100)
    ca_p.append(bs["delta_cagr"]["prob_better"] * 100)
    so_p.append(bs["delta_sortino"]["prob_better"] * 100)
    ul_p.append((1-bs["delta_ulcer"]["prob_better"]) * 100)  # ulcer는 적을수록 좋음

x = np.arange(len(labels_short)); w = 0.20
ax.bar(x - 1.5*w, sh_p, w, label="P(ΔSharpe>0)",  color="#3B82F6", alpha=0.85)
ax.bar(x - 0.5*w, ca_p, w, label="P(ΔCAGR>0)",    color="#10B981", alpha=0.85)
ax.bar(x + 0.5*w, so_p, w, label="P(ΔSortino>0)", color="#8B5CF6", alpha=0.85)
ax.bar(x + 1.5*w, ul_p, w, label="P(Ulcer↓)",     color="#F59E0B", alpha=0.85)
ax.axhline(50, color="black", ls="--", lw=0.8, label="random (50%)")
ax.set_xticks(x); ax.set_xticklabels(labels_short, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Probability strategy beats S1 (%)")
ax.set_title("Paired bootstrap: 각 메트릭에서 S1을 이길 확률 (OOS, 500회)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8, ncol=5, loc="upper right")
ax.set_ylim(0, 105); ax.grid(True, alpha=0.3, axis="y")

plt.savefig(OUT_DIR / "stage4_metrics_recompute.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── JSON 저장 ────────────────────────────────────────────────────────────────
out = {
    "candidates": results,
    "bootstrap_oos_vs_s1": {
        label: {k: v for k, v in bs.items() if k != "raw"}
        for label, bs in bs_results.items()
    },
}

def _to_json(o):
    if isinstance(o, dict):
        return {k: _to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_json(x) for x in o]
    if isinstance(o, (np.floating, np.float64, np.float32)): return float(o)
    if isinstance(o, (np.integer, np.int64)): return int(o)
    return o

with open(OUT_DIR / "stage4_metrics_recompute.json", "w") as f:
    json.dump(_to_json(out), f, indent=2)

print(f"\n  PNG 저장 → {OUT_DIR / 'stage4_metrics_recompute.png'}")
print(f"  JSON 저장 → {OUT_DIR / 'stage4_metrics_recompute.json'}")
print("\n" + "=" * 78)
print("[STAGE 4 완료]")
print("=" * 78)
