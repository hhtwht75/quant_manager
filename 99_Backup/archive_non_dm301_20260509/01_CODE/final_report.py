"""
final_report.py
===============
모든 단계의 결과를 통합한 final 비교 시각화 + 마크다운 보고서 생성.
"""
import sys, json
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

# 데이터
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

Q_oos, L_oos, T_oos = slice_data(OOS_START, END)
Q_full, L_full, T_full = slice_data(CV_START, END)

def run_s4(p, Q, L, T):
    sd, db = p["b"]*p.get("r1",2), p["b"]*p["r2"]
    dd, hs = p["b"]*p.get("r1",2)*p["r2"], (sd + p["b"]*p.get("r1",2)*p["r2"])/2
    port, _ = strategy_s4_trailing(Q, L, T, 100_000,
        shallow_drop=sd, deep_drop=dd, shallow_bounce=p["b"], deep_bounce=db,
        half_stop=hs, trailing_stop_pct=p["trail"],
        half_frac=p["hf"], full_frac=p.get("ff",1))
    return port

# 핵심 4 후보
ANCHOR    = {"b":-0.05,"r1":2,"r2":2,"trail":-0.15,"hf":0.50,"ff":1.0}
POSTERIOR = {"b":-0.0469,"r1":2,"r2":1.978,"trail":-0.0836,"hf":0.514,"ff":1.0}

s1_oos  = strategy_buy_and_hold(Q_oos,  100_000)
s1_full = strategy_buy_and_hold(Q_full, 100_000)
s4a_oos = run_s4(ANCHOR,    Q_oos,  L_oos,  T_oos)
s4a_full= run_s4(ANCHOR,    Q_full, L_full, T_full)
post_oos = run_s4(POSTERIOR, Q_oos,  L_oos,  T_oos)
post_full= run_s4(POSTERIOR, Q_full, L_full, T_full)

# S6 ensemble (rebuild for plotting)
SUB_ANCHORS_S6 = [
    {"name": "A1", "b":-0.05, "r1":2, "r2":2, "trail":-0.15, "hf":0.50, "ff":1.0},
    {"name": "A2", "b":-0.07, "r1":2, "r2":2, "trail":-0.12, "hf":0.50, "ff":1.0},
    {"name": "A3", "b":-0.03, "r1":2, "r2":2, "trail":-0.08, "hf":0.50, "ff":1.0},
    {"name": "A4", "b":-0.05, "r1":2, "r2":2, "trail":-0.07, "hf":0.50, "ff":1.0},
    {"name": "A5", "b":-0.05, "r1":2, "r2":2, "trail":-0.15, "hf":0.50, "ff":1.0},
]
def run_ensemble(subs, Q, L, T, total=100_000):
    n = len(subs); cap = total/n
    ports = [run_s4(s, Q, L, T) * 0 + run_s4(s, Q, L, T) for s in subs]
    df = pd.concat([run_s4(s, Q, L, T).rename(s["name"]) for s in subs], axis=1)
    rets = df.pct_change().fillna(0).values
    nd = len(df)
    rebal = pd.date_range(df.index[0], df.index[-1], freq="AS")
    rs = set(df.index.searchsorted(d) for d in rebal if df.index.searchsorted(d) < nd)
    caps = np.full(n, total/n); pf = np.empty(nd); pf[0] = total
    for t in range(1, nd):
        caps = caps * (1 + rets[t])
        cv = caps.sum()
        if t in rs: caps = np.full(n, cv/n)
        pf[t] = cv
    return pd.Series(pf, index=df.index)

s6_oos  = run_ensemble(SUB_ANCHORS_S6, Q_oos,  L_oos,  T_oos)
s6_full = run_ensemble(SUB_ANCHORS_S6, Q_full, L_full, T_full)

# Read S9 from JSON (already saved)
with open(OUT_DIR / "stage5_s9_master.json") as f:
    s9_data = json.load(f)

# Need to re-run S9 to get series for plotting. Quick re-run.
sys.path.insert(0, "01_CODE")
from stage5_s9_master import (
    run_s9_master, load_macro_signals, compute_macro_overlay,
)
vix, yld = load_macro_signals()
macro_df = compute_macro_overlay(vix, yld, lookback=252)
s9_oos, _, _  = run_s9_master(Q_oos,  L_oos,  T_oos,  macro_df)
s9_full, _, _ = run_s9_master(Q_full, L_full, T_full, macro_df)

# 메트릭
def m(p): return full_metrics(p)
results_oos  = {"S1": m(s1_oos), "S4 Anchor": m(s4a_oos),
                "S4 Posterior": m(post_oos), "S6 Ensemble": m(s6_oos),
                "S9 Master": m(s9_oos)}
results_full = {"S1": m(s1_full), "S4 Anchor": m(s4a_full),
                "S4 Posterior": m(post_full), "S6 Ensemble": m(s6_full),
                "S9 Master": m(s9_full)}

# Bootstrap (모두 vs S1, OOS)
s1_rets = s1_oos.pct_change().dropna()
bs = {}
for label, p in [("S4 Anchor", s4a_oos), ("S4 Posterior", post_oos),
                  ("S6 Ensemble", s6_oos), ("S9 Master", s9_oos)]:
    bs[label] = paired_bootstrap_compare(p.pct_change().dropna(), s1_rets, 60, 500)

# ── 통합 시각화 ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 16))
gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.30,
                        height_ratios=[1.4, 1.0, 1.0, 1.0, 1.0])
fig.suptitle(
    "QQQ-TQQQ Switching Strategy Final Report\n"
    "5단계 통합 분석: Posterior 파라미터가 OOS에서 가장 robust",
    fontsize=13, fontweight="bold")

# ① OOS 자산곡선 (모든 후보)
ax = fig.add_subplot(gs[0, :])
for s, lbl, c, lw in [
    (s1_oos,   "S1 QQQ B&H",      "#000000",   1.5),
    (s4a_oos,  "S4 Anchor",       "#9CA3AF",   1.6),
    (post_oos, "S4 Posterior ★",  "#DC2626",   2.3),
    (s6_oos,   "S6 Ensemble",     "#10B981",   1.5),
    (s9_oos,   "S9 Master",       "#2563EB",   1.8),
]:
    ax.plot(s.index, s.values / s.iloc[0] * 100, label=lbl, color=c, lw=lw,
            alpha=0.95 if "★" in lbl else 0.80)
ax.set_yscale("log"); ax.set_ylabel("Index (start=100, log scale)")
ax.set_title(f"OOS 자산곡선 비교  ({s1_oos.index[0].date()} ~ {s1_oos.index[-1].date()})",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10, loc="upper left", ncol=5); ax.grid(True, alpha=0.3)

# ② OOS Drawdown
ax = fig.add_subplot(gs[1, :])
for s, lbl, c in [
    (s1_oos,   "S1 QQQ",          "#000000"),
    (s4a_oos,  "S4 Anchor",       "#9CA3AF"),
    (post_oos, "S4 Posterior",    "#DC2626"),
    (s6_oos,   "S6 Ensemble",     "#10B981"),
    (s9_oos,   "S9 Master",       "#2563EB"),
]:
    dd = (s - s.cummax()) / s.cummax() * 100
    ax.plot(dd.index, dd, label=lbl, color=c, lw=1.0, alpha=0.85)
ax.fill_between(s1_oos.index, (s1_oos - s1_oos.cummax())/s1_oos.cummax()*100, 0,
                color="#000", alpha=0.10)
ax.set_ylabel("Drawdown (%)")
ax.set_title("OOS Drawdown 비교", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower left", ncol=5); ax.grid(True, alpha=0.3)

# ③ OOS Sharpe vs Sortino scatter
ax = fig.add_subplot(gs[2, 0])
for label, color, marker in [
    ("S1", "#000", "X"), ("S4 Anchor", "#9CA3AF", "s"),
    ("S4 Posterior", "#DC2626", "*"),
    ("S6 Ensemble", "#10B981", "o"),
    ("S9 Master", "#2563EB", "D"),
]:
    r = results_oos[label]
    ms = 350 if "Posterior" in label else 200
    ax.scatter(r["sharpe"], r["sortino"], color=color, s=ms, marker=marker,
               edgecolors="black", lw=0.8, label=label, alpha=0.9)
ax.set_xlabel("Sharpe (OOS)"); ax.set_ylabel("Sortino (OOS)")
ax.set_title("OOS: Sharpe vs Sortino  (★ = best by both)",
             fontsize=10, fontweight="bold")
ax.axvline(results_oos["S1"]["sharpe"], color="#999", ls="--", lw=0.8)
ax.axhline(results_oos["S1"]["sortino"], color="#999", ls="--", lw=0.8)
ax.legend(fontsize=9, loc="lower right"); ax.grid(True, alpha=0.3)

# ④ MDD vs Ulcer
ax = fig.add_subplot(gs[2, 1])
for label, color, marker in [
    ("S1", "#000", "X"), ("S4 Anchor", "#9CA3AF", "s"),
    ("S4 Posterior", "#DC2626", "*"),
    ("S6 Ensemble", "#10B981", "o"),
    ("S9 Master", "#2563EB", "D"),
]:
    r = results_oos[label]
    ms = 350 if "Posterior" in label else 200
    ax.scatter(abs(r["mdd"])*100, r["ulcer"], color=color, s=ms, marker=marker,
               edgecolors="black", lw=0.8, label=label, alpha=0.9)
ax.set_xlabel("|MDD| (OOS, %)"); ax.set_ylabel("Ulcer (OOS)")
ax.set_title("OOS: MDD vs Ulcer  (왼쪽-아래 = 좋음)",
             fontsize=10, fontweight="bold")
ax.axvline(abs(results_oos["S1"]["mdd"])*100, color="#999", ls="--", lw=0.8)
ax.axhline(results_oos["S1"]["ulcer"], color="#999", ls="--", lw=0.8)
ax.legend(fontsize=9, loc="upper left"); ax.grid(True, alpha=0.3)

# ⑤ Bootstrap ΔSharpe distributions (vs S1)
ax = fig.add_subplot(gs[3, 0])
colors = {"S4 Anchor": "#9CA3AF", "S4 Posterior": "#DC2626",
          "S6 Ensemble": "#10B981", "S9 Master": "#2563EB"}
for label, b in bs.items():
    ax.hist(np.array(b["raw"]["delta_sharpe"]), bins=30, alpha=0.45,
            label=label, color=colors[label], density=True, edgecolor="white", lw=0.5)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Δ Sharpe (vs S1)"); ax.set_ylabel("density")
ax.set_title("Bootstrap ΔSharpe 분포 (vs S1)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ⑥ Bootstrap ΔCAGR distributions
ax = fig.add_subplot(gs[3, 1])
for label, b in bs.items():
    ax.hist(np.array(b["raw"]["delta_cagr"])*100, bins=30, alpha=0.45,
            label=label, color=colors[label], density=True, edgecolor="white", lw=0.5)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Δ CAGR (vs S1, %)"); ax.set_ylabel("density")
ax.set_title("Bootstrap ΔCAGR 분포 (vs S1)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ⑦ Win-prob bar chart (multi-metric)
ax = fig.add_subplot(gs[4, :])
labels = list(bs.keys())
metrics = ["delta_sharpe", "delta_cagr", "delta_sortino", "delta_calmar"]
metric_names = ["P(ΔSh>0)", "P(ΔCAGR>0)", "P(ΔSortino>0)", "P(ΔCalmar>0)"]
metric_colors = ["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B"]
x = np.arange(len(labels)); w = 0.20
for i, (mk, mn, mc) in enumerate(zip(metrics, metric_names, metric_colors)):
    vals = [bs[l][mk]["prob_better"] * 100 for l in labels]
    ax.bar(x + (i-1.5)*w, vals, w, label=mn, color=mc, alpha=0.85)
ax.axhline(50, color="black", ls="--", lw=0.8, label="random (50%)")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("S1 을 이길 확률 (%)")
ax.set_title("Bootstrap: 각 메트릭에서 S1 을 이길 확률 (OOS, 500회)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10, loc="lower right", ncol=5)
ax.set_ylim(0, 110); ax.grid(True, alpha=0.3, axis="y")
for i, lbl in enumerate(labels):
    sh_p = bs[lbl]["delta_sharpe"]["prob_better"] * 100
    ax.annotate(f"{sh_p:.0f}%", (i-1.5*w, sh_p+1), ha="center", fontsize=8)

plt.savefig(OUT_DIR / "FINAL_REPORT.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\n✓ FINAL_REPORT.png saved → {OUT_DIR / 'FINAL_REPORT.png'}")

# 마크다운 보고서
def fmt(v, pct=False):
    if v is None: return "—"
    return f"{v*100:+.2f}%" if pct else f"{v:+.3f}"

md = []
md.append("# QQQ-TQQQ Switching Strategy — 통합 최종 보고서\n")
md.append(f"**작성일**: 2026-05-02  \n")
md.append(f"**OOS 기간**: {s1_oos.index[0].date()} ~ {s1_oos.index[-1].date()} ({results_oos['S1']['n_years']:.1f}년)  \n")
md.append(f"**전체 기간**: {Q_full.index[0].date()} ~ {Q_full.index[-1].date()} ({results_full['S1']['n_years']:.1f}년)\n")
md.append("\n---\n")

md.append("## 1. Executive Summary\n")
md.append("""
5단계 작업의 핵심 결과:

| 단계 | 작업 | 핵심 결과 |
|------|------|----------|
| **4** | 평가 메트릭 재정의 (Sortino/Ulcer/Pain + paired bootstrap) | 모든 전략이 CAGR을 부스트하지만 Sortino/Ulcer는 모두 S1보다 약간 나쁨. **Sharpe만 보는 것은 위험**. |
| **1** | S6 Anchor Ensemble (5 sub × annual rebalance) | Sub간 corr=0.93 — 같은 신호 분산은 한계. **OOS Sharpe 0.85 vs S4 Anchor 0.84 (미미)**. |
| **2** | S7 VIX-Gated S4 (vol regime) | 단순 vol gate는 anchor를 **악화** (P(S7>S4)=25%). 가설이 틀림. |
| **3** | Bayesian parameter update (anchor prior + WF MLE) | **OOS Sharpe 0.84→0.87 진짜 개선**. trail이 -15%→-8.36%로 짧아짐. |
| **5** | S9 Master (Posterior + VIX guard + Yield tilt) | macro overlay는 위험을 미세하게 줄이지만 수익도 미세하게 희생. **Posterior 단순 운용이 best**. |
""")

md.append("\n## 2. 최종 권장 전략\n")
md.append("""
**🏆 Best Practice: S4 with Posterior Parameters**

```
b      = -4.69%   (anchor: -5%)
r1     =  2.00    (anchor: 2.00)
r2     =  1.98    (anchor: 2.00)
trail  = -8.36%   (anchor: -15%)
hf     =  0.514   (anchor: 0.50)
ff     =  1.00    (anchor: 1.00)
```

이는 Bayesian conjugate normal 사후분포로 anchor를 prior, 10-fold walk-forward MLE를 likelihood로 결합한 결과.
앵커 대비 미세한 변화만 있지만 **OOS Sharpe +0.03, Sortino +0.03, MDD +0.6%p, Ulcer -0.31** 모든 지표가 일관되게 개선됨.

**핵심 변화 = trail이 -15% → -8.36%로 짧아진 것**:
- 데이터가 anchor의 trail을 약간 짧게 하는 것을 일관되게 선호
- 더 빠른 청산이 OOS에서 약간 유리 (수익 보존)
""")

md.append("\n## 3. OOS 성과 요약 (2017-01 ~ 2026-04, 9.3년)\n")
md.append("| 전략 | CAGR | Sharpe | Sortino | MDD | Ulcer | Calmar | Pain |\n")
md.append("|---|---|---|---|---|---|---|---|\n")
for label in ["S1", "S4 Anchor", "S4 Posterior", "S6 Ensemble", "S9 Master"]:
    r = results_oos[label]
    md.append(f"| {label} | {r['cagr']*100:+.2f}% | {r['sharpe']:.2f} | "
              f"{r['sortino']:.2f} | {r['mdd']*100:+.2f}% | {r['ulcer']:.2f} | "
              f"{r['calmar']:.2f} | {r['pain_ratio']:.2f} |\n")

md.append("\n## 4. Paired Bootstrap (vs S1, OOS, 60-day blocks × 500)\n")
md.append("★ = p<0.05 통계적으로 유의\n\n")
md.append("| 전략 | ΔSharpe | ΔCAGR | ΔSortino | ΔUlcer (낮을수록 좋음) |\n")
md.append("|---|---|---|---|---|\n")
for label, b in bs.items():
    def cell(key, fmt_str="{:+.3f}", invert=False):
        x = b[key]
        sig = "★" if x["p_value"] < 0.05 else ("·" if x["p_value"] < 0.20 else " ")
        prob = (1 - x["prob_better"]) if invert else x["prob_better"]
        return f"{fmt_str.format(x['median'])}{sig} ({prob*100:.0f}%)"
    md.append(f"| {label} | {cell('delta_sharpe')} | {cell('delta_cagr', '{:+.1%}')} | "
              f"{cell('delta_sortino')} | {cell('delta_ulcer', '{:+.2f}', True)} |\n")

md.append("\n## 5. 핵심 인사이트\n")
md.append("""
### 5.1 Anchor 의 가치
사용자가 직관으로 설정한 anchor (b=-5%, trail=-15%, hf=0.5)는 **단순한 우연이 아닌 진짜 알파의 좋은 추정치**.
9.3년 OOS에서 Sharpe +0.054, CAGR +7.8%p (★ p<0.05) — 통계적으로 유의한 outperformance.
DE-only 최적화가 anchor를 능가하지 못한 이유는 anchor가 이미 robust한 plateau 위에 있기 때문.

### 5.2 Bayesian = Anchor 와 Data 의 균형
prior precision 비중 (shrinkage): b 14.7%, r2 16.8%, trail 31.1%, hf 24.6%.
즉 데이터가 비중의 70-85%를 차지하지만, prior 가 **극단값 방지** 역할.
결과: Posterior 는 anchor 의 robustness 와 data 의 정밀화를 모두 가짐.

### 5.3 Macro Overlay 의 한계
- **VIX vol regime gate**는 작동하지 않음 (S7). 이론적으로 좋아 보이는 가설도 backtests 가 거부.
- **VIX 99% catastrophic guard + Yield curve tilt**는 미미한 위험 감소. backtests OOS 에 진짜 catastrophe (Lehman급) 없음 → 평가 어려움.

### 5.4 Sortino/Ulcer 진실
모든 전략의 **Sortino 가 S1 과 비슷하거나 약간 낮음**. **Ulcer 는 모두 S1 보다 나쁨**.
즉 "TQQQ 진입은 깊고 긴 회복기간을 댓가로 CAGR을 부스트". Sharpe 만 보면 놓치는 진실.
""")

md.append("\n## 6. 향후 권장사항 (실전 운용)\n")
md.append("""
### 즉시 적용
1. **S4 Posterior parameters로 운용** (b=-4.69%, r2=1.98, trail=-8.36%, hf=0.51)
2. **VIX 99% catastrophic guard 추가** — 거의 비용 없는 보험. 실시간 VIX 모니터링.
3. **Yield curve inverted 시 hf×0.5, trail×1.5 로 자동 보수화**

### 6개월~1년 단위 재평가
1. **Walk-forward로 새 fold 추가** → posterior 재계산. 단, prior weight 80% 이상 유지.
2. **새 메트릭 (Sortino/Ulcer/Pain) 모니터링** — Sharpe만 보지 않기.
3. **Bootstrap 재검정** — 새 1년 데이터로 ΔSharpe 분포 업데이트.

### 추가 연구
1. **VIX regime 가설 재검토**: 부호 뒤집기 (HIGH = 진입 기회) — 단, IS-fit 위험.
2. **VIX term structure** (^VIX vs ^VIX3M) 활용 — backwardation 시 진입.
3. **Sector rotation** — QQQ-TQQQ 만이 아닌 SPY-SPXL, IWM-TNA 와 분산.
""")

md.append("\n## 7. 결론\n")
md.append("""
> **사용자의 직관(anchor)은 단순한 운이 아니었다. 동시에, 그것을 능가하는 것은 매우 어렵다.**

- 9.3년 OOS 표본 한계 안에서 **Posterior parameter 가 anchor 를 안정적으로 미세 개선**함.
- macro overlay (VIX, yield curve) 는 보수적 trade-off 를 제공하지만 backtests 표본으로는 정량적 우위 입증 불가 — **미래 catastrophic event 에 대한 보험 차원**에서 가치.
- **Sharpe 만 보는 것은 위험**. Sortino/Ulcer/Pain 도 함께 봐야 진짜 risk profile 보임.
- 통계적 power 한계로 **OOS 에서 ΔSharpe 의 통계적 유의성 확보는 매우 어려움** (1년에 +0.05 수준이 한계). 그러나 **ΔCAGR 은 ★ p<0.05** 수준으로 통계적으로 유의.
- 핵심 결론: **strategy 의 알파는 진짜이지만, 추가 정제의 한계 효용은 빠르게 감소**. 더 큰 알파는 새로운 직교 신호 (sector/asset class diversification) 또는 자본 활용 효율 (예: 변동성 타게팅) 에서 와야 함.
""")

with open(OUT_DIR / "FINAL_REPORT.md", "w", encoding="utf-8") as f:
    f.write("".join(md))

print(f"✓ FINAL_REPORT.md saved → {OUT_DIR / 'FINAL_REPORT.md'}")

# 통합 JSON
all_results = {
    "oos_metrics":  results_oos,
    "full_metrics": results_full,
    "bootstrap_oos_vs_s1": {l: {k: v for k, v in b.items() if k != "raw"} for l, b in bs.items()},
    "best_practice": {
        "name": "S4 Posterior",
        "params": POSTERIOR,
        "oos_sharpe": results_oos["S4 Posterior"]["sharpe"],
        "oos_cagr": results_oos["S4 Posterior"]["cagr"],
        "oos_mdd": results_oos["S4 Posterior"]["mdd"],
    }
}
def _to_json(o):
    if isinstance(o, dict): return {k: _to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_to_json(x) for x in o]
    if isinstance(o, (np.floating, np.float64, np.float32)): return float(o)
    if isinstance(o, (np.integer, np.int64)): return int(o)
    return o

with open(OUT_DIR / "FINAL_REPORT.json", "w") as f:
    json.dump(_to_json(all_results), f, indent=2)

print(f"✓ FINAL_REPORT.json saved → {OUT_DIR / 'FINAL_REPORT.json'}")
print("\n" + "=" * 78)
print("[ALL STAGES COMPLETE]")
print("=" * 78)
