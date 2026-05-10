"""
stage1_s6_ensemble.py
=====================
[단계 1] S6: Anchor Ensemble — 5개 직관적 anchor를 동시 운용.

설계
----
각 sub-strategy 가 자체 자본의 1/5 을 운용. 매일 모든 sub의 가치를 합산.
연 1회 (1월 1일) 균등 리밸런스.

5개 sub-anchor (S4 trailing 기반):
  A1 [기본]   : b=-5%,  trail=-15%, hf=0.50  (사용자 원본 anchor)
  A2 [느슨]   : b=-7%,  trail=-12%, hf=0.50  (덜 민감, 신호 적음)
  A3 [민감]   : b=-3%,  trail=-8%,  hf=0.50  (더 민감, 신호 많음)
  A4 [빠른청산]: b=-5%, trail=-7%,  hf=0.50  (수익 빨리 챙김)
  A5 [느린청산]: b=-5%, trail=-15%, hf=0.50  (큰 추세 잡기)

(r1=2, r2=2, full_frac=1.0 공통)

기대 효과
--------
• 단일 파라미터 path-risk 감소
• 서로 다른 regime/사이클에서 다른 sub가 활성 → 분산
• 포트폴리오 Sharpe가 anchor 평균보다 높을 가능성 (corr<1)
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

# ── 5 anchor ────────────────────────────────────────────────────────────────
SUB_ANCHORS = [
    {"name": "A1_기본",       "b": -0.05, "r1": 2.0, "r2": 2.0, "trail": -0.15, "hf": 0.50, "ff": 1.00},
    {"name": "A2_느슨",       "b": -0.07, "r1": 2.0, "r2": 2.0, "trail": -0.12, "hf": 0.50, "ff": 1.00},
    {"name": "A3_민감",       "b": -0.03, "r1": 2.0, "r2": 2.0, "trail": -0.08, "hf": 0.50, "ff": 1.00},
    {"name": "A4_빠른청산",   "b": -0.05, "r1": 2.0, "r2": 2.0, "trail": -0.07, "hf": 0.50, "ff": 1.00},
    {"name": "A5_느린청산",   "b": -0.05, "r1": 2.0, "r2": 2.0, "trail": -0.15, "hf": 0.50, "ff": 1.00},
]


def run_sub(Q, L, T, sub, capital):
    sd, db = sub["b"]*sub["r1"], sub["b"]*sub["r2"]
    dd, hs = sub["b"]*sub["r1"]*sub["r2"], (sd + sub["b"]*sub["r1"]*sub["r2"])/2
    port, events = strategy_s4_trailing(
        Q, L, T, capital,
        shallow_drop=sd, deep_drop=dd,
        shallow_bounce=sub["b"], deep_bounce=db,
        half_stop=hs, trailing_stop_pct=sub["trail"],
        half_frac=sub["hf"], full_frac=sub["ff"]
    )
    return port, events


def run_s6_ensemble(Q, L, T, total_capital=100_000, rebalance_freq="A"):
    """각 sub에 동일 capital 배분, 매년 1월 1일 리밸런스.

    rebalance_freq: 'A' (annual) | 'Q' (quarterly) | 'NONE' (no rebal, drift)
    """
    n_sub = len(SUB_ANCHORS)
    sub_cap = total_capital / n_sub
    # 모든 sub를 한 번에 실행 (전체 기간)
    sub_ports = []
    for sub in SUB_ANCHORS:
        port, _ = run_sub(Q, L, T, sub, sub_cap)
        sub_ports.append(port)
    # DataFrame으로 정렬
    df = pd.concat(sub_ports, axis=1)
    df.columns = [s["name"] for s in SUB_ANCHORS]

    if rebalance_freq == "NONE":
        # drift 그대로 합산
        ensemble = df.sum(axis=1)
        return ensemble, df

    # ── 리밸런싱 적용 ──
    if rebalance_freq == "A":
        rebal_dates = pd.date_range(df.index[0], df.index[-1], freq="AS")  # year start
    else:
        rebal_dates = pd.date_range(df.index[0], df.index[-1], freq="QS")
    rebal_dates = [d for d in rebal_dates if d in df.index or d > df.index[0]]

    # daily returns of each sub
    sub_rets = df.pct_change().fillna(0).values
    n_days, n = sub_rets.shape
    weights = np.ones(n) / n
    capital = total_capital
    portfolio = np.empty(n_days)
    portfolio[0] = capital

    rebal_set = set()
    for d in rebal_dates:
        # find first date >= d
        loc = df.index.searchsorted(d)
        if loc < n_days:
            rebal_set.add(loc)

    sub_caps = np.full(n, capital / n)
    for t in range(1, n_days):
        sub_caps = sub_caps * (1 + sub_rets[t])
        capital = sub_caps.sum()
        if t in rebal_set:
            sub_caps = np.full(n, capital / n)
        portfolio[t] = capital

    return pd.Series(portfolio, index=df.index, name="S6_Ensemble"), df


# ── 데이터 ───────────────────────────────────────────────────────────────────
print("=" * 78)
print("[STAGE 1] S6 Anchor Ensemble (5 sub × annual rebalance)")
print("=" * 78)
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

# ── 실행 ─────────────────────────────────────────────────────────────────────
print("\n각 sub-anchor 정의:")
for s in SUB_ANCHORS:
    print(f"  {s['name']:14s}  b={s['b']*100:+.0f}%  trail={s['trail']*100:+.0f}%  hf={s['hf']:.2f}")

print("\n  ▶ Annual rebalance (매년 1월 균등 1/5)")
t0 = time.time()
s6_cv,   subs_cv   = run_s6_ensemble(Q_cv,   L_cv,   T_cv,   100_000, "A")
s6_oos,  subs_oos  = run_s6_ensemble(Q_oos,  L_oos,  T_oos,  100_000, "A")
s6_full, subs_full = run_s6_ensemble(Q_full, L_full, T_full, 100_000, "A")
print(f"  완료 ({time.time()-t0:.1f}s)")

# ── S1, S4 Anchor 비교용 백테스트 ────────────────────────────────────────────
s1_cv   = strategy_buy_and_hold(Q_cv,   100_000)
s1_oos  = strategy_buy_and_hold(Q_oos,  100_000)
s1_full = strategy_buy_and_hold(Q_full, 100_000)

# A1 (사용자 원본 anchor)도 따로 보존 비교
a1 = SUB_ANCHORS[0]
s4a_cv,   _ = run_sub(Q_cv,   L_cv,   T_cv,   a1, 100_000)
s4a_oos,  _ = run_sub(Q_oos,  L_oos,  T_oos,  a1, 100_000)
s4a_full, _ = run_sub(Q_full, L_full, T_full, a1, 100_000)

# ── 메트릭 ───────────────────────────────────────────────────────────────────
def show_metrics(label, p_cv, p_oos, p_full):
    m_cv, m_oos, m_full = full_metrics(p_cv), full_metrics(p_oos), full_metrics(p_full)
    print(f"  {label:24s}  CV: CAGR={m_cv['cagr']*100:>+5.1f}% Sh={m_cv['sharpe']:.2f} MDD={m_cv['mdd']*100:>+6.1f}% Ulcer={m_cv['ulcer']:>5.2f}  "
          f"|  OOS: CAGR={m_oos['cagr']*100:>+5.1f}% Sh={m_oos['sharpe']:.2f} MDD={m_oos['mdd']*100:>+6.1f}% Ulcer={m_oos['ulcer']:>5.2f}  "
          f"|  Full: CAGR={m_full['cagr']*100:>+5.1f}% Sh={m_full['sharpe']:.2f}")
    return m_cv, m_oos, m_full

print("\n" + "─" * 168)
print("  [기준선]")
m_s1 = show_metrics("S1 (QQQ B&H)",       s1_cv,  s1_oos,  s1_full)
m_s4 = show_metrics("S4 Anchor (A1)",     s4a_cv, s4a_oos, s4a_full)
print("\n  [S6 Ensemble]")
m_s6 = show_metrics("S6 Ensemble (5sub,Y)", s6_cv,  s6_oos,  s6_full)

print("\n  [Sub-anchor 개별 OOS 성과 (참고)]")
for col in subs_oos.columns:
    p = subs_oos[col]
    m = full_metrics(p)
    print(f"    {col:14s}  CAGR={m['cagr']*100:>+5.1f}%  Sh={m['sharpe']:.2f}  MDD={m['mdd']*100:>+6.1f}%  Ulcer={m['ulcer']:>5.2f}")

# Correlation matrix between subs (OOS)
sub_rets_oos = subs_oos.pct_change().dropna()
print("\n  [Sub-strategy correlation matrix (OOS daily returns)]")
corr = sub_rets_oos.corr()
print("    " + "  ".join(f"{c[:6]:>7s}" for c in corr.columns))
for idx, row in corr.iterrows():
    print(f"    {idx[:6]:>7s}  " + "  ".join(f"{v:>7.3f}" for v in row.values))

# 평균 corr (off-diagonal)
mask = ~np.eye(corr.shape[0], dtype=bool)
mean_corr = corr.values[mask].mean()
print(f"\n  → 평균 sub간 corr = {mean_corr:.3f}  (낮을수록 분산효과 큼)")

# ── Bootstrap (S6 vs S1) ─────────────────────────────────────────────────────
print("\n" + "─" * 168)
print("  [Paired Bootstrap: S6 vs S1, OOS, 60일 블록 × 500회]")
s1_rets_oos = s1_oos.pct_change().dropna()
s6_rets_oos = s6_oos.pct_change().dropna()
bs_s6 = paired_bootstrap_compare(s6_rets_oos, s1_rets_oos, 60, 500)
s4a_rets_oos = s4a_oos.pct_change().dropna()
bs_s4a = paired_bootstrap_compare(s4a_rets_oos, s1_rets_oos, 60, 500)

def show_bs(label, bs):
    def cell(key, fmt="{:+.3f}"):
        x = bs[key]; sig = "★" if x["p_value"]<0.05 else ("·" if x["p_value"]<0.20 else " ")
        return f"{fmt.format(x['median'])}{sig} ({x['prob_better']*100:.0f}%)"
    print(f"  {label:24s}  ΔSharpe={cell('delta_sharpe'):>20s}  "
          f"ΔCAGR={cell('delta_cagr','{:+.1%}'):>22s}  "
          f"ΔSortino={cell('delta_sortino'):>20s}  "
          f"ΔUlcer={cell('delta_ulcer'):>20s}")

show_bs("S4 Anchor (A1) vs S1", bs_s4a)
show_bs("S6 Ensemble vs S1",    bs_s6)

# ── 시각화 ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 12))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.28)
fig.suptitle(
    "S6: Anchor Ensemble (5 sub × annual rebalance)\n"
    f"OOS: {Q_oos.index[0].date()} ~ {Q_oos.index[-1].date()}",
    fontsize=12, fontweight="bold")

# ① OOS 자산곡선 (S1, S4 anchor, S6, sub들)
ax = fig.add_subplot(gs[0, :])
ax.plot(s1_oos.index,  s1_oos.values  / s1_oos.iloc[0]  * 100, label="S1 QQQ B&H",
        color="#000", lw=1.5, alpha=0.7)
ax.plot(s4a_oos.index, s4a_oos.values / s4a_oos.iloc[0] * 100, label="S4 Anchor (A1)",
        color="#DC2626", lw=1.6)
ax.plot(s6_oos.index,  s6_oos.values  / s6_oos.iloc[0]  * 100, label="S6 Ensemble",
        color="#2563EB", lw=2.0)
for col in subs_oos.columns:
    p = subs_oos[col]
    ax.plot(p.index, p.values / p.iloc[0] * 100, label=col, lw=0.8, alpha=0.45)
ax.set_yscale("log")
ax.set_ylabel("Index (start=100, log)")
ax.set_title("OOS 자산곡선 (S1 / S4 Anchor / S6 Ensemble + 5 sub)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8, ncol=3, loc="upper left")
ax.grid(True, alpha=0.3)

# ② sub correlation heatmap
ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(corr.values, cmap="RdYlGn_r", vmin=0.5, vmax=1.0)
ax.set_xticks(range(5)); ax.set_xticklabels(corr.columns, fontsize=8, rotation=30, ha="right")
ax.set_yticks(range(5)); ax.set_yticklabels(corr.index, fontsize=8)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                fontsize=8, color="white" if corr.values[i,j]>0.85 else "black")
plt.colorbar(im, ax=ax, fraction=0.05)
ax.set_title(f"Sub corr (OOS daily) — 평균={mean_corr:.3f}",
             fontsize=10, fontweight="bold")

# ③ Sub별 OOS 성과 비교
ax = fig.add_subplot(gs[1, 1])
sub_names = list(subs_oos.columns)
sub_cagrs = [full_metrics(subs_oos[c])["cagr"]*100 for c in sub_names]
sub_sh    = [full_metrics(subs_oos[c])["sharpe"]   for c in sub_names]
y = np.arange(len(sub_names))
ax.barh(y, sub_cagrs, color="#3B82F6", alpha=0.85, label="CAGR (%)")
axb = ax.twiny()
axb.plot(sub_sh, y, "o-", color="#DC2626", lw=1.8, ms=8, label="Sharpe")
ax.axvline(full_metrics(s1_oos)["cagr"]*100, color="#000", ls="--", lw=1, label="S1 CAGR")
axb.axvline(full_metrics(s1_oos)["sharpe"], color="#DC2626", ls="--", lw=1, alpha=0.5, label="S1 Sharpe")
ax.set_yticks(y); ax.set_yticklabels(sub_names, fontsize=9)
ax.set_xlabel("CAGR (%)", color="#3B82F6"); axb.set_xlabel("Sharpe", color="#DC2626")
ax.set_title("Sub-anchor OOS 성과 (CAGR & Sharpe)", fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")
lines1, lbs1 = ax.get_legend_handles_labels(); lines2, lbs2 = axb.get_legend_handles_labels()
ax.legend(lines1+lines2, lbs1+lbs2, fontsize=8, loc="lower right")

# ④ Bootstrap ΔSharpe distribution
ax = fig.add_subplot(gs[2, 0])
ax.hist(np.array(bs_s4a["raw"]["delta_sharpe"]), bins=30, alpha=0.45,
        label="S4 Anchor", color="#DC2626", density=True, edgecolor="white")
ax.hist(np.array(bs_s6["raw"]["delta_sharpe"]), bins=30, alpha=0.45,
        label="S6 Ensemble", color="#2563EB", density=True, edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Δ Sharpe (vs S1)"); ax.set_ylabel("density")
ax.set_title("Paired bootstrap ΔSharpe", fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ⑤ Bootstrap ΔCAGR distribution
ax = fig.add_subplot(gs[2, 1])
ax.hist(np.array(bs_s4a["raw"]["delta_cagr"])*100, bins=30, alpha=0.45,
        label="S4 Anchor", color="#DC2626", density=True, edgecolor="white")
ax.hist(np.array(bs_s6["raw"]["delta_cagr"])*100, bins=30, alpha=0.45,
        label="S6 Ensemble", color="#2563EB", density=True, edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Δ CAGR (vs S1, %)"); ax.set_ylabel("density")
ax.set_title("Paired bootstrap ΔCAGR", fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.savefig(OUT_DIR / "stage1_s6_ensemble.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── JSON 저장 ────────────────────────────────────────────────────────────────
out = {
    "sub_anchors": SUB_ANCHORS,
    "s1": {"cv": m_s1[0], "oos": m_s1[1], "full": m_s1[2]},
    "s4_anchor": {"cv": m_s4[0], "oos": m_s4[1], "full": m_s4[2]},
    "s6_ensemble": {"cv": m_s6[0], "oos": m_s6[1], "full": m_s6[2]},
    "subs_oos": {col: full_metrics(subs_oos[col]) for col in subs_oos.columns},
    "sub_corr_oos_mean": float(mean_corr),
    "bootstrap_s6_vs_s1":  {k: v for k, v in bs_s6.items() if k != "raw"},
    "bootstrap_s4a_vs_s1": {k: v for k, v in bs_s4a.items() if k != "raw"},
}

def _to_json(o):
    if isinstance(o, dict): return {k: _to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_to_json(x) for x in o]
    if isinstance(o, (np.floating, np.float64, np.float32)): return float(o)
    if isinstance(o, (np.integer, np.int64)): return int(o)
    return o

with open(OUT_DIR / "stage1_s6_ensemble.json", "w") as f:
    json.dump(_to_json(out), f, indent=2)

print(f"\n  PNG → {OUT_DIR / 'stage1_s6_ensemble.png'}")
print(f"  JSON → {OUT_DIR / 'stage1_s6_ensemble.json'}")
print("\n[STAGE 1 완료]")
