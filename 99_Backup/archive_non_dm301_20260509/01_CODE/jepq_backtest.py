"""
jepq_backtest.py
================
앵커전략 (trail=-8%) × 방어자산 2종 (QQQ / JEPQ)
벤치마크 5종:
  B1. QQQ  B2. QQQ/TQQQ 50/50  B3. TQQQ
  B4. JEPQ B5. JEPQ/TQQQ 50/50

기간: JEPQ 상장일 (2022-05-04) ~ 2026-04-30
신호원: QLD drawdown (기존과 동일)
변경사항: trail_stop -15% (앵커 기준과 동일)
"""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

from backtest_switching import load_extended_daily, strategy_s4_trailing
from evaluation_metrics import full_metrics, paired_bootstrap_compare

OUT_DIR = Path("03_RESULT")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 파라미터 (anchor with trail=-8%) ─────────────────────────────────────────
PARAMS = dict(
    shallow_drop   = -0.10,   # QLD -10% 터치
    deep_drop      = -0.20,   # QLD -20% 터치
    shallow_bounce = -0.05,   # QLD -5% 반등 → half attack
    deep_bounce    = -0.10,   # QLD -10% 반등 → full attack
    half_stop      = -0.15,   # half attack 중 QLD -15% → exit
    trailing_stop_pct = -0.15,  # ★ trail -15% (통일)
    half_frac      = 0.50,
    full_frac      = 1.00,
)

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
def load_jepq():
    csv = Path("02_DATA/yahoo_extended/JEPQ/JEPQ_daily.csv")
    df = pd.read_csv(csv, index_col="Date", parse_dates=True)
    df.index.name = "Date"
    close = df["Close"]
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close},
                        index=df.index)

qqq  = load_extended_daily("QQQ")
qld  = load_extended_daily("QLD")
tqqq = load_extended_daily("TQQQ")
jepq = load_jepq()

# 공통 날짜 (JEPQ 상장일 기준)
START = pd.Timestamp("2022-05-04")
END   = jepq.index[-1]

common = qqq.index.intersection(qld.index).intersection(tqqq.index).intersection(jepq.index)
common = common[(common >= START) & (common <= END)]

Q   = qqq.loc[common]
Qld = qld.loc[common]
T   = tqqq.loc[common]
J   = jepq.loc[common]

print("=" * 70)
print("JEPQ 상장 이후 백테스트")
print(f"  기간: {common[0].date()} ~ {common[-1].date()}  ({len(common):,}일)")
print(f"  Trail stop: -15%  (앵커와 동일)")
print("=" * 70)

INIT = 100_000

# ── 전략 실행 ────────────────────────────────────────────────────────────────
# 전략 1: 방어자산 = QQQ
strat_qqq, ev_qqq = strategy_s4_trailing(Q, Qld, T, INIT,
    **PARAMS, series_name="S4 QQQ def.")

# 전략 2: 방어자산 = JEPQ
strat_jepq, ev_jepq = strategy_s4_trailing(J, Qld, T, INIT,
    **PARAMS, series_name="S4 JEPQ def.")

# ── 벤치마크 ─────────────────────────────────────────────────────────────────
def buy_and_hold(asset_df, name, cap=INIT):
    """단일 자산 B&H 포트폴리오."""
    shares = cap / asset_df["Close"].iloc[0]
    return pd.Series(shares * asset_df["Close"].values,
                     index=asset_df.index, name=name)

def half_half(a_df, b_df, name, cap=INIT):
    """두 자산 50/50 B&H (리밸런스 없음, drift)."""
    sha = (cap * 0.5) / a_df["Close"].iloc[0]
    shb = (cap * 0.5) / b_df["Close"].iloc[0]
    return pd.Series(
        sha * a_df["Close"].values + shb * b_df["Close"].values,
        index=a_df.index, name=name)

b_qqq       = buy_and_hold(Q, "B1: QQQ only")
b_qqq_tqqq  = half_half(Q, T, "B2: QQQ/TQQQ 50/50")
b_tqqq      = buy_and_hold(T, "B3: TQQQ only")
b_jepq      = buy_and_hold(J, "B4: JEPQ only")
b_jepq_tqqq = half_half(J, T, "B5: JEPQ/TQQQ 50/50")

# ── 메트릭 출력 ─────────────────────────────────────────────────────────────
all_series = {
    "B1: QQQ only":         b_qqq,
    "B2: QQQ/TQQQ 50/50":  b_qqq_tqqq,
    "B3: TQQQ only":        b_tqqq,
    "B4: JEPQ only":        b_jepq,
    "B5: JEPQ/TQQQ 50/50": b_jepq_tqqq,
    "S4 def.QQQ  (trail-8%)":  strat_qqq,
    "S4 def.JEPQ (trail-8%)":  strat_jepq,
}

print(f"\n{'전략':30s}  {'CAGR':>8s}  {'Sharpe':>7s}  {'Sortino':>8s}  "
      f"{'MDD':>8s}  {'Ulcer':>6s}  {'Calmar':>7s}  {'Pain':>6s}")
print("  " + "-" * 100)
metrics = {}
for name, s in all_series.items():
    m = full_metrics(s)
    metrics[name] = m
    print(f"  {name:30s}  {m['cagr']*100:>+7.2f}%  {m['sharpe']:>7.2f}  "
          f"{m['sortino']:>8.2f}  {m['mdd']*100:>+7.2f}%  "
          f"{m['ulcer']:>6.2f}  {m['calmar']:>7.2f}  {m['pain_ratio']:>6.2f}")

# ── Paired bootstrap vs QQQ ─────────────────────────────────────────────────
print(f"\n  [Paired Bootstrap vs B1:QQQ, 60-day block × 500]")
print(f"  {'전략':30s}  {'ΔSharpe':>22s}  {'ΔCAGR':>22s}  {'ΔSortino':>22s}  {'ΔUlcer':>22s}")
print("  " + "-" * 110)
qqq_rets = b_qqq.pct_change().dropna()
bs = {}
for name, s in all_series.items():
    if name == "B1: QQQ only":
        continue
    rets = s.pct_change().dropna()
    b = paired_bootstrap_compare(rets, qqq_rets, 60, 500)
    bs[name] = b
    def cell(key, fmt="{:+.3f}"):
        x = b[key]; sig = "★" if x["p_value"]<0.05 else ("·" if x["p_value"]<0.20 else " ")
        return f"{fmt.format(x['median'])}{sig} ({x['prob_better']*100:>3.0f}%)"
    print(f"  {name:30s}  {cell('delta_sharpe'):>22s}  "
          f"{cell('delta_cagr', '{:+.1%}'):>22s}  "
          f"{cell('delta_sortino'):>22s}  "
          f"{cell('delta_ulcer'):>22s}")

# ── 이벤트 통계 ─────────────────────────────────────────────────────────────
def ev_summary(ev_df, label):
    if ev_df.empty:
        print(f"  {label}: 이벤트 없음")
        return
    for t in ["TO_HALF_ATTACK", "TO_FULL_ATTACK", "TRAIL_EXIT", "TRAIL_FLOOR", "HALF_STOP"]:
        n = (ev_df["type"] == t).sum()
        if n > 0:
            print(f"    {t}: {n}건")

print("\n  [S4 def.QQQ  이벤트]"); ev_summary(ev_qqq,  "QQQ")
print("  [S4 def.JEPQ 이벤트]"); ev_summary(ev_jepq, "JEPQ")

# ── 시각화 ───────────────────────────────────────────────────────────────────
COLORS = {
    "B1: QQQ only":         "#1F2937",
    "B2: QQQ/TQQQ 50/50":  "#6B7280",
    "B3: TQQQ only":        "#9CA3AF",
    "B4: JEPQ only":        "#2563EB",
    "B5: JEPQ/TQQQ 50/50": "#60A5FA",
    "S4 def.QQQ  (trail-8%)":  "#DC2626",
    "S4 def.JEPQ (trail-8%)":  "#7C3AED",
}
STYLES = {
    "B1: QQQ only":         "-",
    "B2: QQQ/TQQQ 50/50":  "--",
    "B3: TQQQ only":        ":",
    "B4: JEPQ only":        "-",
    "B5: JEPQ/TQQQ 50/50": "--",
    "S4 def.QQQ  (trail-8%)":  "-",
    "S4 def.JEPQ (trail-8%)":  "-",
}
LW = {k: 2.5 if "S4" in k else 1.4 for k in COLORS}

fig = plt.figure(figsize=(18, 16))
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.28,
                        height_ratios=[1.5, 1.0, 1.0, 1.0])
fig.suptitle(
    f"QQQ-TQQQ Switching  —  Anchor trail=−8%  |  JEPQ 상장 이후\n"
    f"기간: {common[0].strftime('%Y-%m-%d')} ~ {common[-1].strftime('%Y-%m-%d')}",
    fontsize=13, fontweight="bold")

# ① 자산곡선 전체
ax = fig.add_subplot(gs[0, :])
for name, s in all_series.items():
    idx = (s / s.iloc[0]) * 100
    ax.plot(s.index, idx, label=name,
            color=COLORS[name], lw=LW[name], ls=STYLES[name], alpha=0.92)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
ax.set_ylabel("Index (start=100, log)")
ax.set_title("자산곡선 비교 (log scale)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, ncol=2, loc="upper left"); ax.grid(True, alpha=0.3)

# ② Drawdown
ax = fig.add_subplot(gs[1, :])
for name, s in all_series.items():
    dd = (s - s.cummax()) / s.cummax() * 100
    ax.plot(s.index, dd, label=name,
            color=COLORS[name], lw=LW[name], ls=STYLES[name], alpha=0.85)
ax.set_ylabel("Drawdown (%)")
ax.set_title("Drawdown", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, ncol=2, loc="lower left"); ax.grid(True, alpha=0.3)

# ③ CAGR & Sharpe bar  (방어자산 QQQ vs JEPQ)
ax = fig.add_subplot(gs[2, 0])
names_plot = list(all_series.keys())
cagrs  = [metrics[n]["cagr"]*100  for n in names_plot]
sharpe = [metrics[n]["sharpe"]    for n in names_plot]
x = np.arange(len(names_plot))
colors_bar = [COLORS[n] for n in names_plot]
ax.bar(x - 0.2, cagrs, 0.38, color=colors_bar, alpha=0.85, label="CAGR %", edgecolor="white", lw=0.5)
ax2 = ax.twinx()
ax2.plot(x, sharpe, "o--", color="#F59E0B", ms=8, lw=1.5, label="Sharpe")
for i, (c, sh) in enumerate(zip(cagrs, sharpe)):
    ax.text(i-0.2, max(c,0)+0.2, f"{c:+.1f}", ha="center", fontsize=7, color="white" if c<0 else "#111")
ax.set_xticks(x); ax.set_xticklabels([n.replace(" (trail-8%)", "").replace(": ", "\n")
                                        for n in names_plot], fontsize=7.5, rotation=15, ha="right")
ax.set_ylabel("CAGR (%)"); ax2.set_ylabel("Sharpe", color="#F59E0B")
ax.set_title("CAGR & Sharpe", fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y"); ax.axhline(0, color="black", lw=0.6)

# ④ MDD & Ulcer bar
ax = fig.add_subplot(gs[2, 1])
mdds   = [abs(metrics[n]["mdd"])*100  for n in names_plot]
ulcers = [metrics[n]["ulcer"]         for n in names_plot]
ax.bar(x - 0.2, mdds, 0.38, color=colors_bar, alpha=0.85, label="|MDD| %", edgecolor="white", lw=0.5)
ax2 = ax.twinx()
ax2.plot(x, ulcers, "s--", color="#EF4444", ms=8, lw=1.5, label="Ulcer")
ax.set_xticks(x); ax.set_xticklabels([n.replace(" (trail-8%)", "").replace(": ", "\n")
                                        for n in names_plot], fontsize=7.5, rotation=15, ha="right")
ax.set_ylabel("|MDD| (%)"); ax2.set_ylabel("Ulcer", color="#EF4444")
ax.set_title("|MDD| & Ulcer  (낮을수록 좋음)", fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

# ⑤ Bootstrap win-prob bars (vs B1:QQQ)
ax = fig.add_subplot(gs[3, :])
bs_names = [n for n in names_plot if n != "B1: QQQ only"]
bs_x = np.arange(len(bs_names)); bw = 0.20

def wp(n, key):
    return bs[n][key]["prob_better"] * 100

for i, (mk, mn, mc) in enumerate([
    ("delta_sharpe",  "P(ΔSh>0)",    "#3B82F6"),
    ("delta_cagr",    "P(ΔCAGR>0)",  "#10B981"),
    ("delta_sortino", "P(ΔSortino>0)","#8B5CF6"),
    ("delta_calmar",  "P(ΔCalmar>0)", "#F59E0B"),
]):
    vals = [wp(n, mk) for n in bs_names]
    ax.bar(bs_x + (i-1.5)*bw, vals, bw, label=mn, color=mc, alpha=0.85)
    for j, v in enumerate(vals):
        if v > 70 or v < 30:
            sig = bs[bs_names[j]][mk]
            star = "★" if sig["p_value"] < 0.05 else ""
            ax.text(bs_x[j]+(i-1.5)*bw, v+1, f"{v:.0f}%{star}", ha="center", fontsize=6.5)

ax.axhline(50, color="black", ls="--", lw=0.8, label="random (50%)")
ax.set_xticks(bs_x)
ax.set_xticklabels([n.replace(" (trail-8%)", "") for n in bs_names],
                   fontsize=8.5, rotation=15, ha="right")
ax.set_ylabel("QQQ를 이길 확률 (%)")
ax.set_title("Paired bootstrap: 각 메트릭에서 B1(QQQ) 대비 우위 확률  (500회)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9, ncol=5, loc="upper right")
ax.set_ylim(0, 115); ax.grid(True, alpha=0.3, axis="y")

plt.savefig(OUT_DIR / "jepq_backtest.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 에피소드 비교 시각화 (S4 def.QQQ vs S4 def.JEPQ) ─────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(15, 5.5))
fig2.suptitle("S4 전략 방어자산 비교: QQQ vs JEPQ 이벤트 (trail=-8%)",
              fontsize=12, fontweight="bold")

for ax, ev_df, strat_s, bench_s, title in [
    (axes[0], ev_qqq,  strat_qqq,  b_qqq,  "방어자산 QQQ"),
    (axes[1], ev_jepq, strat_jepq, b_jepq, "방어자산 JEPQ"),
]:
    ax.plot(strat_s.index, strat_s/strat_s.iloc[0]*100,
            color="#DC2626" if "QQQ" in title else "#7C3AED", lw=2.0, label=f"S4 ({title})")
    ax.plot(bench_s.index, bench_s/bench_s.iloc[0]*100,
            color="#1F2937", lw=1.2, alpha=0.6, ls="--",
            label="QQQ" if "QQQ" in title else "JEPQ")
    ax.plot(b_qqq.index, b_qqq/b_qqq.iloc[0]*100,
            color="#9CA3AF", lw=1.0, alpha=0.5, label="B1 QQQ")

    # 이벤트 마킹
    etype_colors = {
        "TO_HALF_ATTACK": "#F59E0B",
        "TO_FULL_ATTACK": "#EF4444",
        "TRAIL_EXIT":     "#10B981",
        "TRAIL_FLOOR":    "#60A5FA",
        "HALF_STOP":      "#DC2626",
    }
    etype_markers = {
        "TO_HALF_ATTACK": "^", "TO_FULL_ATTACK": "v",
        "TRAIL_EXIT": "o", "TRAIL_FLOOR": "s", "HALF_STOP": "X",
    }
    if not ev_df.empty:
        for etype, ecolor in etype_colors.items():
            sub = ev_df[ev_df["type"] == etype]
            if sub.empty:
                continue
            ys = []
            for d in sub["Date"]:
                if d in strat_s.index:
                    ys.append(strat_s.loc[d] / strat_s.iloc[0] * 100)
                else:
                    ys.append(None)
            ys = [y for y in ys if y is not None]
            dates = [d for d, y in zip(sub["Date"], sub["Date"].map(
                lambda d: strat_s.loc[d] / strat_s.iloc[0]*100 if d in strat_s.index else None)) if y is not None]
            if ys:
                ax.scatter([d for d in sub["Date"] if d in strat_s.index],
                           ys, color=ecolor, s=60,
                           marker=etype_markers.get(etype, "o"),
                           zorder=5, label=etype, alpha=0.8)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.set_ylabel("Index (log)")
    m = metrics["S4 def.QQQ  (trail-8%)" if "QQQ" in title else "S4 def.JEPQ (trail-8%)"]
    ax.set_title(f"{title}  |  CAGR={m['cagr']*100:+.1f}%  Sh={m['sharpe']:.2f}  MDD={m['mdd']*100:+.1f}%",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left"); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "jepq_backtest_events.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

# ── JSON 저장 ────────────────────────────────────────────────────────────────
def _j(o):
    if isinstance(o, dict): return {k: _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_j(x) for x in o]
    if isinstance(o, (np.floating, float)): return float(o)
    if isinstance(o, (np.integer, int)): return int(o)
    return o

out = {
    "period": {"start": str(common[0].date()), "end": str(common[-1].date()),
               "n_days": int(len(common))},
    "params": PARAMS,
    "metrics": {name: m for name, m in metrics.items()},
    "bootstrap_vs_qqq": {name: {k: v for k, v in b.items() if k != "raw"}
                          for name, b in bs.items()},
    "events": {
        "S4_def_QQQ":  ev_qqq.to_dict(orient="records") if not ev_qqq.empty else [],
        "S4_def_JEPQ": ev_jepq.to_dict(orient="records") if not ev_jepq.empty else [],
    }
}
with open(OUT_DIR / "jepq_backtest.json", "w") as f:
    json.dump(_j(out), f, indent=2, default=str)

print(f"\n  PNG → {OUT_DIR}/jepq_backtest.png")
print(f"  PNG → {OUT_DIR}/jepq_backtest_events.png")
print(f"  JSON → {OUT_DIR}/jepq_backtest.json")
print("\n[완료]")
