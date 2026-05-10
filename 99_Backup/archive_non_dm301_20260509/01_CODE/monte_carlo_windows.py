"""
monte_carlo_windows.py
======================
1000개 랜덤 구간 Monte Carlo 백테스트

설계
----
• 전체 기간 (2002-01-01 ~ 2026-04-30)에 대해 strategy_s4_trailing 단 1회 실행
  → 이 포트폴리오 시계열이 "ground truth"
  → 가짜 신호 없음 (ATH는 항상 실제 과거 최고점)

• 1000개 랜덤 구간: start ~ end, 최소 2년
  → 각 구간에서 strategy / 3개 벤치마크를 동일 날짜에 슬라이스
  → 상대 메트릭 (ΔCAGR, ΔSharpe, ΔSortino, ΔMDD, ΔUlcer) 계산

• 벤치마크:
  B1: QQQ only
  B2: QQQ/TQQQ 50/50  (drift, no rebalance)
  B3: TQQQ only

• 파라미터: Anchor + trail=-15%
"""

import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from backtest_switching import load_extended_daily, strategy_s4_trailing
from evaluation_metrics import full_metrics

OUT_DIR = Path("03_RESULT")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 데이터 ───────────────────────────────────────────────────────────────────
qqq  = load_extended_daily("QQQ")
qld  = load_extended_daily("QLD")
tqqq = load_extended_daily("TQQQ")
common = qqq.index.intersection(qld.index).intersection(tqqq.index)
START_FULL = pd.Timestamp("2002-01-01")
END_FULL   = common[-1]
common     = common[(common >= START_FULL) & (common <= END_FULL)]

Q_full = qqq.loc[common]
L_full = qld.loc[common]
T_full = tqqq.loc[common]
N = len(common)

print("=" * 72)
print("Monte Carlo Window Backtest  |  Anchor trail=-15%")
print(f"  전체 기간: {common[0].date()} ~ {common[-1].date()}  ({N:,}일)")
print(f"  랜덤 구간: 최소 2년, 1000회")
print("=" * 72)

# ── STEP 1: 전체 기간 단 1회 실행 → ground truth ─────────────────────────────
PARAMS = dict(
    shallow_drop      = -0.10,
    deep_drop         = -0.20,
    shallow_bounce    = -0.05,
    deep_bounce       = -0.10,
    half_stop         = -0.15,
    trailing_stop_pct = -0.15,
    half_frac         = 0.50,
    full_frac         = 1.00,
)

print(f"\n  [STEP 1] 전체 기간 strategy 실행 (ground truth)...")
t0 = time.time()
full_strat, full_events = strategy_s4_trailing(Q_full, L_full, T_full,
                                                100_000, **PARAMS)
print(f"  완료 ({time.time()-t0:.1f}s)  이벤트 수: {len(full_events)}건")
if not full_events.empty:
    for etype, cnt in full_events["type"].value_counts().items():
        print(f"    {etype:16s}: {cnt}건")

# 벤치마크 (전체)
full_b1 = pd.Series(100_000 / Q_full["Close"].iloc[0] * Q_full["Close"].values, index=common)
full_b3 = pd.Series(100_000 / T_full["Close"].iloc[0] * T_full["Close"].values, index=common)

# B2: QQQ/TQQQ 50/50 (no rebalance)
sha = 50_000 / Q_full["Close"].iloc[0]
shb = 50_000 / T_full["Close"].iloc[0]
full_b2 = pd.Series(sha * Q_full["Close"].values + shb * T_full["Close"].values, index=common)

# 전체 메트릭 확인
m_full_s = full_metrics(full_strat)
m_full_b1 = full_metrics(full_b1)
m_full_b2 = full_metrics(full_b2)
m_full_b3 = full_metrics(full_b3)

print(f"\n  전체 기간 성과:")
print(f"  {'전략':20s}  CAGR={m_full_s['cagr']*100:>+6.2f}%  "
      f"Sh={m_full_s['sharpe']:.2f}  MDD={m_full_s['mdd']*100:.2f}%  "
      f"Ulcer={m_full_s['ulcer']:.2f}")
print(f"  {'B1 QQQ':20s}  CAGR={m_full_b1['cagr']*100:>+6.2f}%  "
      f"Sh={m_full_b1['sharpe']:.2f}  MDD={m_full_b1['mdd']*100:.2f}%  "
      f"Ulcer={m_full_b1['ulcer']:.2f}")
print(f"  {'B2 QQQ/TQQQ 50/50':20s}  CAGR={m_full_b2['cagr']*100:>+6.2f}%  "
      f"Sh={m_full_b2['sharpe']:.2f}  MDD={m_full_b2['mdd']*100:.2f}%  "
      f"Ulcer={m_full_b2['ulcer']:.2f}")
print(f"  {'B3 TQQQ':20s}  CAGR={m_full_b3['cagr']*100:>+6.2f}%  "
      f"Sh={m_full_b3['sharpe']:.2f}  MDD={m_full_b3['mdd']*100:.2f}%  "
      f"Ulcer={m_full_b3['ulcer']:.2f}")

# ── STEP 2: 1000개 랜덤 구간 시뮬레이션 ──────────────────────────────────────
print(f"\n  [STEP 2] 1000개 랜덤 구간 슬라이스...")

MIN_DAYS = 2 * 252   # 최소 2년
N_ITER   = 1000
RNG      = np.random.default_rng(seed=42)

records = []
t0 = time.time()
for it in range(N_ITER):
    # 랜덤 start_idx: 최소 MIN_DAYS를 남겨두고
    max_start = N - MIN_DAYS - 1
    si = int(RNG.integers(0, max_start + 1))
    # 랜덤 end_idx: si+MIN_DAYS ~ N-1
    ei = int(RNG.integers(si + MIN_DAYS, N))

    s_date = common[si]
    e_date = common[ei]
    dur_yr = (e_date - s_date).days / 365.25

    # 슬라이스 (ground truth에서)
    ws  = full_strat.iloc[si:ei+1]
    wb1 = full_b1.iloc[si:ei+1]
    wb2 = full_b2.iloc[si:ei+1]
    wb3 = full_b3.iloc[si:ei+1]

    # 정규화 (시작=1)
    ws  = ws  / ws.iloc[0]
    wb1 = wb1 / wb1.iloc[0]
    wb2 = wb2 / wb2.iloc[0]
    wb3 = wb3 / wb3.iloc[0]

    # 포트폴리오 시리즈로 변환
    def to_port(s):
        return (s * 100_000).rename("p")

    ms  = full_metrics(to_port(ws))
    mb1 = full_metrics(to_port(wb1))
    mb2 = full_metrics(to_port(wb2))
    mb3 = full_metrics(to_port(wb3))

    records.append({
        "iter":    it,
        "start":   s_date,
        "end":     e_date,
        "dur_yr":  dur_yr,
        # Strategy absolute
        "s_cagr":    ms["cagr"],
        "s_sharpe":  ms["sharpe"],
        "s_sortino": ms["sortino"],
        "s_mdd":     ms["mdd"],
        "s_ulcer":   ms["ulcer"],
        # B1 QQQ
        "b1_cagr":    mb1["cagr"],
        "b1_sharpe":  mb1["sharpe"],
        "b1_sortino": mb1["sortino"],
        "b1_mdd":     mb1["mdd"],
        "b1_ulcer":   mb1["ulcer"],
        # B2
        "b2_cagr":    mb2["cagr"],
        "b2_sharpe":  mb2["sharpe"],
        "b2_sortino": mb2["sortino"],
        "b2_mdd":     mb2["mdd"],
        "b2_ulcer":   mb2["ulcer"],
        # B3
        "b3_cagr":    mb3["cagr"],
        "b3_sharpe":  mb3["sharpe"],
        "b3_sortino": mb3["sortino"],
        "b3_mdd":     mb3["mdd"],
        "b3_ulcer":   mb3["ulcer"],
    })

df = pd.DataFrame(records)
elapsed = time.time() - t0
print(f"  완료 ({elapsed:.1f}s)  구간 기간 분포:")
print(f"    min={df['dur_yr'].min():.1f}y  median={df['dur_yr'].median():.1f}y  "
      f"mean={df['dur_yr'].mean():.1f}y  max={df['dur_yr'].max():.1f}y")

# 상대값 계산
for bench in ["b1", "b2", "b3"]:
    df[f"d{bench}_cagr"]    = df["s_cagr"]    - df[f"{bench}_cagr"]
    df[f"d{bench}_sharpe"]  = df["s_sharpe"]  - df[f"{bench}_sharpe"]
    df[f"d{bench}_sortino"] = df["s_sortino"] - df[f"{bench}_sortino"]
    df[f"d{bench}_mdd"]     = df["s_mdd"]     - df[f"{bench}_mdd"]  # 양수=전략 MDD 덜 깊음(|DD|↓), 음수 저장값 기준
    df[f"d{bench}_ulcer"]   = df["s_ulcer"]   - df[f"{bench}_ulcer"]  # 음수=전략 Ulcer↓

# ── STEP 3: 결과 집계 ─────────────────────────────────────────────────────────
BENCH_LABELS = {
    "b1": "vs QQQ",
    "b2": "vs QQQ/TQQQ 50/50",
    "b3": "vs TQQQ",
}
METRICS = {
    "cagr":    ("ΔCAGR (%p)",    True,   100),  # higher_better, scale
    "sharpe":  ("ΔSharpe",       True,   1),
    "sortino": ("ΔSortino",      True,   1),
    "mdd":     ("ΔMDD (%p)",     True,   100),  # MDD 음수 저장: Δ>0 이면 전략이 덜 깊은 DD
    "ulcer":   ("ΔUlcer",        False,  1),
}

print(f"\n{'='*90}")
print(f"  [결과 집계] 벤치마크 대비 상대 메트릭 분포 (N={N_ITER} 랜덤 구간)")
print(f"{'='*90}")
print(f"\n  {'비교':22s}  {'메트릭':12s}  {'중앙값':>9s}  {'평균':>9s}  "
      f"{'5%ile':>9s}  {'95%ile':>9s}  {'승률':>8s}")
print("  (승률: ΔCAGR·ΔSh·ΔSo·ΔMDD → P(Δ>0) | ΔUlcer → P(Δ<0) ; MDD·벤치 모두 음수 저장)")
print("  " + "-"*100)

for bench in ["b1", "b2", "b3"]:
    print(f"\n  [{BENCH_LABELS[bench]}]")
    for mkey, (mlabel, higher_better, scale) in METRICS.items():
        col = f"d{bench}_{mkey}"
        vals = df[col] * scale
        median = vals.median()
        mean   = vals.mean()
        p5     = vals.quantile(0.05)
        p95    = vals.quantile(0.95)
        if higher_better:
            win_rate = (vals > 0).mean() * 100
            pval_str = f"{win_rate:.1f}%"
        else:
            # Ulcer만: 양수 지표·낮을수록 좋음 → Δ<0 이면 전략 우위
            win_rate = (vals < 0).mean() * 100
            pval_str = f"{win_rate:.1f}%"
        sig = "★" if win_rate > 75 or win_rate > 65 and abs(median) > 0.01 else ""
        print(f"  {BENCH_LABELS[bench]:22s}  {mlabel:12s}  "
              f"{median:>+8.3f}  {mean:>+9.3f}  {p5:>+9.3f}  {p95:>+9.3f}  "
              f"{'win':>5}{sig}={pval_str}")

# ── STEP 4: 구간 길이별 분석 ─────────────────────────────────────────────────
print(f"\n{'='*90}")
print("  [구간 길이별 승률 vs QQQ]")
print(f"{'='*90}")

bins = [(2,3), (3,5), (5,7), (7,10), (10,30)]
print(f"\n  {'구간':12s}  {'건수':>5s}  {'ΔSharpe중앙':>12s}  {'ΔCAGR중앙':>12s}  "
      f"{'Sharpe승률':>10s}  {'CAGR승률':>10s}  {'MDD개선률':>10s}")
print("  " + "-"*90)
for lo, hi in bins:
    mask = (df["dur_yr"] >= lo) & (df["dur_yr"] < hi)
    sub  = df[mask]
    if len(sub) < 5:
        continue
    n = len(sub)
    sh_med  = (sub["db1_sharpe"]).median()
    ca_med  = (sub["db1_cagr"] * 100).median()
    sh_wr   = (sub["db1_sharpe"] > 0).mean() * 100
    ca_wr   = (sub["db1_cagr"]   > 0).mean() * 100
    mdd_wr  = (sub["db1_mdd"]    > 0).mean() * 100   # MDD 음수: Δ>0 → 전략 |DD| 더 작음
    print(f"  {lo}-{hi}년:       {n:>5}  {sh_med:>+11.3f}  {ca_med:>+11.2f}%  "
          f"{sh_wr:>9.1f}%  {ca_wr:>9.1f}%  {mdd_wr:>9.1f}%")

# ── STEP 5: 연도별 시작 구간 분석 ────────────────────────────────────────────
print(f"\n{'='*90}")
print("  [시작 연도별 성과 (vs QQQ B&H)]")
print(f"{'='*90}")
print(f"\n  {'시작연도':10s}  {'건수':>5s}  {'ΔCAGR중앙':>10s}  {'ΔSharpe중앙':>12s}  "
      f"{'CAGR승률':>9s}  {'Sh승률':>7s}  {'MDD개선':>8s}")
print("  " + "-"*80)
df["start_yr"] = pd.to_datetime(df["start"]).dt.year
for yr in sorted(df["start_yr"].unique()):
    sub = df[df["start_yr"] == yr]
    if len(sub) < 3: continue
    ca_med = (sub["db1_cagr"]*100).median()
    sh_med = sub["db1_sharpe"].median()
    ca_wr  = (sub["db1_cagr"]>0).mean()*100
    sh_wr  = (sub["db1_sharpe"]>0).mean()*100
    mdd_wr = (sub["db1_mdd"]>0).mean()*100
    print(f"  {yr:<10}  {len(sub):>5}  {ca_med:>+9.2f}%  {sh_med:>+11.3f}  "
          f"{ca_wr:>8.1f}%  {sh_wr:>6.1f}%  {mdd_wr:>7.1f}%")

# ── 시각화 ───────────────────────────────────────────────────────────────────
BENCH_COLORS = {"b1": "#1F2937", "b2": "#6B7280", "b3": "#9CA3AF"}
BENCH_NAMES  = {"b1": "QQQ", "b2": "QQQ/TQQQ 50%", "b3": "TQQQ"}

fig = plt.figure(figsize=(18, 22))
gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.30,
                        height_ratios=[1, 1, 1, 1, 1])
fig.suptitle(
    f"Monte Carlo Window Backtest (N={N_ITER})  |  Anchor trail=−8%\n"
    f"전체 기간 {common[0].strftime('%Y-%m-%d')}~{common[-1].strftime('%Y-%m-%d')} → ground truth 슬라이스",
    fontsize=13, fontweight="bold")

# ① 구간 기간 분포
ax = fig.add_subplot(gs[0, 0])
ax.hist(df["dur_yr"], bins=40, color="#3B82F6", alpha=0.8, edgecolor="white", lw=0.5)
ax.axvline(df["dur_yr"].median(), color="#DC2626", ls="--", lw=1.5,
           label=f"median={df['dur_yr'].median():.1f}y")
ax.set_xlabel("구간 길이 (년)"); ax.set_ylabel("건수")
ax.set_title("랜덤 구간 기간 분포", fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ② Strategy vs B1-QQQ CAGR scatter
ax = fig.add_subplot(gs[0, 1])
sc = ax.scatter(df["b1_cagr"]*100, df["s_cagr"]*100, c=df["dur_yr"],
                cmap="plasma", s=15, alpha=0.5, edgecolors="none")
lo = min(df["b1_cagr"].min(), df["s_cagr"].min()) * 100 - 2
hi = max(df["b1_cagr"].max(), df["s_cagr"].max()) * 100 + 2
ax.plot([lo,hi],[lo,hi], "k--", lw=0.8, alpha=0.5, label="1:1")
ax.set_xlabel("QQQ CAGR (%)"); ax.set_ylabel("Strategy CAGR (%)")
ax.set_title("Strategy vs QQQ CAGR (색=기간 길이)", fontsize=10, fontweight="bold")
plt.colorbar(sc, ax=ax, label="기간(년)", fraction=0.05)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ③ ΔSharpe 분포 (세 벤치마크)
ax = fig.add_subplot(gs[1, :])
for bench in ["b1", "b2", "b3"]:
    col = f"d{bench}_sharpe"
    vals = df[col]
    label = f"vs {BENCH_NAMES[bench]}  (win={( vals>0).mean()*100:.1f}%, med={vals.median():+.3f})"
    ax.hist(vals, bins=50, alpha=0.45, label=label,
            color=BENCH_COLORS[bench], density=True, edgecolor="white", lw=0.3)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("ΔSharpe (strategy − benchmark)"); ax.set_ylabel("density")
ax.set_title("ΔSharpe 분포 (0 우측 = 전략 우위)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ④ ΔCAGR 분포
ax = fig.add_subplot(gs[2, :])
for bench in ["b1", "b2", "b3"]:
    col = f"d{bench}_cagr"
    vals = df[col] * 100
    label = f"vs {BENCH_NAMES[bench]}  (win={( vals>0).mean()*100:.1f}%, med={vals.median():+.2f}%p)"
    ax.hist(vals, bins=50, alpha=0.45, label=label,
            color=BENCH_COLORS[bench], density=True, edgecolor="white", lw=0.3)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("ΔCAGR (%p, strategy − benchmark)"); ax.set_ylabel("density")
ax.set_title("ΔCAGR 분포", fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ⑤ ΔMDD & ΔUlcer 분포 (vs QQQ)
ax = fig.add_subplot(gs[3, 0])
vals_mdd = df["db1_mdd"] * 100
ax.hist(vals_mdd, bins=50, color="#DC2626", alpha=0.7,
        density=True, edgecolor="white", lw=0.4)
ax.axvline(0, color="black", lw=0.8)
mdd_win = (vals_mdd > 0).mean()*100
ax.set_xlabel("ΔMDD (%p, strategy − QQQ)"); ax.set_ylabel("density")
ax.set_title(f"ΔMDD vs QQQ  (MDD 음수 저장: Δ>0이면 전략 |DD|↓, 개선율={mdd_win:.1f}%)",
             fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3)

ax = fig.add_subplot(gs[3, 1])
vals_ul = df["db1_ulcer"]
ax.hist(vals_ul, bins=50, color="#7C3AED", alpha=0.7,
        density=True, edgecolor="white", lw=0.4)
ax.axvline(0, color="black", lw=0.8)
ul_win = (vals_ul < 0).mean()*100
ax.set_xlabel("ΔUlcer (strategy − QQQ)"); ax.set_ylabel("density")
ax.set_title(f"ΔUlcer vs QQQ  (음수=전략 더 안정, 개선율={ul_win:.1f}%)",
             fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3)

# ⑥ 구간 길이별 ΔSharpe / ΔCAGR 승률 (vs QQQ)
ax = fig.add_subplot(gs[4, :])
dur_bins = np.arange(2, 15, 0.5)
sh_wr_by_dur  = []
ca_wr_by_dur  = []
so_wr_by_dur  = []
mdd_wr_by_dur = []
ul_wr_by_dur  = []
mid_bins      = []
for lo in dur_bins[:-1]:
    hi = lo + 0.5
    mask = (df["dur_yr"] >= lo) & (df["dur_yr"] < hi)
    sub = df[mask]
    if len(sub) < 10:
        continue
    mid_bins.append((lo+hi)/2)
    sh_wr_by_dur.append((sub["db1_sharpe"]>0).mean()*100)
    ca_wr_by_dur.append((sub["db1_cagr"]>0).mean()*100)
    so_wr_by_dur.append((sub["db1_sortino"]>0).mean()*100)
    mdd_wr_by_dur.append((sub["db1_mdd"]>0).mean()*100)
    ul_wr_by_dur.append((sub["db1_ulcer"]<0).mean()*100)

ax.plot(mid_bins, sh_wr_by_dur,  lw=1.8, label="P(ΔSharpe>0)",  color="#3B82F6", alpha=0.9)
ax.plot(mid_bins, ca_wr_by_dur,  lw=1.8, label="P(ΔCAGR>0)",    color="#10B981", alpha=0.9)
ax.plot(mid_bins, so_wr_by_dur,  lw=1.8, label="P(ΔSortino>0)", color="#8B5CF6", alpha=0.9)
ax.plot(mid_bins, mdd_wr_by_dur, lw=1.8, label="P(ΔMDD>0, |DD|↓)", color="#F59E0B", alpha=0.9, ls="--")
ax.plot(mid_bins, ul_wr_by_dur,  lw=1.8, label="P(ΔUlcer<0)",   color="#EF4444", alpha=0.9, ls="--")
ax.axhline(50, color="black", ls=":", lw=0.8, label="random (50%)")
ax.axhline(70, color="#9CA3AF", ls=":", lw=0.7, alpha=0.6)
ax.set_xlabel("구간 길이 (년)"); ax.set_ylabel("전략이 QQQ를 이길 확률 (%)")
ax.set_title("구간 길이에 따른 QQQ 대비 승률 (moving window, bin=0.5년)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9, ncol=3); ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

plt.savefig(OUT_DIR / "monte_carlo_windows.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 백분위 요약 ───────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("  [백분위 요약 vs QQQ]")
print(f"{'='*90}")
pcts = [5, 10, 25, 50, 75, 90, 95]
for mkey, (mlabel, higher_better, scale) in METRICS.items():
    col = f"db1_{mkey}"
    vals = df[col] * scale
    row = " ".join([f"{np.percentile(vals, p):>+7.3f}" for p in pcts])
    win_pct = (vals > 0 if higher_better else vals < 0).mean() * 100
    print(f"  {mlabel:14s}  [{' | '.join([f'p{p:2d}' for p in pcts])}]")
    print(f"               [{row}]   승률={win_pct:.1f}%")

print(f"\n  PNG → {OUT_DIR}/monte_carlo_windows.png")
print("\n[완료]")
