"""
monte_carlo_compare.py
======================
두 가지 파라미터 설정에 대한 Monte Carlo 비교

Base  : trail=-15%, hf=50%  (Anchor, 통일 트레일)
Case2 : trail=-15%, hf=25%  (Half attack 25%로 축소)

설계: 전체 기간 1회 실행 → ground truth → 1000회 랜덤 슬라이스
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

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
qqq  = load_extended_daily("QQQ")
qld  = load_extended_daily("QLD")
tqqq = load_extended_daily("TQQQ")
common = qqq.index.intersection(qld.index).intersection(tqqq.index)
common = common[(common >= "2002-01-01") & (common <= "2026-04-30")]
Q = qqq.loc[common]; L = qld.loc[common]; T = tqqq.loc[common]
N = len(common)

# ── 파라미터 설정 ─────────────────────────────────────────────────────────────
FIXED = dict(
    shallow_drop   = -0.10,
    deep_drop      = -0.20,
    shallow_bounce = -0.05,
    deep_bounce    = -0.10,
    half_stop      = -0.15,
    full_frac      = 1.00,
)

CASES = {
    "Base (trail=-15%, hf=50%)":  {**FIXED, "trailing_stop_pct": -0.15, "half_frac": 0.50},
    "Case2 (trail=-15%, hf=25%)": {**FIXED, "trailing_stop_pct": -0.15, "half_frac": 0.25},
}
CASE_COLORS = {
    "Base (trail=-15%, hf=50%)":  "#6B7280",
    "Case2 (trail=-15%, hf=25%)": "#2563EB",
}

print("=" * 76)
print("Monte Carlo 비교  (N=1000, 최소 2년)")
print(f"  기간: {common[0].date()} ~ {common[-1].date()}  ({N:,}일)")
print("=" * 76)

# ── STEP 1: 각 케이스 전체 기간 실행 (ground truth) ──────────────────────────
print()
full_ports = {}
for label, params in CASES.items():
    s, ev = strategy_s4_trailing(Q, L, T, 100_000, **params)
    m = full_metrics(s)
    full_ports[label] = s
    n_hs = int((ev["type"] == "HALF_STOP").sum()) if not ev.empty else 0
    n_fa = int((ev["type"] == "TO_FULL_ATTACK").sum()) if not ev.empty else 0
    n_ha = int((ev["type"] == "TO_HALF_ATTACK").sum()) if not ev.empty else 0
    print(f"  [{label}]")
    print(f"    CAGR={m['cagr']*100:>+6.2f}%  Sh={m['sharpe']:.3f}  "
          f"Sortino={m['sortino']:.3f}  MDD={m['mdd']*100:>+6.2f}%  Ulcer={m['ulcer']:.2f}")
    print(f"    이벤트: HALF_ATTACK={n_ha}건  FULL_ATTACK={n_fa}건  HALF_STOP={n_hs}건")
    print()

# 벤치마크
full_b1 = pd.Series(100_000 / Q["Close"].iloc[0] * Q["Close"].values, index=common)

# ── STEP 2: 1000회 랜덤 슬라이스 ─────────────────────────────────────────────
print("  [STEP 2] 1000개 랜덤 구간 슬라이스...")
MIN_DAYS = 2 * 252
N_ITER   = 1000
RNG      = np.random.default_rng(42)   # 동일 seed → 동일 구간 → 직접 비교 가능

# 랜덤 인덱스 사전 생성 (모든 케이스가 동일 구간 사용)
max_start = N - MIN_DAYS - 1
si_arr = RNG.integers(0, max_start + 1, size=N_ITER)
ei_arr = np.array([
    int(RNG.integers(si + MIN_DAYS, N)) for si in si_arr
])

def run_mc(full_port, label):
    recs = []
    for k in range(N_ITER):
        si, ei = si_arr[k], ei_arr[k]
        ws  = full_port.iloc[si:ei+1] / full_port.iloc[si]
        wb1 = full_b1.iloc[si:ei+1]   / full_b1.iloc[si]
        ms  = full_metrics((ws  * 100_000).rename("p"))
        mb1 = full_metrics((wb1 * 100_000).rename("p"))
        dur = (common[ei] - common[si]).days / 365.25
        recs.append({
            "dur_yr":    dur,
            "s_cagr":    ms["cagr"],   "s_sharpe":  ms["sharpe"],
            "s_sortino": ms["sortino"],"s_mdd":     ms["mdd"],
            "s_ulcer":   ms["ulcer"],
            "b1_cagr":   mb1["cagr"],  "b1_sharpe": mb1["sharpe"],
            "b1_sortino":mb1["sortino"],"b1_mdd":   mb1["mdd"],
            "b1_ulcer":  mb1["ulcer"],
        })
    d = pd.DataFrame(recs)
    d["dcagr"]    = (d["s_cagr"]    - d["b1_cagr"])    * 100
    d["dsharpe"]  =  d["s_sharpe"]  - d["b1_sharpe"]
    d["dsortino"] =  d["s_sortino"] - d["b1_sortino"]
    d["dmdd"]     = (d["s_mdd"]     - d["b1_mdd"])     * 100  # MDD 음수 저장: 양수=전략이 덜 깊은 DD
    d["dulcer"]   =  d["s_ulcer"]   - d["b1_ulcer"]           # 음수=개선
    return d

t0 = time.time()
mc = {}
for label in CASES:
    mc[label] = run_mc(full_ports[label], label)
    print(f"    {label}: 완료")
print(f"  ({time.time()-t0:.1f}s)")

# ── STEP 3: 결과 요약 출력 ───────────────────────────────────────────────────
METRICS_DEF = [
    ("dcagr",    "ΔCAGR (%p)",    True),
    ("dsharpe",  "ΔSharpe",       True),
    ("dsortino", "ΔSortino",      True),
    ("dmdd",     "ΔMDD (%p)",      True),   # 음수 MDD: Δ>0 → 전략 |DD| < 벤치
    ("dulcer",   "ΔUlcer",        False),
]

print(f"\n{'='*90}")
print("  [vs QQQ B&H 상대 메트릭 요약]")
print("  승률: ΔCAGR·ΔSh·ΔSo·ΔMDD → P(Δ>0) | ΔUlcer → P(Δ<0)  (MDD 음수 저장)")
print(f"{'='*90}")
print(f"\n  {'메트릭':14s}  {'지표':10s}  ", end="")
for label in CASES:
    short = label[:25]
    print(f"  {short:25s}", end="")
print()
print("  " + "-"*120)

for mkey, mlabel, higher in METRICS_DEF:
    for stat, stat_lbl in [("median", "중앙값"), ("mean", "평균"), ("win%", "승률")]:
        print(f"  {mlabel:14s}  {stat_lbl:8s}  ", end="")
        for label in CASES:
            d = mc[label]
            if stat == "median":
                v = d[mkey].median()
                print(f"  {v:>+24.3f}         ", end="")
            elif stat == "mean":
                v = d[mkey].mean()
                print(f"  {v:>+24.3f}         ", end="")
            elif stat == "win%":
                if higher:
                    wr = (d[mkey] > 0).mean() * 100
                else:
                    wr = (d[mkey] < 0).mean() * 100
                print(f"  {wr:>23.1f}%         ", end="")
        print()
    print("  " + "-"*120)

# 케이스간 직접 비교 (Case2 vs Base)
print(f"\n{'='*90}")
print("  [Case2 vs Base  직접 차이]")
print(f"{'='*90}")
print(f"\n  {'메트릭':14s}  {'지표':10s}  {'Case2-Base (hf 25%)':>28s}")
print("  " + "-"*80)
for mkey, mlabel, higher in METRICS_DEF:
    base_med  = mc["Base (trail=-15%, hf=50%)"][mkey].median()
    c2_med    = mc["Case2 (trail=-15%, hf=25%)"][mkey].median()
    base_wr   = ((mc["Base (trail=-15%, hf=50%)"][mkey] > 0 if higher
                  else mc["Base (trail=-15%, hf=50%)"][mkey] < 0)).mean() * 100
    c2_wr     = ((mc["Case2 (trail=-15%, hf=25%)"][mkey] > 0 if higher
                  else mc["Case2 (trail=-15%, hf=25%)"][mkey] < 0)).mean() * 100
    delta_c2 = c2_med - base_med
    wr_delta_c2 = c2_wr - base_wr
    better_c2 = "↑" if (higher and delta_c2 > 0) or (not higher and delta_c2 < 0) else "↓"
    print(f"  {mlabel:14s}  {'중앙값차':8s}  {delta_c2:>+20.3f} {better_c2}")
    print(f"  {'':14s}  {'승률차':8s}  {wr_delta_c2:>+19.1f}%")

# 구간 길이별 ΔSharpe
print(f"\n{'='*90}")
print("  [구간 길이별 ΔSharpe 중앙값 vs QQQ]")
print(f"{'='*90}")
bins = [(2, 3, "2-3년"), (3, 5, "3-5년"), (5, 7, "5-7년"), (7, 10, "7-10년"), (10, 30, "10년+")]
print(f"\n  {'구간':8s}  {'건수':>5s}  ", end="")
for label in CASES:
    short = label.split("(")[1].rstrip(")")
    print(f"  {short:22s}", end="")
print()
print("  " + "-"*100)
for lo, hi, lbl in bins:
    print(f"  {lbl:8s}", end="")
    for i, (label, d) in enumerate(mc.items()):
        mask = (d["dur_yr"] >= lo) & (d["dur_yr"] < hi)
        sub = d[mask]
        if i == 0:
            print(f"  {len(sub):>5}", end="")
        if len(sub) < 5:
            print(f"  {'—':>22s}", end="")
        else:
            sh_med = sub["dsharpe"].median()
            sh_wr  = (sub["dsharpe"] > 0).mean() * 100
            print(f"  {sh_med:>+7.3f} ({sh_wr:>4.0f}%win)       ", end="")
    print()

# ── 시각화 ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 20))
gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.30,
                        height_ratios=[1.2, 1.1, 1.1, 1.0, 1.1])
fig.suptitle(
    "Monte Carlo 파라미터 비교  (N=1000, 최소 2년, 동일 랜덤 구간)\n"
    "vs QQQ B&H | Base(trail-15%,hf50%) | Case2(trail-15%,hf25%)",
    fontsize=12, fontweight="bold")

# ① 전체 기간 자산곡선
ax = fig.add_subplot(gs[0, :])
ax.plot(full_b1.index, full_b1 / full_b1.iloc[0] * 100,
        label="B1 QQQ", color="#1F2937", lw=1.2, alpha=0.7, ls="--")
for label, port in full_ports.items():
    ax.plot(port.index, port / port.iloc[0] * 100,
            label=label, color=CASE_COLORS[label], lw=1.8, alpha=0.9)
ax.set_yscale("log")
ax.set_ylabel("Index (start=100, log)")
ax.set_title("전체 기간 자산곡선 비교 (2002~2026)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="upper left"); ax.grid(True, alpha=0.3)

# ② ΔSharpe 분포 비교
ax = fig.add_subplot(gs[1, 0])
for label, d in mc.items():
    vals = d["dsharpe"]
    wr = (vals > 0).mean() * 100
    med = vals.median()
    ax.hist(vals, bins=50, alpha=0.5,
            label=f"{label.split('(')[1].rstrip(')')}  win={wr:.0f}%  med={med:+.3f}",
            color=CASE_COLORS[label], density=True, edgecolor="white", lw=0.3)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("ΔSharpe (vs QQQ)"); ax.set_ylabel("density")
ax.set_title("ΔSharpe 분포 (vs QQQ)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)

# ③ ΔCAGR 분포 비교
ax = fig.add_subplot(gs[1, 1])
for label, d in mc.items():
    vals = d["dcagr"]
    wr = (vals > 0).mean() * 100
    med = vals.median()
    ax.hist(vals, bins=50, alpha=0.5,
            label=f"{label.split('(')[1].rstrip(')')}  win={wr:.0f}%  med={med:+.2f}%",
            color=CASE_COLORS[label], density=True, edgecolor="white", lw=0.3)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("ΔCAGR (%p, vs QQQ)"); ax.set_ylabel("density")
ax.set_title("ΔCAGR 분포 (vs QQQ)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)

# ④ ΔMDD 분포 (MDD 음수 저장: Δ>0 이면 전략 낙폭 절댓값↓)
ax = fig.add_subplot(gs[2, 0])
for label, d in mc.items():
    vals = d["dmdd"]
    wr = (vals > 0).mean() * 100
    med = vals.median()
    ax.hist(vals, bins=50, alpha=0.5,
            label=f"{label.split('(')[1].rstrip(')')}  개선={wr:.0f}%  med={med:+.2f}%",
            color=CASE_COLORS[label], density=True, edgecolor="white", lw=0.3)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("ΔMDD (%p, vs QQQ, 양수=전략 |DD|↓)"); ax.set_ylabel("density")
ax.set_title("ΔMDD 분포 (vs QQQ)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

# ⑤ ΔUlcer 분포 (음수 = 개선)
ax = fig.add_subplot(gs[2, 1])
for label, d in mc.items():
    vals = d["dulcer"]
    wr = (vals < 0).mean() * 100
    med = vals.median()
    ax.hist(vals, bins=50, alpha=0.5,
            label=f"{label.split('(')[1].rstrip(')')}  개선={wr:.0f}%  med={med:+.2f}",
            color=CASE_COLORS[label], density=True, edgecolor="white", lw=0.3)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("ΔUlcer (vs QQQ, 음수=개선)"); ax.set_ylabel("density")
ax.set_title("ΔUlcer 분포 (vs QQQ)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

# ⑥ 구간 길이별 ΔSharpe 라인
ax = fig.add_subplot(gs[3, :])
dur_bins = np.arange(2, 15, 0.5)
for label, d in mc.items():
    mids, sh_meds, sh_wrs = [], [], []
    for lo in dur_bins[:-1]:
        hi = lo + 0.5
        mask = (d["dur_yr"] >= lo) & (d["dur_yr"] < hi)
        sub = d[mask]
        if len(sub) < 8:
            continue
        mids.append((lo+hi)/2)
        sh_meds.append(sub["dsharpe"].median())
        sh_wrs.append((sub["dsharpe"] > 0).mean() * 100)
    ax.plot(mids, sh_wrs, lw=2.0, label=f"{label.split('(')[1].rstrip(')')} 승률",
            color=CASE_COLORS[label], alpha=0.9)
ax.axhline(50, color="black", ls=":", lw=0.8, alpha=0.7, label="50%")
ax.axhline(70, color="#9CA3AF", ls=":", lw=0.7, alpha=0.5)
ax.set_xlabel("구간 길이 (년)"); ax.set_ylabel("ΔSharpe > 0 비율 (%)")
ax.set_title("구간 길이별 Sharpe 승률 vs QQQ", fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, 105)

# ⑦ Box plot: 핵심 5개 메트릭 비교
ax = fig.add_subplot(gs[4, :])
cases_list = list(CASES.keys())
positions_base = np.arange(5) * 4
width = 0.8
metric_keys_plot = ["dcagr", "dsharpe", "dsortino", "dmdd", "dulcer"]
metric_labels_plot = ["ΔCAGR(%p)", "ΔSharpe", "ΔSortino", "ΔMDD(%p)", "ΔUlcer"]

# 각 메트릭을 정규화 (표준편차 기준)
for mi, (mkey, mlbl) in enumerate(zip(metric_keys_plot, metric_labels_plot)):
    base_pos = positions_base[mi]
    for ci, (label, d) in enumerate(mc.items()):
        pos = base_pos + (ci - 1) * width
        bplot = ax.boxplot(
            d[mkey], positions=[pos], widths=width*0.85,
            patch_artist=True, notch=True,
            boxprops=dict(facecolor=CASE_COLORS[label], alpha=0.7),
            medianprops=dict(color="white", lw=2),
            whiskerprops=dict(color=CASE_COLORS[label]),
            flierprops=dict(marker=".", ms=2, color=CASE_COLORS[label], alpha=0.3),
            capprops=dict(color=CASE_COLORS[label]),
        )

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(positions_base)
ax.set_xticklabels(metric_labels_plot, fontsize=10)
ax.set_ylabel("vs QQQ (상대값)")
ax.set_title("핵심 메트릭 분포 Boxplot (notch=95% CI of median)\n"
             "회색=Base(trail-15%,hf50%)  청=Case2(hf25%)",
             fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

# 범례
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=CASE_COLORS[l], alpha=0.7, label=l.split("(")[1].rstrip(")"))
                   for l in cases_list]
ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

plt.savefig(OUT_DIR / "monte_carlo_compare.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 최종 요약 테이블 ─────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("  [최종 요약 — vs QQQ B&H  (N=1000 랜덤 구간)]")
print(f"{'='*90}")
print(f"\n  {'메트릭':14s}  {'Base (trail-15%,hf50%)':>28s}  {'Case2 (trail-15%,hf25%)':>28s}")
print("  " + "-"*110)
for mkey, mlabel, higher in METRICS_DEF:
    row = f"  {mlabel:14s}"
    for label in CASES:
        d = mc[label]
        med = d[mkey].median()
        wr  = (d[mkey] > 0 if higher else d[mkey] < 0).mean() * 100
        sig = "★" if wr >= 80 else ("·" if wr >= 65 else " ")
        row += f"  {med:>+10.3f}  ({wr:>4.1f}%{sig})         "
    print(row)

print(f"\n  PNG → {OUT_DIR}/monte_carlo_compare.png")
print("\n[완료]")
