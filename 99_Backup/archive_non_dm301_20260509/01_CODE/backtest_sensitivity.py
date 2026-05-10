"""
전략4 파라미터 민감도 분석 — 재파라미터화 버전

파라미터 구조:
  b      = HALF 반등진입선          (기본: -5%)
  r1     = HALF 하락트리거 / 반등진입 (기본: 2)  → shallow_drop = b × r1
  r2     = FULL / HALF 배수          (기본: 2)  → deep_drop = b × r1 × r2
                                                   deep_bounce = b × r2
  trail  = 트레일링 스탑             (기본: -15%)

파생 (자유변수 아님):
  shallow_bounce = b
  shallow_drop   = b × r1
  deep_bounce    = b × r2
  deep_drop      = b × r1 × r2
  half_stop      = (shallow_drop + deep_drop) / 2
               = b × r1 × (1 + r2) / 2
"""

import sys
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = 'Apple SD Gothic Neo'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_switching import (
    load_yahoo_daily,
    strategy_s4_trailing,
    strategy_buy_and_hold,
    compute_statistics,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR  = BASE_DIR / "03_RESULT" / "sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────── 기본값 ───────────────────────────────────────────────
B_DEF     = -0.05   # HALF 반등진입선
R1_DEF    =  2.0    # 하락트리거 / 반등진입 비율
R2_DEF    =  2.0    # FULL / HALF 배수
TRAIL_DEF = -0.15   # 트레일링 스탑

PARAM_LABELS = {
    "b":     "b: HALF 반등진입선 (기본 -5%)",
    "r1":    "r1: 하락/반등 비율 (기본 2.0)",
    "r2":    "r2: FULL/HALF 배수 (기본 2.0)",
    "trail": "trail: 트레일링 스탑 (기본 -15%)",
}

# ─────────────────────── 파라미터 전개 함수 ──────────────────────────────────
def derive(b: float, r1: float, r2: float, trail: float) -> dict:
    """4개 파라미터 → 전략4의 6개 원시 파라미터 전개."""
    shallow_bounce = b
    shallow_drop   = b * r1
    deep_bounce    = b * r2
    deep_drop      = b * r1 * r2
    half_stop      = (shallow_drop + deep_drop) / 2
    return dict(
        shallow_drop   = shallow_drop,
        shallow_bounce = shallow_bounce,
        deep_drop      = deep_drop,
        deep_bounce    = deep_bounce,
        half_stop      = half_stop,
        trailing_stop  = trail,
    )

# ─────────────────────── 단일 백테스트 실행 ──────────────────────────────────
def run_s4(qqq, qld, tqqq, p: dict) -> dict:
    port, events = strategy_s4_trailing(
        qqq, qld, tqqq,
        initial_capital   = 100_000,
        shallow_drop      = p["shallow_drop"],
        deep_drop         = p["deep_drop"],
        shallow_bounce    = p["shallow_bounce"],
        deep_bounce       = p["deep_bounce"],
        half_stop         = p["half_stop"],
        trailing_stop_pct = p["trailing_stop"],
    )
    st = compute_statistics(port, "S4")
    return {
        "cagr":   float(st["CAGR"].rstrip("%")) / 100,
        "mdd":    float(st["MDD"].rstrip("%")) / 100,
        "sharpe": float(st["샤프 비율"]),
        "n_evt":  len(events),
    }

# ═══════════════════════════ 데이터 로드 ═════════════════════════════════════
print("=" * 65)
print("데이터 로드 중...")
qqq  = load_yahoo_daily("QQQ")
qld  = load_yahoo_daily("QLD")
tqqq = load_yahoo_daily("TQQQ")

common = qqq.index.intersection(qld.index).intersection(tqqq.index)
qqq  = qqq.loc[common]
qld  = qld.loc[common]
tqqq = tqqq.loc[common]
print(f"공통 기간: {common[0].date()} ~ {common[-1].date()}  ({len(common)} 영업일)\n")

# 벤치마크 S1
bh_port   = strategy_buy_and_hold(qqq, 100_000, series_name="S1")
bh_st     = compute_statistics(bh_port, "S1")
bh_cagr   = float(bh_st["CAGR"].rstrip("%")) / 100
bh_mdd    = float(bh_st["MDD"].rstrip("%")) / 100
bh_sharpe = float(bh_st["샤프 비율"])
print(f"[벤치마크 S1] CAGR={bh_cagr:.2%}  MDD={bh_mdd:.2%}  Sharpe={bh_sharpe:.2f}")

# 기본값 S4
def_p     = derive(B_DEF, R1_DEF, R2_DEF, TRAIL_DEF)
def_stats = run_s4(qqq, qld, tqqq, def_p)
print(f"[기본값  S4] CAGR={def_stats['cagr']:.2%}  MDD={def_stats['mdd']:.2%}  Sharpe={def_stats['sharpe']:.2f}")
print(f"             파생: shallow_drop={def_p['shallow_drop']:.0%} bounce={def_p['shallow_bounce']:.0%} "
      f"deep_drop={def_p['deep_drop']:.0%} deep_bounce={def_p['deep_bounce']:.0%} half_stop={def_p['half_stop']:.0%}")
print()

# ═══════════════════════════ OAT 분석 ════════════════════════════════════════
OAT_RANGES = {
    "b":     np.round(np.arange(-0.01, -0.135, -0.005), 4),  # -1% ~ -13%
    "r1":    np.round(np.arange(1.1, 5.1, 0.2), 2),          # 1.1 ~ 5.0
    "r2":    np.round(np.arange(1.1, 5.1, 0.2), 2),          # 1.1 ~ 5.0
    "trail": np.round(np.arange(-0.02, -0.255, -0.01), 3),   # -2% ~ -25%
}

print("OAT 민감도 분석 실행 중...")
t0 = time.time()

oat_results: dict[str, list[dict]] = {}
for pk, vals in OAT_RANGES.items():
    rows = []
    for v in vals:
        b, r1, r2, trail = B_DEF, R1_DEF, R2_DEF, TRAIL_DEF
        if pk == "b":     b     = float(v)
        elif pk == "r1":  r1    = float(v)
        elif pk == "r2":  r2    = float(v)
        elif pk == "trail": trail = float(v)

        # 유효성: r1 > 1, r2 > 1, b < 0, trail < 0
        if r1 <= 1.0 or r2 <= 1.0 or b >= 0 or trail >= 0:
            continue
        p = derive(b, r1, r2, trail)
        try:
            r = run_s4(qqq, qld, tqqq, p)
            # x축 라벨용 값
            if pk == "b":
                xv = v * 100           # % 단위
            elif pk in ("r1", "r2"):
                xv = v                 # 배수 단위
            else:
                xv = v * 100           # % 단위
            rows.append({"xv": xv, **r})
        except Exception as e:
            pass
    oat_results[pk] = rows
    print(f"  {PARAM_LABELS[pk]}: {len(rows)}개 유효 지점")

print(f"OAT 완료 ({time.time()-t0:.1f}s)\n")

# ─── OAT 차트 (4파라미터 × 3지표) ───────────────────────────────────────────
METRICS = [
    ("cagr",   "CAGR",    "#2563EB", bh_cagr,    100),
    ("mdd",    "MDD",     "#DC2626", None,        100),
    ("sharpe", "샤프 비율", "#16A34A", bh_sharpe,   1),
]

fig, axes = plt.subplots(4, 3, figsize=(13, 14))
fig.suptitle(
    "전략4 파라미터 민감도 — OAT\n"
    "b=-5%, r1=2.0, r2=2.0, trail=-15%\n"
    "파생: shallow_drop=b×r1, deep_drop=b×r1×r2, deep_bounce=b×r2, half_stop=(shallow+deep)/2\n"
    "수직 점선=기본값 | 황색 점선=S1 기준",
    fontsize=10, fontweight="bold", y=1.01
)

XLABELS = {
    "b":     "b 값 (%)",
    "r1":    "r1 배수",
    "r2":    "r2 배수",
    "trail": "trail 값 (%)",
}

for ri, pk in enumerate(OAT_RANGES):
    rows = oat_results[pk]
    xs = [r["xv"] for r in rows]
    def_xv = (B_DEF*100 if pk=="b" else R1_DEF if pk=="r1" else R2_DEF if pk=="r2" else TRAIL_DEF*100)

    for ci, (mk, mlabel, mcolor, bench, scale) in enumerate(METRICS):
        ax = axes[ri, ci]
        ys = [r[mk] * scale for r in rows]
        ax.plot(xs, ys, "o-", color=mcolor, lw=1.8, ms=4, zorder=3)
        ax.axvline(x=def_xv, color="#9CA3AF", ls="--", lw=1.2)
        if bench is not None:
            ax.axhline(y=bench * scale, color="#F59E0B", ls=":", lw=1.2, alpha=0.8)
        if ri == 0:
            ax.set_title(mlabel, fontsize=10, fontweight="bold")
        if ci == 0:
            ax.set_ylabel(PARAM_LABELS[pk], fontsize=8)
        ax.set_xlabel(XLABELS[pk], fontsize=7.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, alpha=0.3)
        if mk in ("cagr", "mdd"):
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda y, _: f"{y:.1f}%"))
        else:
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda y, _: f"{y:.2f}"))

plt.tight_layout()
oat_path = OUT_DIR / "oat_sensitivity_v2.png"
plt.savefig(oat_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"OAT 차트 저장 → {oat_path.name}")

# ═══════════════════════════ 2D 히트맵 분석 ══════════════════════════════════
HEATMAP_PAIRS = [
    # (pk_x, pk_y, x_vals, y_vals)
    ("r1", "r2",
     np.round(np.arange(1.1, 4.2, 0.3), 2),
     np.round(np.arange(1.1, 4.2, 0.3), 2)),
    ("b", "trail",
     np.round(np.arange(-0.01, -0.125, -0.01), 3),
     np.round(np.arange(-0.02, -0.235, -0.02), 3)),
    ("b", "r1",
     np.round(np.arange(-0.01, -0.125, -0.01), 3),
     np.round(np.arange(1.1, 4.5, 0.3), 2)),
    ("r2", "trail",
     np.round(np.arange(1.1, 4.2, 0.3), 2),
     np.round(np.arange(-0.02, -0.235, -0.02), 3)),
]

print("\n2D 히트맵 분석 실행 중...")
t0 = time.time()

hm_data = {}
for px, py, xv, yv in HEATMAP_PAIRS:
    sharpe_g = np.full((len(yv), len(xv)), np.nan)
    cagr_g   = np.full((len(yv), len(xv)), np.nan)
    n_valid  = 0
    for xi, xval in enumerate(xv):
        for yi, yval in enumerate(yv):
            b, r1, r2, trail = B_DEF, R1_DEF, R2_DEF, TRAIL_DEF
            if px == "b":     b     = float(xval)
            elif px == "r1":  r1    = float(xval)
            elif px == "r2":  r2    = float(xval)
            elif px == "trail": trail = float(xval)
            if py == "b":     b     = float(yval)
            elif py == "r1":  r1    = float(yval)
            elif py == "r2":  r2    = float(yval)
            elif py == "trail": trail = float(yval)

            if r1 <= 1.0 or r2 <= 1.0 or b >= 0 or trail >= 0:
                continue
            p = derive(b, r1, r2, trail)
            try:
                r = run_s4(qqq, qld, tqqq, p)
                sharpe_g[yi, xi] = r["sharpe"]
                cagr_g[yi, xi]   = r["cagr"] * 100
                n_valid += 1
            except Exception:
                pass
    hm_data[(px, py)] = (xv, yv, sharpe_g, cagr_g)
    print(f"  ({px} × {py}): {n_valid} 유효 지점 / {len(xv)*len(yv)} 격자")

print(f"히트맵 완료 ({time.time()-t0:.1f}s)\n")

# ─── 히트맵 차트 ────────────────────────────────────────────────────────────
def fmt_axis(pk, vals):
    if pk in ("b", "trail"):
        return [f"{v*100:.1f}" for v in vals]
    else:
        return [f"{v:.1f}" for v in vals]

def get_def(pk):
    return {"b": B_DEF, "r1": R1_DEF, "r2": R2_DEF, "trail": TRAIL_DEF}[pk]

fig, axes = plt.subplots(4, 2, figsize=(13, 22))
fig.suptitle(
    "전략4 2D 파라미터 격자 탐색 (샤프 비율 / CAGR)\n"
    "★ = 기본값  |  색: 녹색=높음, 적색=낮음",
    fontsize=11, fontweight="bold"
)

for ri, (px, py, _, _) in enumerate(HEATMAP_PAIRS):
    xv, yv, sharpe_g, cagr_g = hm_data[(px, py)]
    xticks = fmt_axis(px, xv)
    yticks = fmt_axis(py, yv)
    xu = "%" if px in ("b","trail") else "배수"
    yu = "%" if py in ("b","trail") else "배수"

    def_xi = int(np.argmin(np.abs(xv - get_def(px))))
    def_yi = int(np.argmin(np.abs(yv - get_def(py))))

    for ci, (grid, title, fmt_cb) in enumerate([
        (sharpe_g, "샤프 비율", "{:.2f}"),
        (cagr_g,   "CAGR (%)", "{:.1f}%"),
    ]):
        ax = axes[ri, ci]
        masked = np.ma.masked_invalid(grid)
        valid_vals = grid[~np.isnan(grid)]
        if len(valid_vals) == 0:
            ax.set_visible(False)
            continue
        vmin = np.nanpercentile(valid_vals, 5)
        vmax = np.nanpercentile(valid_vals, 95)

        im = ax.imshow(masked, cmap="RdYlGn", aspect="auto",
                       vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

        ax.plot(def_xi, def_yi, "k*", ms=14, zorder=5, label="기본값")
        ax.legend(loc="upper right", fontsize=8)

        ax.set_xticks(range(len(xv)))
        ax.set_xticklabels(xticks, rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(yv)))
        ax.set_yticklabels(yticks, fontsize=6.5)
        ax.set_xlabel(f"{PARAM_LABELS[px]} ({xu})", fontsize=8)
        ax.set_ylabel(f"{PARAM_LABELS[py]} ({yu})", fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")

plt.tight_layout()
hm_path = OUT_DIR / "heatmap_sensitivity_v2.png"
plt.savefig(hm_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"히트맵 저장 → {hm_path.name}")

# ═══════════════════════════ 민감도 요약 ═════════════════════════════════════
print()
print("=" * 72)
print("파라미터 민감도 요약 (샤프 비율 기준, 재파라미터화)")
print("=" * 72)
print(f"  기본 S4:  CAGR={def_stats['cagr']:.2%}  MDD={def_stats['mdd']:.2%}  Sharpe={def_stats['sharpe']:.2f}")
print(f"  벤치마크: CAGR={bh_cagr:.2%}  MDD={bh_mdd:.2%}  Sharpe={bh_sharpe:.2f}")
print()
print(f"  {'파라미터':<36} {'유효':>4}  {'Sharpe 범위':>14}  {'표준편차':>8}  {'S1 초과율':>8}")
print("  " + "─" * 77)

summary = []
for pk in OAT_RANGES:
    rows = oat_results[pk]
    if len(rows) < 2:
        continue
    sharpes = np.array([r["sharpe"] for r in rows])
    summary.append(dict(
        pk=pk,
        n=len(rows),
        smin=sharpes.min(),
        smax=sharpes.max(),
        srange=sharpes.max() - sharpes.min(),
        sstd=sharpes.std(),
        beat_pct=(sharpes > bh_sharpe).mean(),
    ))

summary.sort(key=lambda x: -x["sstd"])

for s in summary:
    beat_bar = "█" * int(s["beat_pct"] * 10) + "░" * (10 - int(s["beat_pct"] * 10))
    print(
        f"  {PARAM_LABELS[s['pk']]:<36} {s['n']:>4}  "
        f"{s['smin']:.2f}~{s['smax']:.2f} (폭:{s['srange']:.2f})"
        f"  {s['sstd']:>6.3f}  "
        f"  {beat_bar} {s['beat_pct']:.0%}"
    )

print("  " + "─" * 77)

max_sstd = max(s["sstd"] for s in summary)
avg_sstd = np.mean([s["sstd"] for s in summary])
min_beat = min(s["beat_pct"] for s in summary)

print(f"\n  [과적합 판단]")
print(f"  - 가장 민감한 파라미터 std: {max_sstd:.3f}")
print(f"  - 전체 파라미터 평균  std: {avg_sstd:.3f}")
print(f"  - 최소 S1 초과 비율:      {min_beat:.0%}")
if avg_sstd < 0.15 and min_beat > 0.55:
    print("  → 파라미터 범위 전반에서 안정적으로 S1 초과. 과적합 위험 낮음.")
elif avg_sstd < 0.25 and min_beat > 0.35:
    print("  → 일부 파라미터에 민감하나 전반적으로 안정적. 중간 수준.")
else:
    print("  → 특정 값에 민감하게 반응. 과적합 가능성 존재. 주의 필요.")

print("=" * 72)

# ─── 요약 차트 ───────────────────────────────────────────────────────────────
fig, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("전략4 민감도 요약 (재파라미터화)", fontsize=12, fontweight="bold")

labels_short = [
    {"b":"b (HALF 반등진입)", "r1":"r1 (하락/반등 비율)", "r2":"r2 (FULL/HALF 배수)", "trail":"trail (트레일링)"}[s["pk"]]
    for s in summary
]
stds  = [s["sstd"]     for s in summary]
beats = [s["beat_pct"]*100 for s in summary]

colors_std  = ["#EF4444" if st > 0.20 else "#F59E0B" if st > 0.10 else "#22C55E" for st in stds]
colors_beat = ["#22C55E" if b >= 70 else "#F59E0B" if b >= 50 else "#EF4444" for b in beats]

ax1, ax2 = axes2

bars1 = ax1.barh(labels_short, stds, color=colors_std, edgecolor="white", lw=0.5)
ax1.axvline(0.10, color="#94A3B8", ls="--", lw=1, label="낮음(0.10)")
ax1.axvline(0.20, color="#F59E0B", ls="--", lw=1, label="높음(0.20)")
ax1.set_xlabel("샤프 표준편차 (클수록 민감)", fontsize=9)
ax1.set_title("파라미터 민감도", fontsize=10, fontweight="bold")
ax1.legend(fontsize=8)
ax1.grid(True, axis="x", alpha=0.3)
for bar, st in zip(bars1, stds):
    ax1.text(st + 0.002, bar.get_y() + bar.get_height()/2, f"{st:.3f}", va="center", fontsize=8.5)

bars2 = ax2.barh(labels_short, beats, color=colors_beat, edgecolor="white", lw=0.5)
ax2.axvline(50, color="#94A3B8", ls="--", lw=1)
ax2.axvline(70, color="#22C55E", ls="--", lw=1, alpha=0.6)
ax2.set_xlabel("S1(QQQ 보유) 샤프 초과 비율 (%)", fontsize=9)
ax2.set_title("파라미터 범위 내 S1 대비 우수 비율", fontsize=10, fontweight="bold")
ax2.set_xlim(0, 110)
ax2.grid(True, axis="x", alpha=0.3)
for bar, b in zip(bars2, beats):
    ax2.text(b + 1, bar.get_y() + bar.get_height()/2, f"{b:.0f}%", va="center", fontsize=8.5)

plt.tight_layout()
sum_path = OUT_DIR / "sensitivity_summary_v2.png"
plt.savefig(sum_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"요약 차트 저장 → {sum_path.name}")
print(f"\n전체 결과: {OUT_DIR}")
