"""
backtest_optimize_s5.py
========================
Strategy 5 파라미터 최적화 (Differential Evolution).

IS  : 2002-10-01 ~ 2019-11-01
OOS : 2019-11-02 ~ 데이터 끝

데이터 : load_extended_daily (synthetic pre-inception + real)

파라미터 (5개)
--------------
  beta        : 진입 트리거  rebound < beta × drop         [0.10, 0.95]
  max_drop    : 포지션 사이징 min(drop/max_drop, 1.0)       [0.03, 0.60]
  min_drop    : 최소 낙폭     drop > min_drop               [0.03, 0.22]
  stop_factor : 손절선       exit if rebound > sf × drop_entry [0.20, 2.00]
  trail       : 트레일링 스탑 TQQQ peak 대비                [-0.50, -0.02]

유효성 조건 : stop_factor > beta + 0.02  (손절 활성, DE 전 구간 use_stop_loss=True)

목적함수 : Sharpe × sqrt(ΔCAGR)   CAGR > S1 AND Sharpe > S1 강제
"""

import sys, warnings, time, json
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "01_CODE")

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Apple SD Gothic Neo"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.optimize import differential_evolution
from backtest_switching import (
    load_extended_daily,
    strategy_s5,
    strategy_buy_and_hold,
    strategy_switching,
    strategy_switching_rsi,
    strategy_s4_trailing,
    compute_statistics,
)
from evaluation_metrics import oos_metric_bundle

OUT_DIR = Path("03_RESULT/sensitivity")


def half_half_buy_hold(a: pd.DataFrame, b: pd.DataFrame, cap: float = 100_000) -> pd.Series:
    """초기 50/50 매수 후 리밸런스 없음 (drift)."""
    sha = (cap * 0.5) / a["Close"].iloc[0]
    shb = (cap * 0.5) / b["Close"].iloc[0]
    return pd.Series(
        sha * a["Close"].values + shb * b["Close"].values,
        index=a.index,
        name="B2",
    )

# ── 데이터 ──────────────────────────────────────────────────────────────────
print("=" * 64)
print("데이터 로드 (extended 1999~)...")
qqq  = load_extended_daily("QQQ")
qld  = load_extended_daily("QLD")
tqqq = load_extended_daily("TQQQ")
common = qqq.index.intersection(qld.index).intersection(tqqq.index)
qqq, qld, tqqq = qqq.loc[common], qld.loc[common], tqqq.loc[common]

IS_START  = pd.Timestamp("2002-10-01")
IS_END    = pd.Timestamp("2019-11-01")
OOS_START = pd.Timestamp("2019-11-02")

qqq_is   = qqq[(qqq.index >= IS_START) & (qqq.index <= IS_END)]
qld_is   = qld[(qld.index >= IS_START) & (qld.index <= IS_END)]
tqqq_is  = tqqq[(tqqq.index >= IS_START) & (tqqq.index <= IS_END)]
qqq_oos  = qqq[qqq.index >= OOS_START]
qld_oos  = qld[qld.index >= OOS_START]
tqqq_oos = tqqq[tqqq.index >= OOS_START]
qqq_all  = qqq[qqq.index >= IS_START]
qld_all  = qld[qld.index >= IS_START]
tqqq_all = tqqq[tqqq.index >= IS_START]

print(f"  IS  (학습): {qqq_is.index[0].date()} ~ {qqq_is.index[-1].date()}  ({len(qqq_is):,}일)")
print(f"  OOS (검증): {qqq_oos.index[0].date()} ~ {qqq_oos.index[-1].date()} ({len(qqq_oos):,}일)")

# ── 보조 함수 ────────────────────────────────────────────────────────────────
def eval_s5(Q, L, T, beta, max_drop, min_drop, stop_factor, trail,
            use_stop_loss=True, position_mode="linear"):
    try:
        port, _ = strategy_s5(
            Q, L, T, 100_000,
            beta=float(beta), max_drop=float(max_drop), min_drop=float(min_drop),
            stop_factor=float(stop_factor), trailing_stop_pct=float(trail),
            use_stop_loss=bool(use_stop_loss),
            position_mode=str(position_mode),
        )
        st = compute_statistics(port, "S5")
        return {"cagr": float(st["CAGR"].rstrip("%"))/100,
                "mdd":  float(st["MDD"].rstrip("%"))/100,
                "sharpe": float(st["샤프 비율"])}
    except Exception:
        return None

def eval_s1(Q):
    st = compute_statistics(strategy_buy_and_hold(Q, 100_000, "S1"), "S1")
    return {"cagr": float(st["CAGR"].rstrip("%"))/100,
            "mdd":  float(st["MDD"].rstrip("%"))/100,
            "sharpe": float(st["샤프 비율"])}

# ── 벤치마크 ─────────────────────────────────────────────────────────────────
s1_is  = eval_s1(qqq_is)
s1_oos = eval_s1(qqq_oos)
s1_all = eval_s1(qqq_all)
print(f"\nS1 벤치마크 (QQQ Buy-and-Hold):")
print(f"  IS  (2002-10 ~ 2019-11-01) CAGR={s1_is['cagr']:.2%}  MDD={s1_is['mdd']:.2%}  Sharpe={s1_is['sharpe']:.3f}")
print(f"  OOS (2019-11-02~) CAGR={s1_oos['cagr']:.2%}  MDD={s1_oos['mdd']:.2%}  Sharpe={s1_oos['sharpe']:.3f}")

# 기본 파라미터 확인 (S5 sanity check: tiered 앵커 — min_drop=10%, 손절 없음)
r_def_is = eval_s5(qqq_is, qld_is, tqqq_is, 0.5, 0.40, 0.10, 0.75, -0.15,
                    use_stop_loss=True, position_mode="exp")
print(f"\nS5 앵커 (β=0.5, exp 25→100%, min/max_drop=10/40%, SL=drop deepen, trail=-15%) IS 확인:")
if r_def_is:
    print(f"  CAGR={r_def_is['cagr']:.2%}  MDD={r_def_is['mdd']:.2%}  Sharpe={r_def_is['sharpe']:.3f}")

# ── Differential Evolution ────────────────────────────────────────────────────
#  params = [beta, max_drop, min_drop, stop_factor, trail]
BOUNDS = [
    (0.10, 0.95),   # beta
    (0.03, 0.60),   # max_drop
    (0.03, 0.22),   # min_drop
    (0.20, 2.00),   # stop_factor
    (-0.50, -0.02), # trail
]

eval_log      = []
eval_count    = [0]
start_time    = time.time()
gen_history   = []
prev_best     = [None]
stall_count   = [0]

def objective(params):
    beta, max_drop, min_drop, stop_factor, trail = params
    eval_count[0] += 1

    if min_drop >= max_drop * 0.98:
        return 600.0 + (min_drop - max_drop) * 1000

    if stop_factor <= beta + 0.02:
        return 500.0 + (beta - stop_factor + 0.02) * 1000

    r = eval_s5(qqq_is, qld_is, tqqq_is, beta, max_drop, min_drop, stop_factor, trail, True)
    if r is None:
        return 1e6

    cagr_viol   = max(0.0, s1_is["cagr"]   - r["cagr"])
    sharpe_viol = max(0.0, s1_is["sharpe"] - r["sharpe"])
    if cagr_viol > 0 or sharpe_viol > 0:
        return 100.0 + cagr_viol * 200 + sharpe_viol * 50

    delta_cagr = r["cagr"] - s1_is["cagr"]
    score      = r["sharpe"] * (delta_cagr ** 0.5)
    eval_log.append({"n": eval_count[0], "score": score, **r,
                     "beta": beta, "max_drop": max_drop, "min_drop": min_drop,
                     "stop_factor": stop_factor, "trail": trail})
    return -score

gen_count = [0]
def callback(xk, convergence):
    gen_count[0] += 1
    best = max(eval_log, key=lambda x: x["score"]) if eval_log else None
    if best is None:
        return
    elapsed = time.time() - start_time

    if prev_best[0] is not None:
        imp = best["score"] - prev_best[0]
        stall_count[0] = 0 if imp > 1e-5 else stall_count[0] + 1
    prev_best[0] = best["score"]

    gen_history.append({
        "gen": gen_count[0], "score": best["score"],
        "sharpe": best["sharpe"], "cagr": best["cagr"], "mdd": best["mdd"],
        "beta": best["beta"], "max_drop": best["max_drop"], "min_drop": best["min_drop"],
        "stop_factor": best["stop_factor"], "trail": best["trail"],
    })

    if gen_count[0] % 10 == 0:
        stall_msg = f" [스탈 {stall_count[0]}세대]" if stall_count[0] >= 5 else ""
        print(f"  Gen {gen_count[0]:>3} | {eval_count[0]:>5,}회 | "
              f"Sharpe={best['sharpe']:.3f}  CAGR={best['cagr']:.1%}  MDD={best['mdd']:.1%}"
              f"  β={best['beta']:.3f}  md={best['max_drop']*100:.1f}%"
              f"  mind={best['min_drop']*100:.1f}%"
              f"  sf={best['stop_factor']:.3f}  tr={best['trail']*100:.1f}%"
              f"  ({elapsed:.0f}s){stall_msg}")

print("\n" + "=" * 64)
print("Differential Evolution — Strategy 5")
print(f"  IS: {IS_START.date()} ~ {IS_END.date()}")
print(f"  5D: beta, max_drop, min_drop, stop_factor, trail")
print(f"  Pop: 15×{len(BOUNDS)}={15*len(BOUNDS)}  |  Max iter: 300세대")
print("=" * 64)

result = differential_evolution(
    objective, BOUNDS,
    seed=42, maxiter=300, popsize=15,
    mutation=(0.5, 1.5), recombination=0.7,
    tol=1e-8, polish=True, disp=False,
    callback=callback,
)

elapsed_total = time.time() - start_time
print(f"\n최적화 완료! ({elapsed_total:.0f}s, {eval_count[0]:,}회)")

# ── 특이사항 리포트 ───────────────────────────────────────────────────────────
print("\n[특이사항 리포트]")
if gen_history:
    scores = [g["score"] for g in gen_history]
    impr   = [scores[i]-scores[i-1] for i in range(1, len(scores))]
    n_early= sum(1 for x in impr[:50] if x > 1e-4)
    n_flat = sum(1 for x in impr[50:] if abs(x) < 1e-5)
    print(f"  초기 50세대 개선 횟수: {n_early}")
    print(f"  50세대 이후 평탄 비율: {n_flat}/{max(1,len(impr[50:]))}")
    print(f"  beta 추이: 초기={gen_history[0]['beta']:.3f}"
          f" → 50세대={gen_history[min(49,len(gen_history)-1)]['beta']:.3f}"
          f" → 최종={gen_history[-1]['beta']:.3f}")
    print(f"  max_drop: 초기={gen_history[0]['max_drop']*100:.1f}%"
          f" → 50세대={gen_history[min(49,len(gen_history)-1)]['max_drop']*100:.1f}%"
          f" → 최종={gen_history[-1]['max_drop']*100:.1f}%")
    print(f"  min_drop: 초기={gen_history[0]['min_drop']*100:.1f}%"
          f" → 50세대={gen_history[min(49,len(gen_history)-1)]['min_drop']*100:.1f}%"
          f" → 최종={gen_history[-1]['min_drop']*100:.1f}%")
    print(f"  stop_factor: {gen_history[-1]['stop_factor']:.4f}  "
          f"  (>beta={result.x[0]:.3f} : {'✓' if result.x[3]>result.x[0]+0.02 else '✗'})")

# ── 최적 파라미터 ─────────────────────────────────────────────────────────────
beta_opt, md_opt, mind_opt, sf_opt, trail_opt = result.x
print("\n" + "=" * 64)
print(f"최적 파라미터 (IS {IS_START.date()} ~ {IS_END.date()} 기준)")
print("=" * 64)
print(f"  beta        = {beta_opt:.4f}   (진입: rebound < β×drop)")
print(f"  max_drop    = {md_opt*100:.2f}%  (포지션 = min(drop/md, 1.0))")
print(f"  min_drop    = {mind_opt*100:.2f}%  (drop > min_drop 필요)")
print(f"  stop_factor = {sf_opt:.4f}   (손절: rebound > sf×drop_entry)")
print(f"  trail       = {trail_opt*100:.2f}%  (TRAILING 모드 트레일링 스탑)")
print(f"  ─ 해석 ─")
print(f"  예시 drop=15%: 진입은 rebound<{beta_opt*15:.1f}%, "
      f"포지션={min(15/md_opt/100,1)*100:.0f}%, 손절=rebound>{sf_opt*15:.1f}%")

# ── 성과 평가 ─────────────────────────────────────────────────────────────────
r_opt_is  = eval_s5(qqq_is,  qld_is,  tqqq_is,  beta_opt, md_opt, mind_opt, sf_opt, trail_opt, True)
r_opt_oos = eval_s5(qqq_oos, qld_oos, tqqq_oos, beta_opt, md_opt, mind_opt, sf_opt, trail_opt, True)
r_opt_all = eval_s5(qqq_all, qld_all, tqqq_all, beta_opt, md_opt, mind_opt, sf_opt, trail_opt, True)
r_def_is  = eval_s5(qqq_is,  qld_is,  tqqq_is,  0.5, 0.20, 0.10, 0.75, -0.15, False)
r_def_oos = eval_s5(qqq_oos, qld_oos, tqqq_oos, 0.5, 0.20, 0.10, 0.75, -0.15, False)
r_def_all = eval_s5(qqq_all, qld_all, tqqq_all, 0.5, 0.20, 0.10, 0.75, -0.15, False)

print("\n" + "=" * 64)
print("성과 비교")
print("=" * 64)
def fmt(label, r, s1):
    oc = "✓" if r["cagr"] > s1["cagr"] else "✗"
    os = "✓" if r["sharpe"] > s1["sharpe"] else "✗"
    print(f"  {label:28}  CAGR={r['cagr']:>7.2%}{oc}  MDD={r['mdd']:>7.2%}  Sharpe={r['sharpe']:.3f}{os}")

print("  ── S1 QQQ B&H ──")
fmt("IS  (2002-10~2019-11)", s1_is,  s1_is)
fmt("OOS (2019-11~)", s1_oos, s1_oos)
print("  ── S5 앵커 (β=0.5, md=20%, min_drop=10%, SL off, tr=-15%) ──")
fmt("IS  (2002-10~2019-11)", r_def_is,  s1_is)
fmt("OOS (2019-11~)", r_def_oos, s1_oos)
fmt("전체", r_def_all, s1_all)
print("  ── S5 DE 최적 (IS, 손절 on) ──")
fmt("IS  (2002-10~2019-11)", r_opt_is,  s1_is)
fmt("OOS (2019-11~)", r_opt_oos, s1_oos)
fmt("전체", r_opt_all, s1_all)

# ── 시각화 ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 18))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)
fig.suptitle(
    "Strategy 5 — DE 최적화 결과\n"
    f"IS: {IS_START.date()} ~ {IS_END.date()}  |  OOS: {OOS_START.date()} ~",
    fontsize=11, fontweight="bold",
)

# ① 수렴 곡선
ax = fig.add_subplot(gs[0, :])
if eval_log:
    ns     = [e["n"] for e in eval_log]
    scores = [e["score"] for e in eval_log]
    best_sf, cur = [], -np.inf
    for s in scores:
        cur = max(cur, s); best_sf.append(cur)
    ax.scatter(ns, scores, s=4, alpha=0.15, color="#9CA3AF")
    ax.plot(ns, best_sf, color="#DC2626", lw=2, label="Best so far")
    # beta 값을 색상으로 표시
    betas = [e["beta"] for e in eval_log]
    sc = ax.scatter(ns, scores, c=betas, cmap="RdYlGn", s=6, alpha=0.4, zorder=3,
                    vmin=0.1, vmax=0.95)
    plt.colorbar(sc, ax=ax, label="beta", fraction=0.03).ax.tick_params(labelsize=7)
ax.set_xlabel("평가 횟수"); ax.set_ylabel("목적 점수 (Sharpe × √ΔCAGR)")
ax.set_title("수렴 곡선 (색상 = beta 값)", fontsize=9, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

# ② 파라미터 추이 (세대별)
ax2 = fig.add_subplot(gs[1, 0])
if gen_history:
    gens = [g["gen"] for g in gen_history]
    ax2.plot(gens, [g["beta"] for g in gen_history], lw=1.8, label="beta")
    ax2.plot(gens, [g["max_drop"] for g in gen_history], lw=1.8, label="max_drop")
    ax2.plot(gens, [g["min_drop"] for g in gen_history], lw=1.8, label="min_drop")
    ax2.plot(gens, [g["stop_factor"] for g in gen_history], lw=1.8, label="stop_factor")
    ax2.plot(gens, [-g["trail"] for g in gen_history], lw=1.8, ls="--", label="|trail|")
    ax2.set_xlabel("세대"); ax2.set_ylabel("값")
    ax2.set_title("세대별 최적 파라미터 추이", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# ③ IS 성과 추이
ax3 = fig.add_subplot(gs[1, 1])
if gen_history:
    ax3.plot(gens, [g["cagr"]*100 for g in gen_history], color="#DC2626", lw=1.5, label="CAGR %")
    ax3.axhline(y=s1_is["cagr"]*100, color="#DC2626", ls=":", lw=1)
    ax3r = ax3.twinx()
    ax3r.plot(gens, [g["sharpe"] for g in gen_history], color="#2563EB", lw=1.5, label="Sharpe")
    ax3r.axhline(y=s1_is["sharpe"], color="#2563EB", ls=":", lw=1)
    ax3.set_xlabel("세대"); ax3.set_ylabel("CAGR %", color="#DC2626")
    ax3r.set_ylabel("Sharpe", color="#2563EB")
    ax3.set_title("세대별 IS 성과", fontsize=9, fontweight="bold")
    ax3.grid(True, alpha=0.3)

# ④ IS/OOS/전체 막대 비교
ax4 = fig.add_subplot(gs[2, :])
periods   = [f"IS ({IS_START.date()}~{IS_END.date()})", f"OOS ({OOS_START.date()}~)", "전체"]
s1_cagrs  = [s1_is["cagr"]*100,   s1_oos["cagr"]*100,   s1_all["cagr"]*100]
def_cagrs = [r_def_is["cagr"]*100, r_def_oos["cagr"]*100, r_def_all["cagr"]*100]
opt_cagrs = [r_opt_is["cagr"]*100, r_opt_oos["cagr"]*100, r_opt_all["cagr"]*100]
s1_shs    = [s1_is["sharpe"],   s1_oos["sharpe"],   s1_all["sharpe"]]
def_shs   = [r_def_is["sharpe"], r_def_oos["sharpe"], r_def_all["sharpe"]]
opt_shs   = [r_opt_is["sharpe"], r_opt_oos["sharpe"], r_opt_all["sharpe"]]
opt_mdds  = [r_opt_is["mdd"]*100, r_opt_oos["mdd"]*100, r_opt_all["mdd"]*100]

x = np.arange(3); w = 0.25
ax4.bar(x-w,  s1_cagrs,  w, label="S1 QQQ",    color="#9CA3AF", alpha=0.9)
ax4.bar(x,    def_cagrs, w, label="S5 앵커(SL off)", color="#2563EB", alpha=0.85)
ax4.bar(x+w,  opt_cagrs, w, label="S5 DE최적", color="#DC2626", alpha=0.85)
ax4r = ax4.twinx()
ax4r.plot(x-w, s1_shs,  "o--", color="#9CA3AF", lw=1.8, ms=7)
ax4r.plot(x,   def_shs, "s--", color="#2563EB", lw=1.8, ms=7)
ax4r.plot(x+w, opt_shs, "^-",  color="#DC2626", lw=2.2, ms=9)
for i,(xp,m) in enumerate(zip(x+w, opt_mdds)):
    ax4.annotate(f"MDD\n{m:.0f}%", (xp, max(0,opt_cagrs[i])+1), ha="center",
                 fontsize=7, color="#DC2626")
ax4.set_xticks(x); ax4.set_xticklabels(periods, fontsize=9)
ax4.set_ylabel("CAGR (%)", fontsize=9); ax4r.set_ylabel("Sharpe", fontsize=9)
ax4.set_title("IS / OOS / 전체 성과 비교 (막대=CAGR, 선=Sharpe)", fontsize=9, fontweight="bold")
lines1,lbs1=ax4.get_legend_handles_labels(); lines2,lbs2=ax4r.get_legend_handles_labels()
ax4.legend(lines1+lines2[:3], lbs1+lbs2[:3], fontsize=8, loc="upper left")
ax4.grid(True, alpha=0.3, axis="y")

plt.savefig(OUT_DIR / "s5_optimization_result.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── OOS: 벤치마크 vs S2–S5 (누적·CAGR·MDD·Sharpe·Sortino·Ulcer) ─────────────
_CAP = 100_000
port_b1 = strategy_buy_and_hold(qqq_oos, _CAP, "B1")
port_b2 = half_half_buy_hold(qqq_oos, tqqq_oos, _CAP)
port_b3 = strategy_buy_and_hold(tqqq_oos, _CAP, "B3")
# S2/S3/S4 앵커: stage4 S4 Anchor와 동일 파라미터 체계 —10 / —20, 반등 —5 / —10, 트레일 —10%
port_s2_anchor_oos, _ = strategy_switching(qqq_oos, qld_oos, tqqq_oos, _CAP)
port_s3_anchor_oos, _ = strategy_switching_rsi(qqq_oos, qld_oos, tqqq_oos, _CAP)
port_s4_anchor_oos, _ = strategy_s4_trailing(
    qqq_oos, qld_oos, tqqq_oos, _CAP,
    shallow_drop=-0.10,
    deep_drop=-0.20,
    shallow_bounce=-0.05,
    deep_bounce=-0.10,
    half_stop=-0.15,
    trailing_stop_pct=-0.15,
    half_frac=0.50,
    full_frac=1.00,
)
port_s5_anc_oos, _ = strategy_s5(
    qqq_oos, qld_oos, tqqq_oos, _CAP,
    beta=0.5, max_drop=0.40, min_drop=0.10,
    trailing_stop_pct=-0.15, use_stop_loss=True, position_mode="exp",
)
port_s5_de_oos, _ = strategy_s5(
    qqq_oos, qld_oos, tqqq_oos, _CAP,
    beta=beta_opt, max_drop=md_opt, min_drop=mind_opt,
    stop_factor=sf_opt, trailing_stop_pct=trail_opt, use_stop_loss=True,
    position_mode="linear",
)

oos_comparison = {
    "B1 QQQ 100%": oos_metric_bundle(port_b1),
    "B2 QQQ/TQQQ 50/50 drift": oos_metric_bundle(port_b2),
    "B3 TQQQ 100%": oos_metric_bundle(port_b3),
    "S2 앵커 (2단 스위칭 -10/-20)": oos_metric_bundle(port_s2_anchor_oos),
    "S3 앵커 (2단+RSI)": oos_metric_bundle(port_s3_anchor_oos),
        "S4 앵커 (트레일링 -15%)": oos_metric_bundle(port_s4_anchor_oos),
    "S5 앵커 (exp, SL=drop deepen)": oos_metric_bundle(port_s5_anc_oos),
    "S5 DE 최적 (IS 튜닝, SL on)": oos_metric_bundle(port_s5_de_oos),
}

print("\n" + "=" * 96)
print(
    f"OOS 성과 비교  ({qqq_oos.index[0].date()} ~ {qqq_oos.index[-1].date()}, "
    f"초기자본 ${_CAP:,.0f})"
)
print("=" * 96)
hdr = (
    f"{'전략':<40} {'누적수익':>10} {'CAGR':>9} {'MDD':>9} "
    f"{'Sharpe':>8} {'Sortino':>8} {'Ulcer':>8}"
)
print(hdr)
print("-" * len(hdr))
for name, m in oos_comparison.items():
    print(
        f"{name:<40} {m['total_return']:>9.2%} {m['cagr']:>8.2%} {m['mdd']:>8.2%} "
        f"{m['sharpe']:>8.2f} {m['sortino']:>8.2f} {m['ulcer']:>8.2f}"
    )
print("=" * 96)

# JSON 저장
summary = {
    "method": f"Differential Evolution (S5, IS={IS_START.date()}~{IS_END.date()})",
    "is_period": f"{IS_START.date()}~{IS_END.date()}",
    "oos_period": f"{OOS_START.date()}~",
    "total_evals": eval_count[0], "elapsed_sec": round(elapsed_total, 1),
    "optimal_params": {
        "beta": round(beta_opt, 5), "max_drop": round(md_opt, 5),
        "min_drop": round(mind_opt, 5), "stop_factor": round(sf_opt, 5),
        "trail": round(trail_opt, 5), "use_stop_loss": True,
    },
    "is_perf":  r_opt_is, "oos_perf": r_opt_oos, "all_perf": r_opt_all,
    "s1_is": s1_is, "s1_oos": s1_oos, "s1_all": s1_all,
    "default_is": r_def_is, "default_oos": r_def_oos, "default_all": r_def_all,
    "gen_history": gen_history,
    "oos_comparison": oos_comparison,
    "oos_date_range": [
        str(qqq_oos.index[0].date()),
        str(qqq_oos.index[-1].date()),
    ],
}
with open(OUT_DIR / "s5_optimization_result.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\ns5_optimization_result.png  저장 완료")
print("s5_optimization_result.json 저장 완료")
