"""
backtest_optimize_extended.py
==============================
확장 데이터(1999~)를 사용한 6파라미터 Differential Evolution 최적화.

• IS  (학습): 1999-03-10 ~ 2016-12-31  ← 닷컴버블·금융위기 포함
• OOS (검증): 2017-01-01 ~ 2026-04-30  ← COVID·2022약세장 포함

데이터: load_extended_daily — 상장 전은 교정된 synthetic, 상장 후는 실제 가격
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
    load_extended_daily, strategy_s4_trailing,
    strategy_buy_and_hold, compute_statistics,
)

OUT_DIR = Path("03_RESULT/sensitivity")

# ── 데이터 로드 & 분할 ─────────────────────────────────────────────────────────
print("=" * 62)
print("데이터 로드 (확장 데이터 1999~)...")
qqq  = load_extended_daily("QQQ")
qld  = load_extended_daily("QLD")
tqqq = load_extended_daily("TQQQ")

common = qqq.index.intersection(qld.index).intersection(tqqq.index)
qqq, qld, tqqq = qqq.loc[common], qld.loc[common], tqqq.loc[common]

IS_END    = pd.Timestamp("2016-12-31")
OOS_START = pd.Timestamp("2017-01-01")

qqq_is   = qqq[qqq.index <= IS_END];  qld_is  = qld[qld.index <= IS_END];  tqqq_is  = tqqq[tqqq.index <= IS_END]
qqq_oos  = qqq[qqq.index >= OOS_START]; qld_oos = qld[qld.index >= OOS_START]; tqqq_oos = tqqq[tqqq.index >= OOS_START]

print(f"  IS  (학습): {qqq_is.index[0].date()} ~ {qqq_is.index[-1].date()}  ({len(qqq_is):,}일)")
print(f"  OOS (검증): {qqq_oos.index[0].date()} ~ {qqq_oos.index[-1].date()} ({len(qqq_oos):,}일)")
print(f"  IS 주요 이벤트: 닷컴버블(2000-02), 금융위기(2007-09), 2010-2016 불마켓")

# ── 보조 함수 ──────────────────────────────────────────────────────────────────
def derive(b, r1, r2, trail):
    sd=b*r1; db=b*r2; dd=b*r1*r2; hs=(sd+dd)/2
    return dict(shallow_drop=sd, shallow_bounce=b, deep_drop=dd,
                deep_bounce=db, half_stop=hs, trailing_stop=trail)

def eval_s4(Q, L, T, p, hf, ff):
    try:
        port, _ = strategy_s4_trailing(
            Q, L, T, 100_000,
            shallow_drop=p["shallow_drop"],  deep_drop=p["deep_drop"],
            shallow_bounce=p["shallow_bounce"], deep_bounce=p["deep_bounce"],
            half_stop=p["half_stop"], trailing_stop_pct=p["trailing_stop"],
            half_frac=hf, full_frac=ff,
        )
        st = compute_statistics(port, "S4")
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

# ── IS 벤치마크 ────────────────────────────────────────────────────────────────
s1_is  = eval_s1(qqq_is)
s1_oos = eval_s1(qqq_oos)
s1_all = eval_s1(qqq)
print(f"\nS1 벤치마크 (QQQ Buy-and-Hold):")
print(f"  IS  (1999-2016) CAGR={s1_is['cagr']:.2%}  MDD={s1_is['mdd']:.2%}  Sharpe={s1_is['sharpe']:.3f}")
print(f"  OOS (2017-2026) CAGR={s1_oos['cagr']:.2%}  MDD={s1_oos['mdd']:.2%}  Sharpe={s1_oos['sharpe']:.3f}")
print(f"  전체(1999-2026) CAGR={s1_all['cagr']:.2%}  MDD={s1_all['mdd']:.2%}  Sharpe={s1_all['sharpe']:.3f}")

# ── Differential Evolution ────────────────────────────────────────────────────
BOUNDS = [
    (-0.20, -0.01),  # b
    ( 1.05,  5.00),  # r1
    ( 1.05,  5.00),  # r2
    (-0.40, -0.02),  # trail
    ( 0.10,  0.90),  # half_frac
    ( 0.10,  1.00),  # full_frac
]

eval_log = []
eval_count = [0]
start_time = time.time()

# 세대별 최고 기록 추적 (특이사항 감지용)
gen_best_history = []

def objective(params):
    b, r1, r2, trail, hf, ff = params
    eval_count[0] += 1
    p = derive(float(b), float(r1), float(r2), float(trail))
    r = eval_s4(qqq_is, qld_is, tqqq_is, p, float(hf), float(ff))

    if r is None:
        return 1e6

    cagr_viol   = max(0.0, s1_is["cagr"]   - r["cagr"])
    sharpe_viol = max(0.0, s1_is["sharpe"] - r["sharpe"])
    if cagr_viol > 0 or sharpe_viol > 0:
        return 100.0 + cagr_viol * 200 + sharpe_viol * 50

    delta_cagr = r["cagr"] - s1_is["cagr"]
    score      = r["sharpe"] * (delta_cagr ** 0.5)
    eval_log.append({"n": eval_count[0], "score": score, **r,
                     "b": b, "r1": r1, "r2": r2, "trail": trail, "hf": hf, "ff": ff})
    return -score

gen_count    = [0]
prev_best    = [None]
stall_count  = [0]

def callback(xk, convergence):
    gen_count[0] += 1
    best = max(eval_log, key=lambda x: x["score"]) if eval_log else None
    if best is None:
        return
    elapsed = time.time() - start_time

    # 수렴 속도 감지 (스탈 횟수)
    if prev_best[0] is not None:
        improvement = best["score"] - prev_best[0]
        if improvement < 1e-5:
            stall_count[0] += 1
        else:
            stall_count[0] = 0

    gen_best_history.append({
        "gen": gen_count[0], "score": best["score"],
        "sharpe": best["sharpe"], "cagr": best["cagr"], "mdd": best["mdd"],
        "b": best["b"], "r1": best["r1"], "r2": best["r2"],
        "trail": best["trail"], "hf": best["hf"], "ff": best["ff"],
    })
    prev_best[0] = best["score"]

    if gen_count[0] % 10 == 0:
        stall_msg = f"  [스탈 {stall_count[0]}세대]" if stall_count[0] >= 5 else ""
        print(f"  Gen {gen_count[0]:>3} | {eval_count[0]:>5,}회 | "
              f"Sharpe={best['sharpe']:.3f}  CAGR={best['cagr']:.1%}  "
              f"MDD={best['mdd']:.1%}  b={best['b']*100:.1f}%  "
              f"hf={best['hf']*100:.0f}%  trail={best['trail']*100:.1f}%"
              f"  ({elapsed:.0f}s){stall_msg}")

print("\n" + "=" * 62)
print("Differential Evolution 최적화 시작")
print(f"  IS: 1999-2016 (닷컴버블·금융위기 포함)")
print(f"  6D: b, r1, r2, trail, half_frac, full_frac")
print(f"  Population: 15 × 6 = 90  |  Max iter: 200세대")
print("=" * 62)

result = differential_evolution(
    objective, BOUNDS,
    seed=42, maxiter=200, popsize=15,
    mutation=(0.5, 1.5), recombination=0.7,
    tol=1e-7, polish=True, disp=False,
    callback=callback,
)

elapsed_total = time.time() - start_time
print(f"\n최적화 완료! ({elapsed_total:.0f}s, {eval_count[0]:,}회 평가)")

# ── 특이사항 리포트 ───────────────────────────────────────────────────────────
if gen_best_history:
    scores = [g["score"] for g in gen_best_history]
    # 수렴 구간 감지
    improvements = [scores[i]-scores[i-1] for i in range(1, len(scores))]
    # 빠른 초기 수렴 (처음 30세대)
    early_jump = sum(1 for x in improvements[:30] if x > 1e-4)
    late_flat  = sum(1 for x in improvements[50:] if abs(x) < 1e-5)
    print(f"\n[특이사항 리포트]")
    print(f"  초기 30세대 개선 횟수: {early_jump}")
    print(f"  50세대 이후 평탄 횟수: {late_flat}/{len(improvements[50:])}")

    # half_frac 추이 (이전 최적화에서 90%로 수렴했던 문제 확인)
    hf_history = [g["hf"]*100 for g in gen_best_history]
    print(f"  half_frac 추이: 초기={hf_history[0]:.1f}% → 50세대={hf_history[min(49,len(hf_history)-1)]:.1f}% → 최종={hf_history[-1]:.1f}%")
    b_history = [g["b"]*100 for g in gen_best_history]
    print(f"  b 추이:         초기={b_history[0]:.1f}% → 50세대={b_history[min(49,len(b_history)-1)]:.1f}% → 최종={b_history[-1]:.1f}%")

# ── 최적 파라미터 ─────────────────────────────────────────────────────────────
b_opt, r1_opt, r2_opt, trail_opt, hf_opt, ff_opt = result.x
p_opt = derive(b_opt, r1_opt, r2_opt, trail_opt)

print("\n" + "=" * 62)
print("최적 파라미터 (IS 1999-2016 기준)")
print("=" * 62)
print(f"  b         = {b_opt*100:.3f}%")
print(f"  r1        = {r1_opt:.4f}×")
print(f"  r2        = {r2_opt:.4f}×")
print(f"  trail     = {trail_opt*100:.3f}%")
print(f"  half_frac = {hf_opt*100:.1f}%")
print(f"  full_frac = {ff_opt*100:.1f}%")
print(f"  ─ 유도 ─")
print(f"  shallow_drop   = {p_opt['shallow_drop']*100:.2f}%")
print(f"  shallow_bounce = {p_opt['shallow_bounce']*100:.2f}%")
print(f"  deep_drop      = {p_opt['deep_drop']*100:.2f}%")
print(f"  deep_bounce    = {p_opt['deep_bounce']*100:.2f}%")
print(f"  half_stop      = {p_opt['half_stop']*100:.2f}%")

# ── 성과 평가 ─────────────────────────────────────────────────────────────────
r_opt_is  = eval_s4(qqq_is,  qld_is,  tqqq_is,  p_opt, hf_opt, ff_opt)
r_opt_oos = eval_s4(qqq_oos, qld_oos, tqqq_oos, p_opt, hf_opt, ff_opt)
r_opt_all = eval_s4(qqq,     qld,     tqqq,     p_opt, hf_opt, ff_opt)

# 기본값 비교
DEF = derive(-0.05, 2.0, 2.0, -0.15)
r_def_is  = eval_s4(qqq_is,  qld_is,  tqqq_is,  DEF, 0.5, 1.0)
r_def_oos = eval_s4(qqq_oos, qld_oos, tqqq_oos, DEF, 0.5, 1.0)
r_def_all = eval_s4(qqq,     qld,     tqqq,     DEF, 0.5, 1.0)

print("\n" + "=" * 62)
print("성과 비교 (벤치마크 = S1 QQQ Buy-and-Hold)")
print("=" * 62)
def fmt(label, r, s1):
    oc = "✓" if r["cagr"] > s1["cagr"] else "✗"
    os = "✓" if r["sharpe"] > s1["sharpe"] else "✗"
    print(f"  {label:27}  CAGR={r['cagr']:>7.2%}{oc}  MDD={r['mdd']:>7.2%}  Sharpe={r['sharpe']:.3f}{os}")

print("  ── S1 벤치마크 ──")
fmt("IS  (1999-2016)",  s1_is,  s1_is)
fmt("OOS (2017-2026)",  s1_oos, s1_oos)
print("  ── DE 최적 S4 ──")
fmt("IS  (1999-2016)",  r_opt_is,  s1_is)
fmt("OOS (2017-2026)",  r_opt_oos, s1_oos)
fmt("전체(1999-2026)",  r_opt_all, s1_all)
print("  ── 기본값 S4 (b=-5%,r1=2,r2=2,t=-15%,hf=50%,ff=100%) ──")
fmt("IS  (1999-2016)",  r_def_is,  s1_is)
fmt("OOS (2017-2026)",  r_def_oos, s1_oos)
fmt("전체(1999-2026)",  r_def_all, s1_all)

# ── 시각화 ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 18))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle(
    "Strategy 4 — DE 최적화 (IS: 1999-2016 | OOS: 2017-2026)\n"
    "닷컴버블·금융위기 포함 확장 데이터로 학습",
    fontsize=11, fontweight="bold",
)

# ① 수렴 곡선
ax = fig.add_subplot(gs[0, :])
if eval_log:
    ns     = [e["n"] for e in eval_log]
    scores = [e["score"] for e in eval_log]
    best_so_far, cur_best = [], -np.inf
    for s in scores:
        cur_best = max(cur_best, s)
        best_so_far.append(cur_best)
    ax.scatter(ns, scores, s=3, alpha=0.15, color="#9CA3AF", label="개별 평가")
    ax.plot(ns, best_so_far, color="#DC2626", lw=2, label="Best so far")

    # half_frac 색상으로 hf 분포 표시
    hf_vals = [e["hf"] for e in eval_log]
    sc = ax.scatter(ns, scores, c=hf_vals, cmap="RdYlGn_r", s=5, alpha=0.3, zorder=3)
    plt.colorbar(sc, ax=ax, label="half_frac", fraction=0.03).ax.tick_params(labelsize=7)

    # 세대 경계 표시
    gen_size = 90  # popsize × n_params
    for g in range(0, int(max(ns)/gen_size)+1, 20):
        ax.axvline(x=g*gen_size, color="#E5E7EB", lw=0.5, zorder=0)

ax.set_xlabel("평가 횟수", fontsize=9); ax.set_ylabel("목적 점수", fontsize=9)
ax.set_title("수렴 곡선 (색상=half_frac 값, 녹=낮음·적=높음)", fontsize=9, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

# ② 파라미터 추이 (세대별)
ax2 = fig.add_subplot(gs[1, 0])
if gen_best_history:
    gens = [g["gen"] for g in gen_best_history]
    ax2.plot(gens, [g["b"]*100      for g in gen_best_history], lw=1.5, label="b (%)")
    ax2.plot(gens, [g["hf"]*100     for g in gen_best_history], lw=1.5, label="half_frac (%)")
    ax2.plot(gens, [g["trail"]*100  for g in gen_best_history], lw=1.5, label="trail (%)")
    ax2.plot(gens, [g["r1"]*10      for g in gen_best_history], lw=1.5, ls="--", label="r1 (×10)")
    ax2.plot(gens, [g["r2"]*10      for g in gen_best_history], lw=1.5, ls="--", label="r2 (×10)")
ax2.set_xlabel("세대", fontsize=9); ax2.set_ylabel("값", fontsize=9)
ax2.set_title("세대별 최적 파라미터 추이", fontsize=9, fontweight="bold")
ax2.legend(fontsize=7.5); ax2.grid(True, alpha=0.3)

# ③ IS 성과 추이
ax3 = fig.add_subplot(gs[1, 1])
if gen_best_history:
    gens = [g["gen"] for g in gen_best_history]
    ax3.plot(gens, [g["cagr"]*100  for g in gen_best_history], lw=1.5, color="#DC2626", label="CAGR (%)")
    ax3.axhline(y=s1_is["cagr"]*100, color="#DC2626", ls=":", lw=1, label=f"S1 CAGR ({s1_is['cagr']:.1%})")
    ax3r = ax3.twinx()
    ax3r.plot(gens, [g["sharpe"] for g in gen_best_history], lw=1.5, color="#2563EB", label="Sharpe")
    ax3r.axhline(y=s1_is["sharpe"], color="#2563EB", ls=":", lw=1)
    ax3.set_xlabel("세대", fontsize=9); ax3.set_ylabel("CAGR (%)", fontsize=9)
    ax3r.set_ylabel("Sharpe", fontsize=9)
    ax3.set_title("세대별 IS 성과 개선", fontsize=9, fontweight="bold")
    lines1,lbs1=ax3.get_legend_handles_labels(); lines2,lbs2=ax3r.get_legend_handles_labels()
    ax3.legend(lines1+lines2, lbs1+lbs2, fontsize=7.5)
    ax3.grid(True, alpha=0.3)

# ④ IS/OOS/전체 비교 막대
ax4 = fig.add_subplot(gs[2, :])
periods    = ["IS (1999-2016)", "OOS (2017-2026)", "전체 (1999-2026)"]
s1_cagrs   = [s1_is["cagr"]*100,     s1_oos["cagr"]*100,     s1_all["cagr"]*100]
opt_cagrs  = [r_opt_is["cagr"]*100,  r_opt_oos["cagr"]*100,  r_opt_all["cagr"]*100]
def_cagrs  = [r_def_is["cagr"]*100,  r_def_oos["cagr"]*100,  r_def_all["cagr"]*100]
s1_shs     = [s1_is["sharpe"],       s1_oos["sharpe"],        s1_all["sharpe"]]
opt_shs    = [r_opt_is["sharpe"],    r_opt_oos["sharpe"],     r_opt_all["sharpe"]]
def_shs    = [r_def_is["sharpe"],    r_def_oos["sharpe"],     r_def_all["sharpe"]]
s1_mdds    = [s1_is["mdd"]*100,      s1_oos["mdd"]*100,       s1_all["mdd"]*100]
opt_mdds   = [r_opt_is["mdd"]*100,   r_opt_oos["mdd"]*100,    r_opt_all["mdd"]*100]
def_mdds   = [r_def_is["mdd"]*100,   r_def_oos["mdd"]*100,    r_def_all["mdd"]*100]

x=np.arange(3); w=0.25
b1=ax4.bar(x-w,  s1_cagrs,  w, label="S1 QQQ",      color="#9CA3AF", alpha=0.9)
b2=ax4.bar(x,    def_cagrs, w, label="S4 기본값",    color="#2563EB", alpha=0.85)
b3=ax4.bar(x+w,  opt_cagrs, w, label="S4 DE최적",   color="#DC2626", alpha=0.85)
ax4r=ax4.twinx()
ax4r.plot(x-w,   s1_shs,  "o--", color="#9CA3AF", lw=1.8, ms=8)
ax4r.plot(x,     def_shs, "s--", color="#2563EB", lw=1.8, ms=8)
ax4r.plot(x+w,   opt_shs, "^-",  color="#DC2626", lw=2.2, ms=9)
# MDD 점 (아래 방향)
for i,(x_,m) in enumerate(zip(x-w, s1_mdds)):  ax4r.annotate(f"{m:.0f}%", (x_,-0.05), ha="center", fontsize=6.5, color="#9CA3AF")
for i,(x_,m) in enumerate(zip(x,   opt_mdds)): ax4r.annotate(f"{m:.0f}%", (x_,-0.05), ha="center", fontsize=6.5, color="#DC2626")
ax4.set_xticks(x); ax4.set_xticklabels(periods, fontsize=9)
ax4.set_ylabel("CAGR (%)", fontsize=9); ax4r.set_ylabel("Sharpe", fontsize=9)
ax4.set_title("IS / OOS / 전체 성과 비교 (막대=CAGR, 선=Sharpe)", fontsize=9, fontweight="bold")
lines1,lbs1=ax4.get_legend_handles_labels(); lines2,lbs2=ax4r.get_legend_handles_labels()
ax4.legend(lines1+lines2[:3], lbs1+lbs2[:3], fontsize=8, loc="upper left")
ax4.grid(True, alpha=0.3, axis="y")

plt.savefig(OUT_DIR/"de_extended_result.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# JSON 저장
summary = {
    "method": "Differential Evolution (scipy) + Extended Data",
    "is_period": "1999-2016", "oos_period": "2017-2026",
    "total_evals": eval_count[0], "elapsed_sec": round(elapsed_total, 1),
    "optimal_params": {
        "b": round(b_opt,5), "r1": round(r1_opt,4),
        "r2": round(r2_opt,4), "trail": round(trail_opt,5),
        "half_frac": round(hf_opt,4), "full_frac": round(ff_opt,4),
    },
    "derived_params": {k:round(v,5) for k,v in p_opt.items()},
    "is_perf":  r_opt_is, "oos_perf": r_opt_oos, "all_perf": r_opt_all,
    "s1_is": s1_is, "s1_oos": s1_oos, "s1_all": s1_all,
    "default_is": r_def_is, "default_oos": r_def_oos, "default_all": r_def_all,
    "gen_history": gen_best_history,
}
with open(OUT_DIR/"de_extended_result.json","w") as f:
    json.dump(summary, f, indent=2)

print("\nde_extended_result.png  저장 완료")
print("de_extended_result.json 저장 완료")
