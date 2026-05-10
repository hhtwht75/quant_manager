"""
backtest_optimize.py
====================
Strategy 4의 6개 파라미터를 Differential Evolution으로 최적화.

최적화 설계
-----------
• 학습(IS)  : 2010-01-01 ~ 2019-12-31  (파라미터 결정에만 사용)
• 검증(OOS) : 2020-01-01 ~ 2026-04-30  (최적화 완료 후 블라인드 평가)

6 파라미터
----------
  b        : HALF 반등진입선            (-20% ~ -1%)
  r1       : shallow_drop / b 배수      (1.05× ~ 5×)
  r2       : deep / shallow 배수        (1.05× ~ 5×)
  trail    : 트레일링 스탑              (-40% ~ -2%)
  half_frac: HALF 어택 시 TQQQ 비중    (10% ~ 90%)
  full_frac: FULL 어택 시 TQQQ 비중    (10% ~ 100%)

유도 파라미터 (derive 함수)
---------------------------
  shallow_drop  = b × r1
  shallow_bounce= b
  deep_drop     = b × r1 × r2
  deep_bounce   = b × r2
  half_stop     = (shallow_drop + deep_drop) / 2

최적화 알고리즘: Differential Evolution (scipy)
• 이유: 비볼록·다봉 블랙박스 함수 → gradient-free 전역 최적화 필요
        DE는 population-based mutation/crossover로 local optima 탈출 효과적
• 목적함수: Sharpe × sqrt(ΔCAGR)  (ΔCAGR = S4_CAGR − S1_CAGR, 양수 강제)
• 제약: CAGR > S1_CAGR AND Sharpe > S1_Sharpe  (패널티 함수로 처리)
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
    load_yahoo_daily, strategy_s4_trailing,
    strategy_buy_and_hold, compute_statistics,
)

OUT_DIR = Path("03_RESULT/sensitivity")

# ── 데이터 로드 & 분할 ─────────────────────────────────────────────────────────
print("=" * 60)
print("데이터 로드 및 기간 분할...")
qqq  = load_yahoo_daily("QQQ")
qld  = load_yahoo_daily("QLD")
tqqq = load_yahoo_daily("TQQQ")
common = qqq.index.intersection(qld.index).intersection(tqqq.index)
qqq, qld, tqqq = qqq.loc[common], qld.loc[common], tqqq.loc[common]

IS_END    = pd.Timestamp("2019-12-31")
OOS_START = pd.Timestamp("2020-01-01")

qqq_is  = qqq[qqq.index <= IS_END];  qld_is  = qld[qld.index <= IS_END];  tqqq_is  = tqqq[tqqq.index <= IS_END]
qqq_oos = qqq[qqq.index >= OOS_START]; qld_oos = qld[qld.index >= OOS_START]; tqqq_oos = tqqq[tqqq.index >= OOS_START]

print(f"  IS  (학습): {qqq_is.index[0].date()}  ~  {qqq_is.index[-1].date()}  ({len(qqq_is):,}일)")
print(f"  OOS (검증): {qqq_oos.index[0].date()} ~  {qqq_oos.index[-1].date()} ({len(qqq_oos):,}일)")

# ── 보조 함수 ──────────────────────────────────────────────────────────────────
def derive(b: float, r1: float, r2: float, trail: float) -> dict:
    sd = b * r1
    db = b * r2
    dd = b * r1 * r2
    hs = (sd + dd) / 2
    return dict(shallow_drop=sd, shallow_bounce=b, deep_drop=dd,
                deep_bounce=db, half_stop=hs, trailing_stop=trail)

def eval_s4(Q, L, T, p: dict, hf: float, ff: float):
    try:
        port, _ = strategy_s4_trailing(
            Q, L, T, 100_000,
            shallow_drop=p["shallow_drop"],  deep_drop=p["deep_drop"],
            shallow_bounce=p["shallow_bounce"], deep_bounce=p["deep_bounce"],
            half_stop=p["half_stop"],          trailing_stop_pct=p["trailing_stop"],
            half_frac=hf, full_frac=ff,
        )
        st = compute_statistics(port, "S4")
        return {
            "cagr":   float(st["CAGR"].rstrip("%")) / 100,
            "mdd":    float(st["MDD"].rstrip("%"))  / 100,
            "sharpe": float(st["샤프 비율"]),
        }
    except Exception:
        return None

def eval_s1(Q):
    st = compute_statistics(strategy_buy_and_hold(Q, 100_000, "S1"), "S1")
    return {
        "cagr":   float(st["CAGR"].rstrip("%")) / 100,
        "mdd":    float(st["MDD"].rstrip("%"))  / 100,
        "sharpe": float(st["샤프 비율"]),
    }

# ── IS 벤치마크 ────────────────────────────────────────────────────────────────
s1_is  = eval_s1(qqq_is)
s1_oos = eval_s1(qqq_oos)
s1_all = eval_s1(qqq)
print(f"\nS1 벤치마크:")
print(f"  IS  CAGR={s1_is['cagr']:.2%}  Sharpe={s1_is['sharpe']:.3f}  MDD={s1_is['mdd']:.2%}")
print(f"  OOS CAGR={s1_oos['cagr']:.2%}  Sharpe={s1_oos['sharpe']:.3f}  MDD={s1_oos['mdd']:.2%}")

# ── Differential Evolution 최적화 ─────────────────────────────────────────────
#
# 파라미터 bounds  [b, r1, r2, trail, half_frac, full_frac]
BOUNDS = [
    (-0.20, -0.01),  # b
    ( 1.05,  5.00),  # r1
    ( 1.05,  5.00),  # r2
    (-0.40, -0.02),  # trail
    ( 0.10,  0.90),  # half_frac
    ( 0.10,  1.00),  # full_frac
]

# 진행 상황 로그
eval_log: list[dict] = []
start_time = time.time()
eval_count = [0]

def objective(params: np.ndarray) -> float:
    b, r1, r2, trail, hf, ff = params
    eval_count[0] += 1
    p = derive(float(b), float(r1), float(r2), float(trail))
    r = eval_s4(qqq_is, qld_is, tqqq_is, p, float(hf), float(ff))

    if r is None:
        return 1e6

    # 제약 조건: CAGR > S1, Sharpe > S1 — 위반 시 강한 패널티
    cagr_viol   = max(0.0, s1_is["cagr"]   - r["cagr"])
    sharpe_viol = max(0.0, s1_is["sharpe"] - r["sharpe"])
    if cagr_viol > 0 or sharpe_viol > 0:
        return 100.0 + cagr_viol * 200 + sharpe_viol * 50

    # 목적함수: 최대화 대상 → 부호 반전
    delta_cagr = r["cagr"] - s1_is["cagr"]  # 항상 양수
    score      = r["sharpe"] * (delta_cagr ** 0.5)

    eval_log.append({"n": eval_count[0], "score": score,
                     "sharpe": r["sharpe"], "cagr": r["cagr"], "mdd": r["mdd"],
                     "b": b, "r1": r1, "r2": r2, "trail": trail, "hf": hf, "ff": ff})
    return -score

# 진행 콜백 (50 세대마다 출력)
gen_count = [0]
def callback(xk, convergence):
    gen_count[0] += 1
    if gen_count[0] % 10 == 0:
        best = max(eval_log, key=lambda x: x["score"]) if eval_log else None
        elapsed = time.time() - start_time
        if best:
            print(f"  Gen {gen_count[0]:>3} | {eval_count[0]:>5}회 | "
                  f"best Sharpe={best['sharpe']:.3f} CAGR={best['cagr']:.1%} | {elapsed:.0f}s")

print("\n" + "=" * 60)
print("Differential Evolution 최적화 시작 (IS: 2010-2019)")
print(f"  파라미터: b, r1, r2, trail, half_frac, full_frac (6D)")
print(f"  Population: 15 × 6 = 90개  |  Max iter: 200세대")
print(f"  목적함수: Sharpe × sqrt(ΔCAGR),  CAGR>S1 & Sharpe>S1 강제")
print("=" * 60)

result = differential_evolution(
    objective,
    BOUNDS,
    seed=42,
    maxiter=200,
    popsize=15,          # 15 × n_params = 90 individuals
    mutation=(0.5, 1.5),
    recombination=0.7,
    tol=1e-7,
    polish=True,         # 최종 Nelder-Mead 로컬 정제
    disp=False,
    callback=callback,
)

elapsed_total = time.time() - start_time
print(f"\n최적화 완료! ({elapsed_total:.0f}s, {eval_count[0]:,}회 평가)")

# ── 최적 파라미터 추출 ────────────────────────────────────────────────────────
b_opt, r1_opt, r2_opt, trail_opt, hf_opt, ff_opt = result.x
p_opt = derive(b_opt, r1_opt, r2_opt, trail_opt)

print("\n" + "=" * 60)
print("최적 파라미터 (IS 학습 기준)")
print("=" * 60)
print(f"  b         = {b_opt*100:.3f}%  (HALF 반등진입선)")
print(f"  r1        = {r1_opt:.3f}×  (shallow_drop / b)")
print(f"  r2        = {r2_opt:.3f}×  (deep / shallow 배수)")
print(f"  trail     = {trail_opt*100:.3f}%  (트레일링 스탑)")
print(f"  half_frac = {hf_opt*100:.1f}%  (HALF 어택 TQQQ 비중)")
print(f"  full_frac = {ff_opt*100:.1f}%  (FULL 어택 TQQQ 비중)")
print("  ─ 유도 파라미터 ─")
print(f"  shallow_drop  = {p_opt['shallow_drop']*100:.2f}%")
print(f"  shallow_bounce= {p_opt['shallow_bounce']*100:.2f}%")
print(f"  deep_drop     = {p_opt['deep_drop']*100:.2f}%")
print(f"  deep_bounce   = {p_opt['deep_bounce']*100:.2f}%")
print(f"  half_stop     = {p_opt['half_stop']*100:.2f}%")

# ── IS 성과 ────────────────────────────────────────────────────────────────────
r_opt_is  = eval_s4(qqq_is,  qld_is,  tqqq_is,  p_opt, hf_opt, ff_opt)
r_opt_oos = eval_s4(qqq_oos, qld_oos, tqqq_oos, p_opt, hf_opt, ff_opt)
r_opt_all = eval_s4(qqq,     qld,     tqqq,     p_opt, hf_opt, ff_opt)

print("\n" + "=" * 60)
print("성과 비교")
print("=" * 60)
def fmt_row(label, r, s1):
    ok_c = "✓" if r["cagr"] > s1["cagr"] else "✗"
    ok_s = "✓" if r["sharpe"] > s1["sharpe"] else "✗"
    print(f"  {label:20}  CAGR={r['cagr']:>7.2%}{ok_c}  MDD={r['mdd']:>7.2%}  Sharpe={r['sharpe']:.3f}{ok_s}")

print("  [S1 벤치마크]")
fmt_row("IS  (2010-19)",   s1_is,  s1_is)
fmt_row("OOS (2020-26)",   s1_oos, s1_oos)
print("  [최적 S4]")
fmt_row("IS  (2010-19)",   r_opt_is,  s1_is)
fmt_row("OOS (2020-26)",   r_opt_oos, s1_oos)
fmt_row("전체 (2010-26)",  r_opt_all, s1_all)

# 기본값 비교
DEF_B, DEF_R1, DEF_R2, DEF_TR, DEF_HF, DEF_FF = -0.05, 2.0, 2.0, -0.15, 0.50, 1.00
p_def = derive(DEF_B, DEF_R1, DEF_R2, DEF_TR)
r_def_is  = eval_s4(qqq_is,  qld_is,  tqqq_is,  p_def, DEF_HF, DEF_FF)
r_def_oos = eval_s4(qqq_oos, qld_oos, tqqq_oos, p_def, DEF_HF, DEF_FF)
r_def_all = eval_s4(qqq,     qld,     tqqq,     p_def, DEF_HF, DEF_FF)
print("  [기본값 S4  (b=-5%,r1=2,r2=2,trail=-15%,hf=50%,ff=100%)]")
fmt_row("IS  (2010-19)",   r_def_is,  s1_is)
fmt_row("OOS (2020-26)",   r_def_oos, s1_oos)
fmt_row("전체 (2010-26)",  r_def_all, s1_all)

# ── 시각화 ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
fig.suptitle(
    "Strategy 4 — Differential Evolution 최적화 결과\n"
    "IS(2010-2019) 학습 → OOS(2020-2026) 블라인드 검증",
    fontsize=11, fontweight="bold",
)

# ① 최적화 수렴 곡선
ax = fig.add_subplot(gs[0, :2])
if eval_log:
    ns     = [e["n"] for e in eval_log]
    scores = [e["score"] for e in eval_log]
    best_so_far = []
    cur_best = -np.inf
    for s in scores:
        cur_best = max(cur_best, s)
        best_so_far.append(cur_best)
    ax.scatter(ns, scores, s=4, alpha=0.2, color="#9CA3AF", label="개별 평가")
    ax.plot(ns, best_so_far, color="#DC2626", lw=2, label="Best so far")
ax.set_xlabel("평가 횟수", fontsize=9)
ax.set_ylabel("목적 점수 (Sharpe × √ΔCAGR)", fontsize=9)
ax.set_title("수렴 곡선 (DE 최적화 과정)", fontsize=9, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ② 파라미터 분포 (유효 평가 기록)
ax2 = fig.add_subplot(gs[0, 2])
valid_log = [e for e in eval_log if e["score"] > 0]
if valid_log:
    param_labels = ["b (%)", "r1 (×)", "r2 (×)", "trail (%)", "half_frac (%)", "full_frac (%)"]
    opt_vals     = [b_opt*100, r1_opt, r2_opt, trail_opt*100, hf_opt*100, ff_opt*100]
    default_vals = [DEF_B*100, DEF_R1, DEF_R2, DEF_TR*100, DEF_HF*100, DEF_FF*100]
    # normalise to [0,1] within bounds
    bound_lo     = [-20, 1.05, 1.05, -40, 10, 10]
    bound_hi     = [-1,  5.0,  5.0, -2,  90, 100]
    opt_norm     = [(v-lo)/(hi-lo) for v,lo,hi in zip(opt_vals, bound_lo, bound_hi)]
    def_norm     = [(v-lo)/(hi-lo) for v,lo,hi in zip(default_vals, bound_lo, bound_hi)]
    y = np.arange(6)
    ax2.barh(y, opt_norm, 0.4, color="#DC2626", alpha=0.8, label="최적값")
    ax2.barh(y-0.4, def_norm, 0.4, color="#2563EB", alpha=0.6, label="기본값")
    ax2.set_yticks(y); ax2.set_yticklabels(param_labels, fontsize=8)
    ax2.set_xlabel("정규화 파라미터 값 (범위 내 위치)", fontsize=8)
    ax2.set_title("파라미터 비교 (최적 vs 기본값)", fontsize=9, fontweight="bold")
    ax2.axvline(x=0.5, color="#9CA3AF", ls="--", lw=1)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="x")

# ③ IS/OOS/전체 CAGR & Sharpe 비교 막대
ax3 = fig.add_subplot(gs[1, :2])
periods   = ["IS (2010-19)", "OOS (2020-26)", "전체 (2010-26)"]
s1_cagrs  = [s1_is["cagr"]*100,  s1_oos["cagr"]*100,  s1_all["cagr"]*100]
opt_cagrs = [r_opt_is["cagr"]*100, r_opt_oos["cagr"]*100, r_opt_all["cagr"]*100]
def_cagrs = [r_def_is["cagr"]*100, r_def_oos["cagr"]*100, r_def_all["cagr"]*100]
s1_shs    = [s1_is["sharpe"],  s1_oos["sharpe"],  s1_all["sharpe"]]
opt_shs   = [r_opt_is["sharpe"], r_opt_oos["sharpe"], r_opt_all["sharpe"]]
def_shs   = [r_def_is["sharpe"], r_def_oos["sharpe"], r_def_all["sharpe"]]

x = np.arange(3); w = 0.25
ax3.bar(x - w, s1_cagrs, w, label="S1 (QQQ Hold)",  color="#9CA3AF", alpha=0.9)
ax3.bar(x,     def_cagrs, w, label="S4 기본값",       color="#2563EB", alpha=0.9)
ax3.bar(x + w, opt_cagrs, w, label="S4 DE최적",       color="#DC2626", alpha=0.9)
ax3r = ax3.twinx()
ax3r.plot(x - w, s1_shs,  "o--", color="#9CA3AF", lw=1.8, ms=7)
ax3r.plot(x,     def_shs, "s--", color="#2563EB", lw=1.8, ms=7)
ax3r.plot(x + w, opt_shs, "^-",  color="#DC2626", lw=2.2, ms=9)
ax3.set_xticks(x); ax3.set_xticklabels(periods, fontsize=9)
ax3.set_ylabel("CAGR (%)", fontsize=9); ax3r.set_ylabel("Sharpe 비율", fontsize=9)
ax3.set_title("CAGR (막대) & Sharpe (선): IS / OOS / 전체", fontsize=9, fontweight="bold")
ax3.legend(fontsize=8, loc="upper left"); ax3.grid(True, alpha=0.3, axis="y")

# ④ MDD 비교
ax4 = fig.add_subplot(gs[1, 2])
mdds = {
    "S1": [s1_is["mdd"]*100, s1_oos["mdd"]*100, s1_all["mdd"]*100],
    "S4 기본값": [r_def_is["mdd"]*100, r_def_oos["mdd"]*100, r_def_all["mdd"]*100],
    "S4 DE최적": [r_opt_is["mdd"]*100, r_opt_oos["mdd"]*100, r_opt_all["mdd"]*100],
}
colors = {"S1": "#9CA3AF", "S4 기본값": "#2563EB", "S4 DE최적": "#DC2626"}
x = np.arange(3); w = 0.25
for i, (label, vals) in enumerate(mdds.items()):
    ax4.bar(x + (i-1)*w, vals, w, label=label, color=colors[label], alpha=0.85)
ax4.set_xticks(x); ax4.set_xticklabels(["IS", "OOS", "전체"], fontsize=9)
ax4.set_ylabel("MDD (%)", fontsize=9)
ax4.set_title("최대 낙폭(MDD) 비교", fontsize=9, fontweight="bold")
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, axis="y")
ax4.invert_yaxis()

plt.savefig(OUT_DIR / "de_optimization_result.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── JSON 저장 ─────────────────────────────────────────────────────────────────
summary = {
    "method": "Differential Evolution (scipy)",
    "is_period": "2010-2019", "oos_period": "2020-2026",
    "total_evals": eval_count[0], "elapsed_sec": round(elapsed_total, 1),
    "optimal_params": {
        "b": round(b_opt, 5), "r1": round(r1_opt, 4),
        "r2": round(r2_opt, 4), "trail": round(trail_opt, 5),
        "half_frac": round(hf_opt, 4), "full_frac": round(ff_opt, 4),
    },
    "derived_params": {k: round(v, 5) for k, v in p_opt.items()},
    "is_performance":  r_opt_is,
    "oos_performance": r_opt_oos,
    "all_performance": r_opt_all,
    "s1_is":  s1_is, "s1_oos": s1_oos, "s1_all": s1_all,
    "default_is": r_def_is, "default_oos": r_def_oos, "default_all": r_def_all,
}
with open(OUT_DIR / "de_optimization_result.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nde_optimization_result.png  저장 완료")
print("de_optimization_result.json 저장 완료")
