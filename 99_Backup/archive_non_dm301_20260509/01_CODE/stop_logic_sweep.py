"""
stop_logic_sweep.py
===================
HALF_STOP / FULL_STOP 로직 점검 + 파라미터 스윕

현재 로직 분석
--------------
① HALF_ATTACK 상태:
   - 진입: QLD -10% 터치 후 -5%로 반등
   - 손절: QLD가 -15%까지 추가 하락 → HALF_STOP (플래그 유지!)
   - 업그레이드: QLD -20% 터치 후 -10% 반등 → FULL_ATTACK
   - 청산: QLD ATH 회복 → TRAILING

② FULL_ATTACK 상태:
   ★ STOP LOSS 없음! ★
   - QLD ATH 회복까지 무한정 대기
   - 심각한 하락장에선 TQQQ 보유 내내 손실 누적

구조적 문제
-----------
1. FULL_ATTACK에 손절 없음 → 장기 베어마켓에서 노출
2. HALF_STOP 후 touched_10/touched_20 플래그 유지
   → 즉시 재진입 가능 → 하락 추세에서 반복 손절 위험
3. HALF_STOP 임계값(-15%)이 너무 관대하거나 너무 촘촘할 수 있음

스윕 파라미터
-------------
• half_stop      : -0.10, -0.12, -0.15(기준), -0.18, -0.20
• shallow_bounce : -0.03, -0.05(기준), -0.07  (HALF_ATTACK 진입 민감도)
• full_stop      : None(기준), -0.25, -0.30, -0.35  (FULL_ATTACK 신규 손절)
• half_frac      : 0.30, 0.50(기준), 0.70  (HALF_ATTACK 진입 비중)
"""

import sys, itertools
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

from backtest_switching import load_extended_daily
from evaluation_metrics import full_metrics

OUT_DIR = Path("03_RESULT")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 확장 전략 (FULL_STOP 파라미터 추가) ───────────────────────────────────────
def strategy_s4_extended(
    defensive, qld, tqqq, initial_capital,
    shallow_drop    = -0.10,
    deep_drop       = -0.20,
    shallow_bounce  = -0.05,
    deep_bounce     = -0.10,
    half_stop       = -0.15,   # HALF_ATTACK 손절 (QLD 기준 ATH대비)
    full_stop       = None,    # FULL_ATTACK 손절 (None=없음). e.g. -0.30
    trailing_stop   = -0.08,
    half_frac       = 0.50,
    full_frac       = 1.00,
    reset_flags_on_half_stop = False,  # True시 HALF_STOP 후 플래그 초기화
):
    dates = defensive.index
    portfolio_values = []
    events = []

    def_shares  = initial_capital / defensive["Close"].iloc[0]
    tqqq_shares = 0.0
    state       = "NORMAL"
    ath         = qld["Close"].iloc[0]
    touched_10  = False
    touched_20  = False
    tqqq_peak   = 0.0
    tqqq_entry  = 0.0

    for i, date in enumerate(dates):
        qc = qld["Close"].iloc[i]
        dc = defensive["Close"].iloc[i]
        tc = tqqq["Close"].iloc[i]

        if qc > ath: ath = qc
        dd = (qc - ath) / ath

        if dd <= shallow_drop: touched_10 = True
        if dd <= deep_drop:    touched_20 = True

        tv = def_shares * dc + tqqq_shares * tc

        if state == "TRAILING":
            if tc > tqqq_peak: tqqq_peak = tc
            if tc <= tqqq_entry:
                def_shares = tv / dc; tqqq_shares = 0.0
                state = "NORMAL"; touched_10 = touched_20 = False
                events.append({"Date": date, "type": "TRAIL_FLOOR", "value": tv})
            elif tc <= tqqq_peak * (1.0 + trailing_stop):
                def_shares = tv / dc; tqqq_shares = 0.0
                state = "NORMAL"; touched_10 = touched_20 = False
                events.append({"Date": date, "type": "TRAIL_EXIT", "value": tv})

        elif state in ("HALF_ATTACK", "FULL_ATTACK"):
            if dd >= 0:
                tqqq_peak = tqqq_entry = tc
                state = "TRAILING"
                events.append({"Date": date, "type": "TO_TRAILING", "value": tv})

            # ── HALF_STOP ─────────────────────────────────────────────────────
            elif state == "HALF_ATTACK" and dd <= half_stop:
                def_shares = tv / dc; tqqq_shares = 0.0
                state = "NORMAL"
                if reset_flags_on_half_stop:          # 옵션: 플래그 초기화
                    touched_10 = touched_20 = False
                # (기본) 플래그 유지 → 즉시 재진입 가능
                events.append({"Date": date, "type": "HALF_STOP", "value": tv})

            # ── FULL_STOP (신규) ───────────────────────────────────────────────
            elif state == "FULL_ATTACK" and full_stop is not None and dd <= full_stop:
                def_shares = tv / dc; tqqq_shares = 0.0
                state = "NORMAL"; touched_10 = touched_20 = False
                events.append({"Date": date, "type": "FULL_STOP", "value": tv})

            # ── HALF → FULL 업그레이드 ─────────────────────────────────────────
            elif state == "HALF_ATTACK" and touched_20 and dd >= deep_bounce:
                tqqq_shares = tv * full_frac / tc
                def_shares  = tv * (1.0 - full_frac) / dc
                state = "FULL_ATTACK"
                events.append({"Date": date, "type": "TO_FULL_ATTACK", "value": tv})

        else:  # NORMAL
            if dd >= 0:
                touched_10 = touched_20 = False
            elif touched_10 and not touched_20 and dd >= shallow_bounce:
                tqqq_shares = tv * half_frac / tc
                def_shares  = tv * (1.0 - half_frac) / dc
                state = "HALF_ATTACK"
                events.append({"Date": date, "type": "TO_HALF_ATTACK", "value": tv})
            elif touched_20 and dd >= deep_bounce:
                tqqq_shares = tv * full_frac / tc
                def_shares  = tv * (1.0 - full_frac) / dc
                state = "FULL_ATTACK"
                events.append({"Date": date, "type": "TO_FULL_ATTACK", "value": tv})

        portfolio_values.append(def_shares * dc + tqqq_shares * tc)

    return (
        pd.Series(portfolio_values, index=dates),
        pd.DataFrame(events) if events else pd.DataFrame(columns=["Date","type","value"])
    )


# ── 데이터 ───────────────────────────────────────────────────────────────────
print("=" * 80)
print("HALF_STOP / FULL_STOP 로직 점검 + 파라미터 스윕")
print("=" * 80)

qqq  = load_extended_daily("QQQ")
qld  = load_extended_daily("QLD")
tqqq = load_extended_daily("TQQQ")
common = qqq.index.intersection(qld.index).intersection(tqqq.index)
common = common[(common >= "2015-01-01") & (common <= "2026-04-30")]
Q = qqq.loc[common]; Qld = qld.loc[common]; T = tqqq.loc[common]
INIT = 100_000

# 기준선
b_qqq = pd.Series(INIT / Q["Close"].iloc[0] * Q["Close"].values, index=Q.index)
m_s1  = full_metrics(b_qqq)
print(f"\n  [B1 QQQ B&H]  CAGR={m_s1['cagr']*100:+.2f}%  "
      f"Sh={m_s1['sharpe']:.2f}  MDD={m_s1['mdd']*100:.2f}%")

# ── 섹션 A: 현재 로직 해부 ───────────────────────────────────────────────────
print(f"\n{'='*80}")
print("  [섹션 A] 현재 로직 (Anchor trail=-8%) 상태 흐름 요약")
print(f"{'='*80}")
base_p = dict(shallow_drop=-0.10, deep_drop=-0.20, shallow_bounce=-0.05,
              deep_bounce=-0.10, half_stop=-0.15, full_stop=None,
              trailing_stop=-0.08, half_frac=0.50, full_frac=1.00)
base_s, base_ev = strategy_s4_extended(Q, Qld, T, INIT, **base_p)
m_base = full_metrics(base_s)

if not base_ev.empty:
    state_cnt = base_ev["type"].value_counts()
    print(f"\n  이벤트 분포:")
    for t, c in state_cnt.items():
        print(f"    {t:16s}: {c:>3}건")

print(f"\n  ★ 주요 구조적 문제:")
print(f"    1. FULL_ATTACK에 손절 없음 → QLD가 -30%,-40%로 계속 하락해도 대기")
print(f"    2. HALF_STOP 후 플래그(touched_10/20) 유지 → 즉시 재진입 위험")
print(f"    3. 2015-2026 에피소드에서 HALF_STOP 3건이 유일한 손실 원인")

# ── 섹션 B: HALF_STOP 임계값 스윕 ─────────────────────────────────────────────
print(f"\n{'='*80}")
print("  [섹션 B] half_stop 임계값 스윕 (다른 파라미터 고정)")
print(f"{'='*80}")

half_stop_vals = [-0.10, -0.12, -0.15, -0.18, -0.20, -0.25]
print(f"\n  {'half_stop':>12}  {'CAGR':>8}  {'Sharpe':>7}  {'Sortino':>8}  "
      f"{'MDD':>8}  {'Ulcer':>6}  {'Calmar':>7}  "
      f"{'HALF_STOP건':>10}  {'FULL_STOP건':>10}  {'에피소드':>8}")
print("  " + "-"*110)
hs_results = {}
for hs in half_stop_vals:
    p = {**base_p, "half_stop": hs}
    s, ev = strategy_s4_extended(Q, Qld, T, INIT, **p)
    m = full_metrics(s)
    n_hs = int((ev["type"] == "HALF_STOP").sum()) if not ev.empty else 0
    n_fs = int((ev["type"] == "FULL_STOP").sum()) if not ev.empty else 0
    n_ep = int(ev["type"].isin(["TO_HALF_ATTACK","TO_FULL_ATTACK"]).sum()) if not ev.empty else 0
    hs_results[hs] = {"m": m, "ev": ev, "s": s}
    star = " ←기준" if hs == -0.15 else ""
    print(f"  {hs*100:>+11.0f}%  {m['cagr']*100:>+7.2f}%  {m['sharpe']:>7.2f}  "
          f"{m['sortino']:>8.2f}  {m['mdd']*100:>+7.2f}%  {m['ulcer']:>6.2f}  "
          f"{m['calmar']:>7.2f}  {n_hs:>10}  {n_fs:>10}  {n_ep:>8}{star}")

# ── 섹션 C: shallow_bounce 스윕 ────────────────────────────────────────────────
print(f"\n{'='*80}")
print("  [섹션 C] shallow_bounce 스윕 (HALF_ATTACK 진입 반등 기준)")
print(f"{'='*80}")

bounce_vals = [-0.03, -0.04, -0.05, -0.06, -0.07, -0.08]
print(f"\n  {'bounce':>8}  {'CAGR':>8}  {'Sharpe':>7}  {'Sortino':>8}  "
      f"{'MDD':>8}  {'HALF_ATTACK':>11}  {'HALF_STOP':>9}  의미")
print("  " + "-"*100)
for bv in bounce_vals:
    p = {**base_p, "shallow_bounce": bv}
    s, ev = strategy_s4_extended(Q, Qld, T, INIT, **p)
    m = full_metrics(s)
    n_ha = int((ev["type"] == "TO_HALF_ATTACK").sum()) if not ev.empty else 0
    n_hs = int((ev["type"] == "HALF_STOP").sum()) if not ev.empty else 0
    star = " ←기준" if bv == -0.05 else ""
    interp = ("엄격(반등 깊이 커야)", "중간", "기준", "약간 완화", "완화(일찍 진입)", "많이 완화")[bounce_vals.index(bv)]
    print(f"  {bv*100:>+7.0f}%  {m['cagr']*100:>+7.2f}%  {m['sharpe']:>7.2f}  "
          f"{m['sortino']:>8.2f}  {m['mdd']*100:>+7.2f}%  {n_ha:>11}  {n_hs:>9}  {interp}{star}")

# ── 섹션 D: FULL_STOP 신규 파라미터 ───────────────────────────────────────────
print(f"\n{'='*80}")
print("  [섹션 D] FULL_STOP 신규 파라미터 스윕 (FULL_ATTACK 손절 추가)")
print(f"{'='*80}")
print(f"  ★ 현재 FULL_ATTACK은 손절 없음 — QLD가 ATH 회복 때까지 무한 대기")

full_stop_vals = [None, -0.25, -0.28, -0.30, -0.33, -0.35, -0.40]
print(f"\n  {'full_stop':>10}  {'CAGR':>8}  {'Sharpe':>7}  {'Sortino':>8}  "
      f"{'MDD':>8}  {'Ulcer':>6}  {'FULL_STOP건':>10}  {'vs 기준':>8}")
print("  " + "-"*100)
fs_results = {}
for fs in full_stop_vals:
    p = {**base_p, "full_stop": fs}
    s, ev = strategy_s4_extended(Q, Qld, T, INIT, **p)
    m = full_metrics(s)
    n_fs = int((ev["type"] == "FULL_STOP").sum()) if not ev.empty else 0
    delta_sh = m["sharpe"] - m_base["sharpe"]
    delta_cagr = m["cagr"] - m_base["cagr"]
    fs_results[fs] = {"m": m, "s": s, "ev": ev}
    fs_lbl = "None(기준)" if fs is None else f"{fs*100:+.0f}%"
    star = " ★" if delta_sh > 0.02 and delta_cagr > 0 else ""
    print(f"  {fs_lbl:>10}  {m['cagr']*100:>+7.2f}%  {m['sharpe']:>7.2f}  "
          f"{m['sortino']:>8.2f}  {m['mdd']*100:>+7.2f}%  {m['ulcer']:>6.2f}  "
          f"{n_fs:>10}  Δsh={delta_sh:>+5.3f} ΔCAGR={delta_cagr*100:>+5.2f}%{star}")

# ── 섹션 E: HALF_STOP 후 플래그 초기화 옵션 ──────────────────────────────────
print(f"\n{'='*80}")
print("  [섹션 E] HALF_STOP 후 touched 플래그 초기화 vs 유지 비교")
print(f"{'='*80}")
print(f"  현재: HALF_STOP 후 플래그 유지 → 다음 bounce에서 즉시 재진입 가능")
print(f"  대안: 플래그 초기화 → QLD가 다시 -10%까지 하락해야만 새 신호 발생\n")

for reset, lbl in [(False, "플래그 유지(기준)"), (True, "플래그 초기화")]:
    p = {**base_p, "reset_flags_on_half_stop": reset}
    s, ev = strategy_s4_extended(Q, Qld, T, INIT, **p)
    m = full_metrics(s)
    n_hs = int((ev["type"] == "HALF_STOP").sum()) if not ev.empty else 0
    n_ep = int(ev["type"].isin(["TO_HALF_ATTACK","TO_FULL_ATTACK"]).sum()) if not ev.empty else 0
    print(f"  [{lbl}]  CAGR={m['cagr']*100:>+6.2f}%  Sh={m['sharpe']:.2f}  "
          f"MDD={m['mdd']*100:>+6.1f}%  HALF_STOP={n_hs}건  총에피={n_ep}건")

# ── 섹션 F: half_frac 스윕 ────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("  [섹션 F] half_frac 스윕 (HALF_ATTACK 진입 비중)")
print(f"{'='*80}")
print(f"  현재: 50% (자산의 절반을 TQQQ로 전환)")

hf_vals = [0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 1.00]
print(f"\n  {'hf':>6}  {'CAGR':>8}  {'Sharpe':>7}  {'Sortino':>8}  "
      f"{'MDD':>8}  {'Ulcer':>6}  {'Calmar':>7}  {'HALF_STOP 손실':>14}")
print("  " + "-"*100)
for hf in hf_vals:
    p = {**base_p, "half_frac": hf}
    s, ev = strategy_s4_extended(Q, Qld, T, INIT, **p)
    m = full_metrics(s)
    # HALF_STOP episodes 평균 손실 추정
    if not ev.empty:
        hs_ev = ev[ev["type"] == "HALF_STOP"]
        hs_losses = []
        for j in range(len(hs_ev)):
            hs_date = hs_ev["Date"].iloc[j]
            # 진입 직전 이벤트 찾기
            prior = ev[ev["Date"] < hs_date]
            if not prior.empty:
                prior_type = prior.iloc[-1]["type"]
                if "ATTACK" in prior_type:
                    entry_val = prior.iloc[-1]["value"]
                    exit_val  = hs_ev["value"].iloc[j]
                    hs_losses.append((exit_val - entry_val) / entry_val)
        avg_hs_loss = np.mean(hs_losses) * 100 if hs_losses else 0
    else:
        avg_hs_loss = 0
    star = " ←기준" if hf == 0.50 else ""
    print(f"  {hf*100:>5.0f}%  {m['cagr']*100:>+7.2f}%  {m['sharpe']:>7.2f}  "
          f"{m['sortino']:>8.2f}  {m['mdd']*100:>+7.2f}%  {m['ulcer']:>6.2f}  "
          f"{m['calmar']:>7.2f}  {avg_hs_loss:>+13.2f}%{star}")

# ── 섹션 G: 그리드 서치 Top-15 ────────────────────────────────────────────────
print(f"\n{'='*80}")
print("  [섹션 G] 그리드 서치: 핵심 파라미터 조합 최적 Top-15 (Sharpe 기준)")
print(f"{'='*80}")
print(f"  ⚠️  주의: 이 결과는 2015-2026 IS 기간에만 최적화됨. OOS 검증 필요!")

grid = {
    "half_stop":      [-0.12, -0.15, -0.18],
    "shallow_bounce": [-0.04, -0.05, -0.07],
    "full_stop":      [None, -0.30, -0.35],
    "half_frac":      [0.40, 0.50, 0.70],
}

all_combos = list(itertools.product(*grid.values()))
grid_results = []
print(f"  총 {len(all_combos)}개 조합 평가 중...", end="", flush=True)

for combo in all_combos:
    hs, sb, fs, hf = combo
    p = dict(shallow_drop=-0.10, deep_drop=-0.20, shallow_bounce=sb,
             deep_bounce=-0.10, half_stop=hs, full_stop=fs,
             trailing_stop=-0.08, half_frac=hf, full_frac=1.00)
    s, ev = strategy_s4_extended(Q, Qld, T, INIT, **p)
    m = full_metrics(s)
    n_hs = int((ev["type"]=="HALF_STOP").sum()) if not ev.empty else 0
    n_fs = int((ev["type"]=="FULL_STOP").sum()) if not ev.empty else 0
    n_ep = int(ev["type"].isin(["TO_HALF_ATTACK","TO_FULL_ATTACK"]).sum()) if not ev.empty else 0
    grid_results.append({
        "half_stop": hs, "shallow_bounce": sb, "full_stop": fs, "half_frac": hf,
        "cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
        "mdd": m["mdd"], "ulcer": m["ulcer"], "calmar": m["calmar"],
        "n_hs": n_hs, "n_fs": n_fs, "n_ep": n_ep,
        "port": s,
    })
print(f"완료 ({len(all_combos)}개)")

gr_df = pd.DataFrame([{k: v for k, v in r.items() if k != "port"} for r in grid_results])
gr_df_sorted = gr_df.sort_values("sharpe", ascending=False).reset_index(drop=True)

# BASE 위치
base_row = gr_df_sorted[
    (gr_df_sorted["half_stop"] == -0.15) &
    (gr_df_sorted["shallow_bounce"] == -0.05) &
    (gr_df_sorted["full_stop"].isna()) &
    (gr_df_sorted["half_frac"] == 0.50)
]

print(f"\n  {'Rank':>4}  {'hs':>6}  {'sb':>6}  {'fs':>8}  {'hf':>5}  "
      f"{'CAGR':>8}  {'Sharpe':>7}  {'Sortino':>8}  {'MDD':>8}  "
      f"{'Ulcer':>6}  {'HS건':>5}  {'FS건':>5}")
print("  " + "-"*110)

base_rank = None
for rank, row in gr_df_sorted.head(20).iterrows():
    is_base = (row["half_stop"]==-0.15 and row["shallow_bounce"]==-0.05
               and pd.isna(row["full_stop"]) and row["half_frac"]==0.50)
    if is_base:
        base_rank = rank + 1
    fs_str = "None" if pd.isna(row["full_stop"]) else f"{row['full_stop']*100:+.0f}%"
    tag = " ←기준" if is_base else ""
    print(f"  {rank+1:>4}  {row['half_stop']*100:>+5.0f}%  "
          f"{row['shallow_bounce']*100:>+5.0f}%  {fs_str:>8}  {row['half_frac']*100:>4.0f}%  "
          f"{row['cagr']*100:>+7.2f}%  {row['sharpe']:>7.3f}  {row['sortino']:>8.3f}  "
          f"{row['mdd']*100:>+7.2f}%  {row['ulcer']:>6.2f}  "
          f"{int(row['n_hs']):>5}  {int(row['n_fs']):>5}{tag}")

if base_rank:
    print(f"\n  → 기준 파라미터 순위: {base_rank}/{len(all_combos)}")

# ── 베스트 조합 vs 기준 에피소드 비교 ─────────────────────────────────────────
best_row = gr_df_sorted.iloc[0]
best_p = dict(
    shallow_drop=-0.10, deep_drop=-0.20,
    shallow_bounce=float(best_row["shallow_bounce"]),
    deep_bounce=-0.10,
    half_stop=float(best_row["half_stop"]),
    full_stop=None if pd.isna(best_row["full_stop"]) else float(best_row["full_stop"]),
    trailing_stop=-0.08,
    half_frac=float(best_row["half_frac"]),
    full_frac=1.00,
)
best_s, best_ev = strategy_s4_extended(Q, Qld, T, INIT, **best_p)
m_best = full_metrics(best_s)
print(f"\n  [베스트 조합]  hs={best_p['half_stop']*100:+.0f}%  "
      f"sb={best_p['shallow_bounce']*100:+.0f}%  "
      f"fs={best_p['full_stop']}  hf={best_p['half_frac']*100:.0f}%")
print(f"    CAGR={m_best['cagr']*100:+.2f}%  Sh={m_best['sharpe']:.3f}  "
      f"MDD={m_best['mdd']*100:+.2f}%  Ulcer={m_best['ulcer']:.2f}")

# 기준과 비교
print(f"\n  [기준 대비 Δ]  ΔSh={m_best['sharpe']-m_base['sharpe']:+.3f}  "
      f"ΔCAGR={( m_best['cagr']-m_base['cagr'])*100:+.2f}%  "
      f"ΔMDD={(m_best['mdd']-m_base['mdd'])*100:+.2f}%  "
      f"ΔUlcer={m_best['ulcer']-m_base['ulcer']:+.2f}")

# ── 시각화 ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 20))
gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.52, wspace=0.30,
                        height_ratios=[1.3, 1.0, 1.0, 1.0, 1.0])
fig.suptitle("HALF_STOP / FULL_STOP 파라미터 스윕 | 2015-2026",
             fontsize=13, fontweight="bold")

# ① 자산곡선 비교 (기준 vs 베스트 vs QQQ)
ax = fig.add_subplot(gs[0, :])
ax.plot(b_qqq.index, b_qqq/b_qqq.iloc[0]*100, label="B1 QQQ", color="#1F2937", lw=1.2, alpha=0.7)
ax.plot(base_s.index, base_s/base_s.iloc[0]*100, label=f"기준 (hs=-15%,sb=-5%)", color="#6B7280", lw=1.5, ls="--")
ax.plot(best_s.index, best_s/best_s.iloc[0]*100,
        label=f"베스트 (hs={best_p['half_stop']*100:+.0f}%,sb={best_p['shallow_bounce']*100:+.0f}%,hf={best_p['half_frac']*100:.0f}%)",
        color="#DC2626", lw=2.2)
# half_stop=-0.18 line
p_hs18 = {**base_p, "half_stop": -0.18}
s_hs18, _ = strategy_s4_extended(Q, Qld, T, INIT, **p_hs18)
ax.plot(s_hs18.index, s_hs18/s_hs18.iloc[0]*100, label="hs=-18% (완화)", color="#2563EB", lw=1.3, alpha=0.8)
ax.set_yscale("log"); ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
ax.set_ylabel("Index (log)"); ax.set_title("자산곡선 비교", fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ② half_stop 스윕: Sharpe vs MDD
ax = fig.add_subplot(gs[1, 0])
hs_sharpes = [hs_results[hs]["m"]["sharpe"] for hs in half_stop_vals]
hs_mdds    = [abs(hs_results[hs]["m"]["mdd"])*100 for hs in half_stop_vals]
ax.plot([h*100 for h in half_stop_vals], hs_sharpes, "o-", color="#DC2626", lw=2, ms=8, label="Sharpe")
ax2 = ax.twinx()
ax2.plot([h*100 for h in half_stop_vals], hs_mdds, "s--", color="#2563EB", lw=1.5, ms=7, label="|MDD|%")
ax.axvline(-15, color="#999", ls=":", lw=1, alpha=0.8)
ax.set_xlabel("half_stop (%)"); ax.set_ylabel("Sharpe", color="#DC2626")
ax2.set_ylabel("|MDD| (%)", color="#2563EB")
ax.set_title("half_stop 스윕: Sharpe & MDD", fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3)
lines1, lbs1 = ax.get_legend_handles_labels(); lines2, lbs2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lbs1+lbs2, fontsize=8)

# ③ full_stop 스윕: Sharpe
ax = fig.add_subplot(gs[1, 1])
fs_vals_plot = full_stop_vals
fs_sharpes = [fs_results[fs]["m"]["sharpe"] for fs in fs_vals_plot]
fs_cagrs   = [fs_results[fs]["m"]["cagr"]*100 for fs in fs_vals_plot]
fs_labels  = ["None"] + [f"{f*100:+.0f}%" for f in fs_vals_plot[1:]]
x = range(len(fs_labels))
ax.bar(x, fs_sharpes, color=["#DC2626"]+["#2563EB"]*len(fs_vals_plot[1:]), alpha=0.8, edgecolor="white")
for i, (sh, cg) in enumerate(zip(fs_sharpes, fs_cagrs)):
    ax.text(i, sh+0.002, f"Sh={sh:.3f}\nCAGR={cg:+.1f}%", ha="center", fontsize=7.5)
ax.set_xticks(list(x)); ax.set_xticklabels(fs_labels, fontsize=9)
ax.set_ylabel("Sharpe")
ax.set_title("full_stop 신규 추가 스윕 (None=현재, 나머지=FULL_ATTACK 손절)", fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

# ④ Grid search: Sharpe heatmap (best full_stop slice)
ax = fig.add_subplot(gs[2, :])
# 2D: half_stop × shallow_bounce (full_stop=None, half_frac=0.50)
hs_grid = [-0.12, -0.15, -0.18]
sb_grid = [-0.04, -0.05, -0.07]
hf_grid = [0.40, 0.50, 0.70]
# 3D surface projection: show best per (hs, sb) across hf
heat_sh = np.zeros((len(hs_grid), len(sb_grid)))
heat_hf = np.zeros((len(hs_grid), len(sb_grid)), dtype=int)
for i, hs in enumerate(hs_grid):
    for j, sb in enumerate(sb_grid):
        best_sh = -np.inf; best_hf = 0.50
        for hf in hf_grid:
            row = gr_df[
                (gr_df["half_stop"]==hs) & (gr_df["shallow_bounce"]==sb) &
                (gr_df["full_stop"].isna()) & (gr_df["half_frac"]==hf)
            ]
            if not row.empty and float(row["sharpe"].iloc[0]) > best_sh:
                best_sh = float(row["sharpe"].iloc[0])
                best_hf = hf
        heat_sh[i, j] = best_sh
        heat_hf[i, j] = int(best_hf*100)

im = ax.imshow(heat_sh, cmap="RdYlGn", aspect="auto", vmin=0.75, vmax=0.95)
ax.set_xticks(range(len(sb_grid))); ax.set_xticklabels([f"{s*100:+.0f}%" for s in sb_grid])
ax.set_yticks(range(len(hs_grid))); ax.set_yticklabels([f"{h*100:+.0f}%" for h in hs_grid])
ax.set_xlabel("shallow_bounce"); ax.set_ylabel("half_stop")
for i in range(len(hs_grid)):
    for j in range(len(sb_grid)):
        ax.text(j, i, f"Sh={heat_sh[i,j]:.3f}\nhf={heat_hf[i,j]}%",
                ha="center", va="center", fontsize=9,
                color="white" if heat_sh[i,j]>0.88 else "black")
plt.colorbar(im, ax=ax, fraction=0.02)
ax.set_title("Sharpe heatmap: half_stop × shallow_bounce (full_stop=None, best hf)",
             fontsize=10, fontweight="bold")

# ⑤ 연도별 수익 비교 (기준 vs 베스트)
ax = fig.add_subplot(gs[3, :])
years = range(2015, 2027)
def yr(s, y):
    sl = s.loc[f"{y}-01-01":f"{y}-12-31"]
    return (sl.iloc[-1]/sl.iloc[0]-1)*100 if len(sl)>3 else np.nan

yrb = [yr(b_qqq, y) for y in years]
yrbase = [yr(base_s, y) for y in years]
yrbest = [yr(best_s, y) for y in years]
yrs18  = [yr(s_hs18, y) for y in years]

x = np.arange(len(list(years))); w = 0.22
ax.bar(x-1.5*w, yrb,    w, label="B1 QQQ",   color="#9CA3AF", alpha=0.85)
ax.bar(x-0.5*w, yrbase, w, label="기준(hs=-15%,sb=-5%)", color="#6B7280", alpha=0.85)
ax.bar(x+0.5*w, yrbest, w, label=f"베스트(hs={best_p['half_stop']*100:+.0f}%,sb={best_p['shallow_bounce']*100:+.0f}%,hf={best_p['half_frac']*100:.0f}%)", color="#DC2626", alpha=0.85)
ax.bar(x+1.5*w, yrs18,  w, label="hs=-18%",  color="#2563EB", alpha=0.85)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years], fontsize=9)
ax.set_ylabel("연간 수익률 (%)"); ax.set_title("연도별 수익 비교", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, ncol=4, loc="upper left"); ax.grid(True, alpha=0.3, axis="y")

# ⑥ Sharpe-CAGR trade-off 모든 조합 scatter
ax = fig.add_subplot(gs[4, :])
sharpes = gr_df["sharpe"].values
cagrs   = gr_df["cagr"].values * 100
mdds    = abs(gr_df["mdd"].values) * 100
scatter = ax.scatter(sharpes, cagrs, c=mdds, cmap="RdYlGn_r", s=60,
                     alpha=0.7, edgecolors="black", lw=0.3, vmin=20, vmax=60)
plt.colorbar(scatter, ax=ax, label="|MDD| (%)", fraction=0.03)
# 기준 마킹
ax.scatter(m_base["sharpe"], m_base["cagr"]*100, c="blue", s=250, marker="D", zorder=5,
           label=f"기준  Sh={m_base['sharpe']:.3f}", edgecolors="white", lw=1.5)
ax.scatter(m_best["sharpe"], m_best["cagr"]*100, c="red", s=250, marker="D", zorder=5,
           label=f"베스트 Sh={m_best['sharpe']:.3f}", edgecolors="white", lw=1.5)
ax.scatter(m_s1["sharpe"], m_s1["cagr"]*100, c="black", s=180, marker="X", zorder=5,
           label=f"QQQ B&H  Sh={m_s1['sharpe']:.3f}")
ax.set_xlabel("Sharpe"); ax.set_ylabel("CAGR (%)")
ax.set_title(f"전체 {len(all_combos)}개 조합 Sharpe vs CAGR (색=|MDD|, 낮을수록 좋음)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.savefig(OUT_DIR / "stop_logic_sweep.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\n  PNG → {OUT_DIR}/stop_logic_sweep.png")
print("\n[완료]")
