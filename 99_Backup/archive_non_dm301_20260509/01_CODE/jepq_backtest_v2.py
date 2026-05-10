"""
jepq_backtest_v2.py
====================
기간: 2015-01-01 ~ 2026-04-30
파라미터: Anchor trail=-15% (나머지 anchor 동일)
방어자산: QQQ / JEPQ-hybrid (2022-05 이전 QQQ, 이후 JEPQ)
벤치마크: B1 QQQ, B2 QQQ/TQQQ 50/50, B3 TQQQ, B4 JEPQ(2022~), B5 JEPQ/TQQQ(2022~)
에피소드 정밀분석 포함
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
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm

from backtest_switching import load_extended_daily, strategy_s4_trailing
from evaluation_metrics import full_metrics, paired_bootstrap_compare

OUT_DIR = Path("03_RESULT")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 파라미터 ─────────────────────────────────────────────────────────────────
PARAMS = dict(
    shallow_drop      = -0.10,
    deep_drop         = -0.20,
    shallow_bounce    = -0.05,
    deep_bounce       = -0.10,
    half_stop         = -0.15,
    trailing_stop_pct = -0.15,   # ★ trail -15%
    half_frac         = 0.50,
    full_frac         = 1.00,
)
START      = pd.Timestamp("2015-01-01")
JEPQ_IPO   = pd.Timestamp("2022-05-04")
END        = pd.Timestamp("2026-04-30")

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_jepq():
    df = pd.read_csv("02_DATA/yahoo_extended/JEPQ/JEPQ_daily.csv",
                     index_col="Date", parse_dates=True)
    c = df["Close"]
    return pd.DataFrame({"Open":c,"High":c,"Low":c,"Close":c}, index=df.index)

qqq  = load_extended_daily("QQQ")
qld  = load_extended_daily("QLD")
tqqq = load_extended_daily("TQQQ")
jepq_raw = load_jepq()

# 공통 인덱스
common_full = qqq.index.intersection(qld.index).intersection(tqqq.index)
common_full = common_full[(common_full >= START) & (common_full <= END)]

Q    = qqq.loc[common_full]
Qld  = qld.loc[common_full]
T    = tqqq.loc[common_full]

# JEPQ-hybrid 방어자산: IPO 이전 QQQ, 이후 JEPQ — 가격 연속성 보장
# 전략은 내부적으로 "보유 주식 수 × 현재 가격"으로 자산 추적.
# 방어 자산을 QQQ→JEPQ로 전환할 때 가격 레벨이 달라도 되지만,
# 전환 시점의 포트폴리오 가치에서 JEPQ 주식 수를 새로 산정하면 OK.
# 그러나 현재 strategy_s4_trailing은 def_close를 일관된 하나의 시리즈로 씀.
# 해결책: QQQ 수익률로 IPO 이전을 구성, 이후 JEPQ 수익률로 이어붙인
# 연속 총수익 지수(chained return index)를 방어자산으로 사용.
jepq_common = jepq_raw.index.intersection(common_full)
pre_jepq    = common_full[common_full < JEPQ_IPO]
post_jepq   = common_full[common_full >= JEPQ_IPO]

# QQQ total-return index (pre-JEPQ 기간)
qqq_tr_pre = Q.loc[pre_jepq, "Close"] / Q.loc[pre_jepq[0], "Close"]
# JEPQ total-return index (post-JEPQ 기간), 100에서 시작
jepq_tr_post = jepq_raw.loc[post_jepq, "Close"] / jepq_raw.loc[post_jepq[0], "Close"]
# 이어붙이기: 전환 시점에서 1.0으로 끊고 QQQ 마지막 수준을 곱해 연속화
last_pre = float(qqq_tr_pre.iloc[-1]) if len(qqq_tr_pre) > 0 else 1.0
hybrid_close = pd.concat([
    qqq_tr_pre * 100,                   # pre-IPO: QQQ 수익 기반
    jepq_tr_post * last_pre * 100,       # post-IPO: JEPQ 수익 기반, 이음새 연속
])
# OHLC 형태로 만들기
J_hybrid = pd.DataFrame({
    "Open": hybrid_close, "High": hybrid_close,
    "Low": hybrid_close, "Close": hybrid_close,
}, index=hybrid_close.index)

INIT = 100_000

print("=" * 70)
print("백테스트 2015–2026  |  Trail stop -15%")
print(f"  기간: {common_full[0].date()} ~ {common_full[-1].date()}  ({len(common_full):,}일)")
print(f"  JEPQ 상장: {JEPQ_IPO.date()}  (이전은 QQQ 방어자산)")
print("=" * 70)

# ── 전략 실행 ─────────────────────────────────────────────────────────────────
strat_qqq,  ev_qqq  = strategy_s4_trailing(Q,        Qld, T, INIT, **PARAMS, series_name="S4 def.QQQ")
strat_jepq, ev_jepq = strategy_s4_trailing(J_hybrid, Qld, T, INIT, **PARAMS, series_name="S4 def.JEPQ-hybrid")

# ── 벤치마크 ─────────────────────────────────────────────────────────────────
def bnh(asset, name, cap=INIT):
    sh = cap / asset["Close"].iloc[0]
    return pd.Series(sh * asset["Close"].values, index=asset.index, name=name)

def half_half(a, b, name, cap=INIT):
    sha = (cap * .5) / a["Close"].iloc[0]
    shb = (cap * .5) / b["Close"].iloc[0]
    return pd.Series(sha * a["Close"].values + shb * b["Close"].values,
                     index=a.index, name=name)

# JEPQ 벤치마크는 공통 기간만 (IPO 이후)
J_oos = jepq_raw.loc[jepq_common]
b_qqq       = bnh(Q,   "B1: QQQ")
b_qqq_tqqq  = half_half(Q, T, "B2: QQQ/TQQQ 50/50")
b_tqqq      = bnh(T,   "B3: TQQQ")
b_jepq      = bnh(J_oos,  "B4: JEPQ")          # 2022-05 이후만
b_jepq_tqqq = half_half(J_oos, T.loc[jepq_common], "B5: JEPQ/TQQQ 50/50")  # 2022-05 이후

# ── 메트릭 ───────────────────────────────────────────────────────────────────
def m(s): return full_metrics(s)

all_full = {
    "B1: QQQ":           b_qqq,
    "B2: QQQ/TQQQ 50/50": b_qqq_tqqq,
    "B3: TQQQ":          b_tqqq,
    "S4 def.QQQ":        strat_qqq,
    "S4 def.JEPQ-hybrid":strat_jepq,
}
all_since_jepq = {
    "B1: QQQ (2022~)":          b_qqq.loc[jepq_common],
    "B4: JEPQ (2022~)":         b_jepq,
    "B5: JEPQ/TQQQ (2022~)":    b_jepq_tqqq,
    "S4 def.QQQ (2022~)":       strat_qqq.loc[jepq_common],
    "S4 def.JEPQ (2022~)":      strat_jepq.loc[jepq_common],
}

print(f"\n{'전략':30s}  {'기간':>15s}  {'CAGR':>8s}  {'Sharpe':>7s}  {'Sortino':>8s}  {'MDD':>8s}  {'Ulcer':>6s}  {'Calmar':>7s}")
print("  " + "-"*110)
for name, s in {**all_full, **{k:v for k,v in all_since_jepq.items() if k not in all_full}}.items():
    if name in all_full:
        period = "2015–2026"
    else:
        period = "2022–2026"
    mx = m(s)
    print(f"  {name:30s}  {period:>15s}  {mx['cagr']*100:>+7.2f}%  {mx['sharpe']:>7.2f}  "
          f"{mx['sortino']:>8.2f}  {mx['mdd']*100:>+7.2f}%  {mx['ulcer']:>6.2f}  {mx['calmar']:>7.2f}")

# ── 에피소드 정밀분석 ─────────────────────────────────────────────────────────
def build_episodes(events_df: pd.DataFrame, strat_series: pd.Series,
                   qqq_series: pd.Series, tqqq_series: pd.Series) -> pd.DataFrame:
    """이벤트 로그 → 완결된 에피소드 목록."""
    if events_df.empty:
        return pd.DataFrame()

    ev = events_df.copy()
    ev["Date"] = pd.to_datetime(ev["Date"])
    ev = ev.sort_values("Date").reset_index(drop=True)

    ENTRY_TYPES  = {"TO_HALF_ATTACK", "TO_FULL_ATTACK"}
    EXIT_TYPES   = {"TRAIL_EXIT", "TRAIL_FLOOR", "HALF_STOP"}
    UPGRADE      = {"TO_FULL_ATTACK"}

    episodes = []
    i = 0
    while i < len(ev):
        if ev.loc[i, "type"] not in ENTRY_TYPES:
            i += 1
            continue
        entry_date  = ev.loc[i, "Date"]
        entry_type  = ev.loc[i, "type"]
        entry_val   = ev.loc[i, "value"]
        upgraded    = False

        j = i + 1
        while j < len(ev) and ev.loc[j, "type"] not in EXIT_TYPES:
            if ev.loc[j, "type"] in UPGRADE:
                upgraded = True
            j += 1

        if j >= len(ev):
            break
        exit_date = ev.loc[j, "Date"]
        exit_type = ev.loc[j, "type"]
        exit_val  = ev.loc[j, "value"]

        # 에피소드 슬라이스
        mask = (strat_series.index >= entry_date) & (strat_series.index <= exit_date)
        s_ep = strat_series.loc[mask]
        q_ep = qqq_series.loc[mask] if qqq_series.index.isin([entry_date]).any() else \
               qqq_series.reindex(s_ep.index, method="ffill")
        t_ep = tqqq_series.reindex(s_ep.index, method="ffill")

        # 에피소드 수익
        strat_ret  = (s_ep.iloc[-1] / s_ep.iloc[0] - 1) if len(s_ep) > 0 else 0
        qqq_ret    = (q_ep.iloc[-1] / q_ep.iloc[0] - 1) if len(q_ep) > 0 else 0
        tqqq_ret   = (t_ep.iloc[-1] / t_ep.iloc[0] - 1) if len(t_ep) > 0 else 0
        duration   = (exit_date - entry_date).days
        n_trading  = int(mask.sum())

        episodes.append({
            "episode_no":  len(episodes) + 1,
            "entry_date":  entry_date,
            "exit_date":   exit_date,
            "duration_days": duration,
            "n_trading_days": n_trading,
            "entry_type":  entry_type,
            "exit_type":   exit_type,
            "upgraded":    upgraded,
            "entry_val":   entry_val,
            "exit_val":    exit_val,
            "strat_ret":   strat_ret,
            "qqq_ret":     qqq_ret,
            "tqqq_ret":    tqqq_ret,
            "alpha_vs_qqq": strat_ret - qqq_ret,
            "win":         strat_ret > qqq_ret,
        })
        i = j + 1

    return pd.DataFrame(episodes)


ep_qqq  = build_episodes(ev_qqq,  strat_qqq,  b_qqq, T["Close"])
ep_jepq = build_episodes(ev_jepq, strat_jepq, b_qqq, T["Close"])

def print_episodes(ep: pd.DataFrame, title: str):
    if ep.empty:
        print(f"\n  [{title}] 에피소드 없음")
        return
    print(f"\n{'='*100}")
    print(f"  [{title}]  총 {len(ep)}개 에피소드")
    print(f"{'='*100}")
    hdr = (f"  {'No':>3}  {'진입':>12}  {'청산':>12}  {'기간(일)':>8}  "
           f"{'진입유형':>14}  {'청산유형':>12}  {'업그레이드':>6}  "
           f"{'전략수익':>9}  {'QQQ수익':>9}  {'TQQQ수익':>9}  {'vs QQQ':>9}  {'승패':>4}")
    print(hdr)
    print("  " + "-"*120)
    ep2 = ep.reset_index(drop=True)
    win_count = 0
    for row_idx in range(len(ep2)):
        no     = int(ep2.at[row_idx, "episode_no"])
        ed     = ep2.at[row_idx, "entry_date"]
        xd     = ep2.at[row_idx, "exit_date"]
        dur    = int(ep2.at[row_idx, "duration_days"])
        etyp   = str(ep2.at[row_idx, "entry_type"])
        xtyp   = str(ep2.at[row_idx, "exit_type"])
        upg    = bool(ep2.at[row_idx, "upgraded"])
        sr     = float(ep2.at[row_idx, "strat_ret"])
        qr     = float(ep2.at[row_idx, "qqq_ret"])
        tr_    = float(ep2.at[row_idx, "tqqq_ret"])
        al     = float(ep2.at[row_idx, "alpha_vs_qqq"])
        win    = bool(ep2.at[row_idx, "win"])
        flag   = "✅" if win else "❌"
        upg_s  = "⬆" if upg else " "
        print(f"  {no:>3}  "
              f"{str(ed.date()):>12}  "
              f"{str(xd.date()):>12}  "
              f"{dur:>8}  "
              f"{etyp:>14}  "
              f"{xtyp:>12}  "
              f"{upg_s:>6}  "
              f"{sr*100:>+8.2f}%  "
              f"{qr*100:>+8.2f}%  "
              f"{tr_*100:>+8.2f}%  "
              f"{al*100:>+8.2f}%  "
              f"{flag:>4}")
        if win:
            win_count += 1
    print("  " + "-"*120)
    total_strat = ep["strat_ret"].mean()
    total_qqq   = ep["qqq_ret"].mean()
    total_alpha = ep["alpha_vs_qqq"].mean()
    print(f"  {'평균':>70}  {total_strat*100:>+8.2f}%  "
          f"{total_qqq*100:>+8.2f}%  "
          f"{'':>9}  "
          f"{total_alpha*100:>+8.2f}%  "
          f"{win_count}/{len(ep)} ({win_count/len(ep)*100:.0f}%)")

    # Exit type 집계
    print(f"\n  [청산 유형 분석]")
    for etype in ["TRAIL_EXIT", "TRAIL_FLOOR", "HALF_STOP"]:
        sub = ep[ep["exit_type"] == etype]
        if sub.empty:
            continue
        wins = int(sub["win"].sum())
        avg_ret = float(sub["strat_ret"].mean()) * 100
        avg_alpha = float(sub["alpha_vs_qqq"].mean()) * 100
        print(f"    {etype:12s}  {len(sub):2}건  승률={wins}/{len(sub)}  "
              f"평균수익={avg_ret:>+6.2f}%  평균alpha={avg_alpha:>+6.2f}%")

    # Entry type 집계
    print(f"\n  [진입 유형 분석]")
    for etype in ["TO_HALF_ATTACK", "TO_FULL_ATTACK"]:
        sub = ep[ep["entry_type"] == etype]
        if sub.empty:
            continue
        wins = int(sub["win"].sum())
        avg_ret = float(sub["strat_ret"].mean()) * 100
        print(f"    {etype:16s}  {len(sub):2}건  승률={wins}/{len(sub)}  평균수익={avg_ret:>+6.2f}%")

print_episodes(ep_qqq,  "S4 def.QQQ  — 에피소드 상세 (2015-2026)")
print_episodes(ep_jepq, "S4 def.JEPQ-hybrid — 에피소드 상세 (2015-2026)")

# ── 연도별 성과 ───────────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print("  [연도별 CAGR]")
print("  " + "-"*100)
print(f"  {'Year':>6}  {'B1 QQQ':>9}  {'B2 50/50':>9}  {'B3 TQQQ':>9}  "
      f"{'S4 def.QQQ':>12}  {'S4 def.JEPQ':>13}  {'alpha(S4vQQQ)':>15}")
print("  " + "-"*100)
for year in range(2015, 2027):
    ys = pd.Timestamp(f"{year}-01-01")
    ye = pd.Timestamp(f"{year}-12-31")
    def yr_ret(s):
        sl = s.loc[(s.index >= ys) & (s.index <= ye)]
        if len(sl) < 5:
            return None
        return sl.iloc[-1] / sl.iloc[0] - 1
    rets = {n: yr_ret(s) for n, s in {
        "B1: QQQ": b_qqq, "B2: QQQ/TQQQ 50/50": b_qqq_tqqq,
        "B3: TQQQ": b_tqqq,
        "S4 def.QQQ": strat_qqq, "S4 def.JEPQ-hybrid": strat_jepq,
    }.items()}
    r = rets
    def fmt(v):
        if v is None: return "   —    "
        return f"{v*100:>+7.1f}%"
    alpha = None if r["S4 def.QQQ"] is None or r["B1: QQQ"] is None else \
            r["S4 def.QQQ"] - r["B1: QQQ"]
    print(f"  {year:>6}  {fmt(r['B1: QQQ']):>9}  {fmt(r['B2: QQQ/TQQQ 50/50']):>9}  "
          f"{fmt(r['B3: TQQQ']):>9}  {fmt(r['S4 def.QQQ']):>12}  "
          f"{fmt(r['S4 def.JEPQ-hybrid']):>13}  "
          f"{('   —    ' if alpha is None else f'{alpha*100:>+7.1f}%'):>15}")

# ── 시각화 ───────────────────────────────────────────────────────────────────
COLORS = {
    "B1: QQQ":             "#1F2937",
    "B2: QQQ/TQQQ 50/50":  "#6B7280",
    "B3: TQQQ":            "#9CA3AF",
    "B4: JEPQ (2022~)":    "#2563EB",
    "B5: JEPQ/TQQQ (2022~)":"#60A5FA",
    "S4 def.QQQ":          "#DC2626",
    "S4 def.JEPQ-hybrid":  "#7C3AED",
}

fig = plt.figure(figsize=(18, 22))
gs_main = gridspec.GridSpec(5, 2, figure=fig, hspace=0.52, wspace=0.28,
                             height_ratios=[1.4, 1.0, 1.0, 1.0, 1.2])
fig.suptitle(
    f"QQQ-TQQQ Switching | Anchor trail=−8% | 2015–2026\n"
    f"방어자산: QQQ (전체) / JEPQ-hybrid (2022-05 이후 JEPQ, 이전 QQQ)",
    fontsize=13, fontweight="bold")

# ① 자산곡선 (전체)
ax = fig.add_subplot(gs_main[0, :])
for name, s in {**all_full}.items():
    ax.plot(s.index, s / s.iloc[0] * 100, label=name,
            color=COLORS.get(name, "#000"), lw=2.2 if "S4" in name else 1.2,
            alpha=0.95 if "S4" in name else 0.70,
            ls="-" if "S4" in name or name=="B1: QQQ" else ("--" if "50/50" in name else ":"))
# JEPQ 벤치마크 (2022~ 시점 정규화)
for name, s in all_since_jepq.items():
    if "B4" in name or "B5" in name:
        ax.plot(s.index, s / s.iloc[0] * 100, label=name,
                color=COLORS.get(name, "#3B82F6"), lw=1.0, alpha=0.70,
                ls="--" if "50/50" in name else "-")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
ax.axvline(JEPQ_IPO, color="#2563EB", ls=":", lw=1.0, alpha=0.5)
ax.text(JEPQ_IPO, ax.get_ylim()[0]*1.1 if ax.get_ylim()[0]>0 else 60,
        "JEPQ IPO", fontsize=8, color="#2563EB", alpha=0.8, rotation=90, va="bottom")
ax.set_ylabel("Index (start=100, log)")
ax.set_title("자산곡선 (log scale, JEPQ 벤치마크는 상장일 기준 100)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, ncol=3, loc="upper left"); ax.grid(True, alpha=0.3)

# ② Drawdown
ax = fig.add_subplot(gs_main[1, :])
for name, s in all_full.items():
    dd = (s - s.cummax()) / s.cummax() * 100
    ax.plot(s.index, dd, label=name,
            color=COLORS.get(name, "#000"), lw=2.0 if "S4" in name else 1.0,
            alpha=0.9 if "S4" in name else 0.60)
ax.axvline(JEPQ_IPO, color="#2563EB", ls=":", lw=0.8, alpha=0.4)
ax.set_ylabel("Drawdown (%)"); ax.set_title("Drawdown 비교", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, ncol=3, loc="lower left"); ax.grid(True, alpha=0.3)

# ③ 에피소드별 수익 bar (S4 def.QQQ)
ax = fig.add_subplot(gs_main[2, :])
if not ep_qqq.empty:
    colors_ep = ["#10B981" if w else "#EF4444" for w in ep_qqq["win"]]
    bars = ax.bar(range(len(ep_qqq)), ep_qqq["strat_ret"]*100,
                  color=colors_ep, edgecolor="white", lw=0.5, label="S4 strat ret")
    ax.plot(range(len(ep_qqq)), ep_qqq["qqq_ret"]*100,
            "o--", color="#1F2937", ms=6, lw=1.2, label="QQQ 동기간", alpha=0.8)
    ax.plot(range(len(ep_qqq)), ep_qqq["tqqq_ret"]*100,
            "^:", color="#6B7280", ms=5, lw=0.8, label="TQQQ 동기간", alpha=0.6)
    ax.axhline(0, color="black", lw=0.8)

    # 에피소드 번호 + 날짜 레이블
    for i, (_, r) in enumerate(ep_qqq.iterrows()):
        ax.text(i, max(r["strat_ret"]*100, 0) + 0.5,
                f"#{int(r['episode_no'])}\n{r['entry_date'].strftime('%y.%m')}",
                ha="center", fontsize=7, va="bottom", color="#111")
        # exit type 표시
        etype_short = {"TRAIL_EXIT":"TR", "TRAIL_FLOOR":"TF", "HALF_STOP":"HS"}
        ax.text(i, min(r["strat_ret"]*100, 0) - 0.5,
                etype_short.get(r["exit_type"], "?"),
                ha="center", fontsize=6.5, va="top", color="#555")

    win_n = ep_qqq["win"].sum(); total = len(ep_qqq)
    ax.set_ylabel("에피소드 수익률 (%)"); ax.set_xticks([])
    ax.set_title(f"S4 def.QQQ — 에피소드별 수익  (승리={win_n}/{total}={win_n/total*100:.0f}%,  "
                 f"TR=TrailExit, TF=TrailFloor, HS=HalfStop)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3, axis="y")

# ④ 에피소드 scatter: 기간 vs 수익
ax = fig.add_subplot(gs_main[3, 0])
if not ep_qqq.empty:
    cmap = cm.RdYlGn; norm = Normalize(vmin=-0.20, vmax=0.30)
    for _, r in ep_qqq.iterrows():
        c = cmap(norm(r["strat_ret"]))
        ax.scatter(r["duration_days"], r["strat_ret"]*100, s=200,
                   color=c, edgecolors="black", lw=0.7, alpha=0.9)
        ax.annotate(f"#{int(r['episode_no'])}", (r["duration_days"], r["strat_ret"]*100),
                    fontsize=7.5, xytext=(3,3), textcoords="offset points")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("에피소드 기간 (calendar days)")
    ax.set_ylabel("전략 수익률 (%)")
    ax.set_title("기간 vs 수익 scatter  (색=수익: 녹=양, 적=음)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

# ⑤ Alpha vs QQQ scatter
ax = fig.add_subplot(gs_main[3, 1])
if not ep_qqq.empty:
    for _, r in ep_qqq.iterrows():
        c = "#10B981" if r["win"] else "#EF4444"
        ax.scatter(r["qqq_ret"]*100, r["alpha_vs_qqq"]*100, s=200,
                   color=c, edgecolors="black", lw=0.7, alpha=0.9)
        ax.annotate(f"#{int(r['episode_no'])}", (r["qqq_ret"]*100, r["alpha_vs_qqq"]*100),
                    fontsize=7.5, xytext=(3,3), textcoords="offset points")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("QQQ 동기간 수익 (%)")
    ax.set_ylabel("Alpha vs QQQ (%)")
    ax.set_title("Alpha vs QQQ — QQQ 하락장/상승장별 우위",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    green_p = mpatches.Patch(color="#10B981", label="Win (strategy > QQQ)")
    red_p   = mpatches.Patch(color="#EF4444", label="Lose (strategy < QQQ)")
    ax.legend(handles=[green_p, red_p], fontsize=8)

# ⑥ 연도별 bar
ax = fig.add_subplot(gs_main[4, :])
years = list(range(2015, 2027))
year_rets = {}
for name, s in all_full.items():
    yr_r = []
    for yr in years:
        sl = s.loc[f"{yr}-01-01":f"{yr}-12-31"]
        yr_r.append((sl.iloc[-1]/sl.iloc[0]-1)*100 if len(sl)>5 else np.nan)
    year_rets[name] = yr_r

x = np.arange(len(years)); n = len(all_full); w = 0.16
palette = ["#1F2937","#6B7280","#9CA3AF","#DC2626","#7C3AED"]
for i, (name, rets) in enumerate(year_rets.items()):
    ax.bar(x + (i-n//2)*w, rets, w,
           label=name, color=palette[i], alpha=0.85, edgecolor="white", lw=0.4)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years], fontsize=9)
ax.set_ylabel("연간 수익률 (%)")
ax.set_title("연도별 수익률 비교", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, ncol=5, loc="upper left"); ax.grid(True, alpha=0.3, axis="y")

plt.savefig(OUT_DIR / "jepq_backtest_v2.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 에피소드 상세 시각화 (에피소드별 가격 트래킹) ────────────────────────────
if not ep_qqq.empty:
    n_ep = len(ep_qqq)
    ncols = 4
    nrows = (n_ep + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(18, 4.5 * nrows))
    fig2.suptitle("S4 def.QQQ — 에피소드별 상세 (QLD ATH대비 낙폭 + 포트폴리오 vs QQQ)",
                  fontsize=12, fontweight="bold")
    axes2_flat = axes2.flatten() if nrows > 1 else [axes2] if ncols == 1 else axes2.flatten()

    for idx, (_, r) in enumerate(ep_qqq.iterrows()):
        ax = axes2_flat[idx]
        ed = r["entry_date"]; xd = r["exit_date"]
        # 에피소드 주변 ±30일
        buf = pd.Timedelta(days=30)
        w_start = max(ed - buf, Qld.index[0])
        w_end   = min(xd + buf, Qld.index[-1])
        mask = (Qld.index >= w_start) & (Qld.index <= w_end)

        qld_w = Qld.loc[mask, "Close"]
        qqq_w = Q.loc[mask, "Close"]
        tqqq_w = T.loc[mask, "Close"]
        strat_w = strat_qqq.loc[mask]

        # QLD ATH대비 낙폭 (창 내 ATH 기준)
        ath = Qld.loc[Qld.index <= ed, "Close"].max()
        dd_w = (qld_w - ath) / ath * 100

        ax2 = ax.twinx()
        # 포트폴리오 & QQQ (정규화)
        ref_s = strat_w.iloc[0]; ref_q = qqq_w.iloc[0]
        ax.plot(strat_w.index, strat_w/ref_s*100,
                color="#DC2626", lw=1.8, label="S4", zorder=3)
        ax.plot(qqq_w.index, qqq_w/ref_q*100,
                color="#1F2937", lw=1.2, ls="--", alpha=0.7, label="QQQ", zorder=2)

        # QLD 낙폭 (우축)
        ax2.fill_between(dd_w.index, dd_w, 0,
                         where=dd_w<0, color="#6B7280", alpha=0.15)
        ax2.plot(dd_w.index, dd_w, color="#6B7280", lw=0.8, alpha=0.7, label="QLD DD%")
        ax2.axhline(-10, color="#F59E0B", ls=":", lw=0.7, alpha=0.7)
        ax2.axhline(-20, color="#EF4444", ls=":", lw=0.7, alpha=0.7)
        ax2.set_ylabel("QLD DD (%)", fontsize=7, color="#6B7280")

        # 진입/청산 마킹
        for d, c, m_s, lbl in [
            (ed, "#F59E0B", "^", "Entry"), (xd, "#10B981" if r["win"] else "#EF4444", "v", "Exit")
        ]:
            if d in strat_w.index:
                ax.scatter([d], [strat_w.loc[d]/ref_s*100], color=c, s=120, marker=m_s,
                           zorder=5, linewidths=0)

        # 에피소드 구간 음영
        ep_mask = (strat_w.index >= ed) & (strat_w.index <= xd)
        ep_dates = strat_w.index[ep_mask]
        if len(ep_dates) > 1:
            ax.axvspan(ep_dates[0], ep_dates[-1], color="#FEF3C7", alpha=0.4, zorder=1)

        win_str = "✅" if r["win"] else "❌"
        upg_str = "↑FULL" if r["upgraded"] else ""
        ax.set_title(
            f"#{int(r['episode_no'])} {ed.strftime('%y.%m.%d')}→{xd.strftime('%y.%m.%d')} {win_str}\n"
            f"{r['entry_type'].replace('TO_','')}{upg_str}→{r['exit_type']}  "
            f"strat={r['strat_ret']*100:+.1f}%  QQQ={r['qqq_ret']*100:+.1f}%  "
            f"α={r['alpha_vs_qqq']*100:+.1f}%",
            fontsize=7.5, fontweight="bold")
        ax.set_ylabel("Index (start=100)", fontsize=7)
        ax.tick_params(axis="x", labelsize=6.5, rotation=20)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=6.5, loc="upper left")

    for idx in range(n_ep, len(axes2_flat)):
        axes2_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "jepq_backtest_v2_episodes.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

# ── JSON 저장 ────────────────────────────────────────────────────────────────
def _j(o):
    if isinstance(o, dict):  return {k: _j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_j(x) for x in o]
    if isinstance(o, (np.floating, float)): return float(o)
    if isinstance(o, (np.integer, int)):    return int(o)
    if isinstance(o, pd.Timestamp):         return str(o.date())
    if isinstance(o, bool):                 return bool(o)
    return o

out = {
    "params": PARAMS,
    "period_full": {"start": str(common_full[0].date()), "end": str(common_full[-1].date())},
    "metrics_full": {name: m(s) for name, s in all_full.items()},
    "metrics_jepq_period": {name: m(s) for name, s in all_since_jepq.items()},
    "episodes_qqq":  ep_qqq.to_dict(orient="records") if not ep_qqq.empty else [],
    "episodes_jepq": ep_jepq.to_dict(orient="records") if not ep_jepq.empty else [],
}
with open(OUT_DIR / "jepq_backtest_v2.json", "w") as f:
    json.dump(_j(out), f, indent=2, default=str)

print(f"\n  메인 PNG  → {OUT_DIR}/jepq_backtest_v2.png")
print(f"  에피소드  → {OUT_DIR}/jepq_backtest_v2_episodes.png")
print(f"  JSON      → {OUT_DIR}/jepq_backtest_v2.json")
print("\n[완료]")
