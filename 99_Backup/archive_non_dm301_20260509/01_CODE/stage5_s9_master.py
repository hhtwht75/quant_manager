"""
stage5_s9_master.py
===================
[단계 5] S9 Master = S6 Ensemble (Posterior anchors) + VIX guard + Yield-curve tilt.

설계
----
**기본 base**: S6 Anchor Ensemble — 5 sub (Posterior anchor 주변에 분포)
              각 sub 1/5 자본, 매년 1월 균등 리밸런스

**Macro overlay** (각 sub의 진입 의사결정 시점에 적용):

  1) VIX Catastrophic Guard:
     • VIX_now > 99th percentile (252d rolling) → 강제 NORMAL (진입 X)
     • 예: 2008-09 Lehman, 2020-03 COVID, 2018-02 Volmageddon
     • 핵심: "감점" 시그널 — 알파를 추가하지 않고 catastrophic risk 만 차단

  2) Yield Curve Tilt (10Y - 3M):
     • Inverted (spread < 0)  → hf × 0.5,  trail × 1.5 (보수적 진입, 빠른 청산)
     • Normal (spread > +0.5%) → 변경 없음
     • 그 사이 → hf × 0.75, trail × 1.25 (약한 보수적)

이 두 macro 시그널은 **base S4 신호와 직교**:
  • QLD drawdown: equity-internal mean reversion
  • VIX: option-implied vol regime
  • Yield: macro recession risk

→ 작은 알파의 통계적 robustness 가 누적됨.
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

from backtest_switching import load_extended_daily, strategy_buy_and_hold
from evaluation_metrics import full_metrics, paired_bootstrap_compare

OUT_DIR = Path("03_RESULT/sensitivity")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Posterior anchor (단계 3 결과) ────────────────────────────────────────────
POSTERIOR_BASE = {
    "b":     -0.0469,
    "r1":     2.00,
    "r2":     1.978,
    "trail": -0.0836,
    "hf":     0.514,
    "ff":     1.00,
}

# 5 sub (posterior 중심으로 ensemble)
SUB_ANCHORS = [
    {"name": "S1_main",  **POSTERIOR_BASE},
    {"name": "S2_loose", **{**POSTERIOR_BASE, "b": -0.07,  "trail": -0.12}},
    {"name": "S3_sens",  **{**POSTERIOR_BASE, "b": -0.03,  "trail": -0.07}},
    {"name": "S4_fast",  **{**POSTERIOR_BASE, "trail": -0.06}},
    {"name": "S5_slow",  **{**POSTERIOR_BASE, "trail": -0.13}},
]


# ── Macro signal loader ──────────────────────────────────────────────────────
def load_macro_signals():
    vix = pd.read_csv("02_DATA/yahoo_extended/VIX/VIX_daily.csv",
                       index_col="Date", parse_dates=True)["Close"]
    vix.name = "VIX"
    yld = pd.read_csv("02_DATA/yahoo_extended/YIELD/YIELD_daily.csv",
                       index_col="Date", parse_dates=True)
    yld_spread = yld["SPREAD_10Y_3M"]
    yld_spread.name = "SPREAD"
    return vix, yld_spread


def compute_macro_overlay(vix: pd.Series, yld_spread: pd.Series,
                           lookback: int = 252) -> pd.DataFrame:
    """매일의 macro regime 라벨 + multiplier."""
    vix_rank = vix.rolling(lookback, min_periods=lookback//4).rank(pct=True)
    yld_aligned = yld_spread.reindex(vix.index, method="ffill")

    # VIX guard
    vix_guard = (vix_rank > 0.99).astype(int)   # 1 = 진입 차단

    # Yield tilt
    hf_mult    = pd.Series(1.0, index=vix.index)
    trail_mult = pd.Series(1.0, index=vix.index)
    inverted   = (yld_aligned < 0).fillna(False)
    flat       = ((yld_aligned >= 0) & (yld_aligned < 0.005)).fillna(False)

    hf_mult[inverted]  = 0.5
    trail_mult[inverted] = 1.5
    hf_mult[flat]      = 0.75
    trail_mult[flat]   = 1.25

    return pd.DataFrame({
        "vix": vix.values,
        "vix_rank": vix_rank.values,
        "vix_guard": vix_guard.values,
        "yld_spread": yld_aligned.values,
        "hf_mult": hf_mult.values,
        "trail_mult": trail_mult.values,
    }, index=vix.index)


# ── Single sub-strategy with macro overlay ───────────────────────────────────
def strategy_s4_macro_aware(
    qqq, qld, tqqq, macro_df, initial_capital, sub,
):
    """단일 sub anchor + macro overlay 적용 S4-trailing."""
    dates = qqq.index
    portfolio_values = []
    events = []

    def_shares  = initial_capital / qqq["Close"].iloc[0]
    tqqq_shares = 0.0
    state       = "NORMAL"
    ath         = qld["Close"].iloc[0]
    touched_10  = False
    touched_20  = False

    tqqq_trail_peak  = 0.0
    tqqq_trail_entry = 0.0
    cur_trail        = sub["trail"]

    sd_b = sub["b"] * sub["r1"]
    db_b = sub["b"] * sub["r2"]
    dd_b = sub["b"] * sub["r1"] * sub["r2"]
    hs_b = (sd_b + dd_b) / 2

    macro_idx = macro_df.index

    for i, date in enumerate(dates):
        qld_close  = qld["Close"].iloc[i]
        qqq_close  = qqq["Close"].iloc[i]
        tqqq_close = tqqq["Close"].iloc[i]

        # macro 조회
        loc = macro_idx.searchsorted(date, "right") - 1
        if loc < 0:
            vix_guard = 0
            hf_mult, trail_mult = 1.0, 1.0
        else:
            row = macro_df.iloc[loc]
            vix_guard = int(row["vix_guard"])
            hf_mult = float(row["hf_mult"])
            trail_mult = float(row["trail_mult"])

        if qld_close > ath:
            ath = qld_close
        dd = (qld_close - ath) / ath

        if dd <= sd_b: touched_10 = True
        if dd <= dd_b: touched_20 = True

        total_value = def_shares * qqq_close + tqqq_shares * tqqq_close

        # ── TRAILING ──
        if state == "TRAILING":
            if tqqq_close > tqqq_trail_peak:
                tqqq_trail_peak = tqqq_close
            if tqqq_close <= tqqq_trail_entry:
                def_shares  = total_value / qqq_close
                tqqq_shares = 0.0
                state       = "NORMAL"
                touched_10 = touched_20 = False
                events.append({"Date": date, "type": "TRAIL_FLOOR", "value": total_value})
            elif tqqq_close <= tqqq_trail_peak * (1.0 + cur_trail):
                def_shares  = total_value / qqq_close
                tqqq_shares = 0.0
                state       = "NORMAL"
                touched_10 = touched_20 = False
                events.append({"Date": date, "type": "TRAIL_EXIT", "value": total_value})

        # ── HALF / FULL ATTACK ──
        elif state in ("HALF_ATTACK", "FULL_ATTACK"):
            if dd >= 0:
                tqqq_trail_peak  = tqqq_close
                tqqq_trail_entry = tqqq_close
                # macro: trail 시점 multiplier 적용
                cur_trail = float(np.clip(sub["trail"] * trail_mult,
                                           -0.30, -0.03))
                state = "TRAILING"
                events.append({"Date": date, "type": "TO_TRAILING", "value": total_value})
            elif state == "HALF_ATTACK" and dd <= hs_b:
                def_shares  = total_value / qqq_close
                tqqq_shares = 0.0
                state       = "NORMAL"
                events.append({"Date": date, "type": "HALF_STOP", "value": total_value})
            elif state == "HALF_ATTACK" and touched_20 and dd >= db_b:
                tqqq_shares = total_value * sub["ff"] / tqqq_close
                def_shares  = total_value * (1.0 - sub["ff"]) / qqq_close
                state       = "FULL_ATTACK"
                events.append({"Date": date, "type": "TO_FULL_ATTACK", "value": total_value})

        # ── NORMAL ──
        elif state == "NORMAL":
            if dd >= 0:
                touched_10 = touched_20 = False

            # ★ VIX CATASTROPHIC GUARD: 진입 차단 ★
            elif vix_guard == 1:
                pass   # 진입 시그널 무시

            elif touched_10 and not touched_20 and dd >= sub["b"]:
                cur_hf = float(np.clip(sub["hf"] * hf_mult, 0.10, 1.0))
                tqqq_shares = total_value * cur_hf / tqqq_close
                def_shares  = total_value * (1.0 - cur_hf) / qqq_close
                state       = "HALF_ATTACK"
                events.append({"Date": date, "type": "TO_HALF_ATTACK",
                               "value": total_value, "hf_used": cur_hf})
            elif touched_20 and dd >= db_b:
                cur_ff = float(np.clip(sub["ff"] * hf_mult, 0.20, 1.0))
                tqqq_shares = total_value * cur_ff / tqqq_close
                def_shares  = total_value * (1.0 - cur_ff) / qqq_close
                state       = "FULL_ATTACK"
                events.append({"Date": date, "type": "TO_FULL_ATTACK",
                               "value": total_value, "ff_used": cur_ff})

        portfolio_values.append(def_shares * qqq_close + tqqq_shares * tqqq_close)

    return pd.Series(portfolio_values, index=dates), pd.DataFrame(events) if events else pd.DataFrame(columns=["Date", "type", "value"])


def run_s9_master(qqq, qld, tqqq, macro_df, initial_capital=100_000):
    """S6 ensemble + macro overlay = S9."""
    n_sub = len(SUB_ANCHORS)
    sub_cap = initial_capital / n_sub
    sub_ports = []
    sub_events = {}
    for sub in SUB_ANCHORS:
        port, ev = strategy_s4_macro_aware(qqq, qld, tqqq, macro_df, sub_cap, sub)
        sub_ports.append(port.rename(sub["name"]))
        sub_events[sub["name"]] = ev
    df = pd.concat(sub_ports, axis=1)

    # Annual rebalance (1월 1일)
    sub_rets = df.pct_change().fillna(0).values
    n_days, n = sub_rets.shape
    rebal_dates = pd.date_range(df.index[0], df.index[-1], freq="AS")
    rebal_set = set()
    for d in rebal_dates:
        loc = df.index.searchsorted(d)
        if loc < n_days:
            rebal_set.add(loc)
    sub_caps = np.full(n, initial_capital / n)
    portfolio = np.empty(n_days)
    portfolio[0] = initial_capital
    for t in range(1, n_days):
        sub_caps = sub_caps * (1 + sub_rets[t])
        cap = sub_caps.sum()
        if t in rebal_set:
            sub_caps = np.full(n, cap / n)
        portfolio[t] = cap
    return pd.Series(portfolio, index=df.index, name="S9_Master"), df, sub_events


# ── 데이터 ───────────────────────────────────────────────────────────────────
print("=" * 78)
print("[STAGE 5] S9 Master = S6 Ensemble (Posterior) + VIX Guard + Yield Tilt")
print("=" * 78)

qqq  = load_extended_daily("QQQ")
qld  = load_extended_daily("QLD")
tqqq = load_extended_daily("TQQQ")
common = qqq.index.intersection(qld.index).intersection(tqqq.index)
qqq, qld, tqqq = qqq.loc[common], qld.loc[common], tqqq.loc[common]

vix, yld_spread = load_macro_signals()
macro_df = compute_macro_overlay(vix, yld_spread, lookback=252)
print(f"  Macro days: {len(macro_df):,}")
print(f"  VIX guard active: {macro_df['vix_guard'].sum():,}일 ({macro_df['vix_guard'].mean()*100:.1f}%)")
print(f"  Yield inverted: {(macro_df['yld_spread']<0).sum():,}일 "
      f"({(macro_df['yld_spread']<0).mean()*100:.1f}%)")

CV_START  = pd.Timestamp("2002-01-01")
OOS_START = pd.Timestamp("2017-01-01")
END = qqq.index[-1]

def slice_data(start, end):
    m = (qqq.index >= start) & (qqq.index <= end)
    return qqq[m], qld[m], tqqq[m]

Q_cv, L_cv, T_cv    = slice_data(CV_START, OOS_START - pd.Timedelta(days=1))
Q_oos, L_oos, T_oos = slice_data(OOS_START, END)
Q_full, L_full, T_full = slice_data(CV_START, END)

# ── 백테스트 ────────────────────────────────────────────────────────────────
print("\n  S9 백테스트 실행...")
t0 = time.time()
s9_cv,   subs_cv,   ev_cv   = run_s9_master(Q_cv,   L_cv,   T_cv,   macro_df)
s9_oos,  subs_oos,  ev_oos  = run_s9_master(Q_oos,  L_oos,  T_oos,  macro_df)
s9_full, subs_full, ev_full = run_s9_master(Q_full, L_full, T_full, macro_df)
print(f"  완료 ({time.time()-t0:.1f}s)")

# 비교 (S1, S4 anchor, Posterior, S6 ensemble - 모두 macro 없음)
from backtest_switching import strategy_s4_trailing
def run_s4_simple(Q, L, T, p):
    sd, db = p["b"]*p.get("r1",2), p["b"]*p["r2"]
    dd, hs = p["b"]*p.get("r1",2)*p["r2"], (sd + p["b"]*p.get("r1",2)*p["r2"])/2
    port, _ = strategy_s4_trailing(Q, L, T, 100_000,
        shallow_drop=sd, deep_drop=dd, shallow_bounce=p["b"], deep_bounce=db,
        half_stop=hs, trailing_stop_pct=p["trail"],
        half_frac=p["hf"], full_frac=p.get("ff",1))
    return port

s1_cv, s1_oos, s1_full = (
    strategy_buy_and_hold(Q_cv, 100_000),
    strategy_buy_and_hold(Q_oos, 100_000),
    strategy_buy_and_hold(Q_full, 100_000),
)

ANCHOR  = {"b":-0.05,"r1":2,"r2":2,"trail":-0.15,"hf":0.50,"ff":1.0}
s4a_cv  = run_s4_simple(Q_cv,  L_cv,  T_cv,  ANCHOR)
s4a_oos = run_s4_simple(Q_oos, L_oos, T_oos, ANCHOR)
s4a_full= run_s4_simple(Q_full,L_full,T_full,ANCHOR)

POST = {**POSTERIOR_BASE}
post_cv  = run_s4_simple(Q_cv,  L_cv,  T_cv,  POST)
post_oos = run_s4_simple(Q_oos, L_oos, T_oos, POST)
post_full= run_s4_simple(Q_full,L_full,T_full,POST)

# ── 결과 출력 ────────────────────────────────────────────────────────────────
def show_metrics(label, p_cv, p_oos, p_full):
    m_cv, m_oos, m_full = full_metrics(p_cv), full_metrics(p_oos), full_metrics(p_full)
    print(f"  {label:24s}  CV: CAGR={m_cv['cagr']*100:>+5.1f}% Sh={m_cv['sharpe']:.2f} MDD={m_cv['mdd']*100:>+6.1f}% Ulcer={m_cv['ulcer']:>5.2f}  "
          f"|  OOS: CAGR={m_oos['cagr']*100:>+5.1f}% Sh={m_oos['sharpe']:.2f} MDD={m_oos['mdd']*100:>+6.1f}% Ulcer={m_oos['ulcer']:>5.2f}  "
          f"|  Full: CAGR={m_full['cagr']*100:>+5.1f}% Sh={m_full['sharpe']:.2f}")
    return m_cv, m_oos, m_full

print("\n" + "─" * 168)
m_s1   = show_metrics("S1 (QQQ B&H)",       s1_cv,  s1_oos,  s1_full)
m_s4a  = show_metrics("S4 Anchor (사용자)", s4a_cv, s4a_oos, s4a_full)
m_post = show_metrics("S4 Posterior",       post_cv, post_oos, post_full)
m_s9   = show_metrics("S9 Master ★",        s9_cv,  s9_oos,  s9_full)

# 진입 이벤트 통계
print(f"\n  S9 OOS 진입 이벤트: {sum(len(e) for e in ev_oos.values()):,}건 (5 sub 합산)")
total_entries_macro = 0
for name, ev in ev_oos.items():
    if not ev.empty:
        n_entries = ev[ev["type"].isin(["TO_HALF_ATTACK","TO_FULL_ATTACK"])].shape[0]
        total_entries_macro += n_entries

# Bootstrap
print("\n" + "─" * 168)
s1_rets   = s1_oos.pct_change().dropna()
s4a_rets  = s4a_oos.pct_change().dropna()
post_rets = post_oos.pct_change().dropna()
s9_rets   = s9_oos.pct_change().dropna()

bs_s4a = paired_bootstrap_compare(s4a_rets, s1_rets, 60, 500)
bs_post = paired_bootstrap_compare(post_rets, s1_rets, 60, 500)
bs_s9   = paired_bootstrap_compare(s9_rets,   s1_rets, 60, 500)
bs_s9_vs_post = paired_bootstrap_compare(s9_rets, post_rets, 60, 500)

print("  [Paired Bootstrap, OOS, 60-day blocks × 500]")
def show_bs(label, bs):
    def cell(key, fmt="{:+.3f}"):
        x = bs[key]; sig = "★" if x["p_value"]<0.05 else ("·" if x["p_value"]<0.20 else " ")
        return f"{fmt.format(x['median'])}{sig} ({x['prob_better']*100:>3.0f}%)"
    print(f"  {label:24s}  ΔSharpe={cell('delta_sharpe'):>20s}  "
          f"ΔCAGR={cell('delta_cagr','{:+.1%}'):>22s}  "
          f"ΔSortino={cell('delta_sortino'):>20s}  "
          f"ΔUlcer={cell('delta_ulcer'):>20s}")

show_bs("S4 Anchor vs S1",  bs_s4a)
show_bs("S4 Posterior vs S1", bs_post)
show_bs("S9 Master vs S1",   bs_s9)
show_bs("S9 Master vs Post", bs_s9_vs_post)

# ── 시각화 ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 14))
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.28)
fig.suptitle("S9 Master: S6 Ensemble (Posterior anchor) + VIX Guard + Yield Curve Tilt",
             fontsize=12, fontweight="bold")

# ① OOS 자산곡선
ax = fig.add_subplot(gs[0, :])
for s, lbl, c, lw in [
    (s1_oos, "S1 QQQ", "#000", 1.3),
    (s4a_oos, "S4 Anchor (b=-5%)", "#9CA3AF", 1.3),
    (post_oos, "S4 Posterior", "#DC2626", 1.5),
    (s9_oos, "S9 Master ★", "#2563EB", 2.0),
]:
    ax.plot(s.index, s.values / s.iloc[0] * 100, label=lbl, color=c, lw=lw,
            alpha=0.9 if "S9" in lbl else 0.75)
ax.set_yscale("log"); ax.set_ylabel("Index (start=100, log)")
ax.set_title(f"OOS 자산곡선  ({s9_oos.index[0].date()} ~ {s9_oos.index[-1].date()})",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9, loc="upper left"); ax.grid(True, alpha=0.3)

# ② 누적 drawdown
ax = fig.add_subplot(gs[1, :])
for s, lbl, c in [
    (s1_oos, "S1 QQQ", "#000"),
    (s4a_oos, "S4 Anchor", "#9CA3AF"),
    (post_oos, "S4 Posterior", "#DC2626"),
    (s9_oos, "S9 Master", "#2563EB"),
]:
    dd = (s - s.cummax()) / s.cummax() * 100
    ax.fill_between(dd.index, dd, 0, alpha=0.25, color=c, label=lbl)
ax.set_ylabel("Drawdown (%)")
ax.set_title("OOS Drawdown 누적", fontsize=10, fontweight="bold")
ax.legend(fontsize=9, loc="lower left"); ax.grid(True, alpha=0.3)

# ③ Bootstrap ΔSharpe
ax = fig.add_subplot(gs[2, 0])
for label, bs, color in [
    ("Anchor", bs_s4a, "#9CA3AF"),
    ("Posterior", bs_post, "#DC2626"),
    ("S9 Master", bs_s9, "#2563EB"),
]:
    ax.hist(np.array(bs["raw"]["delta_sharpe"]), bins=30, alpha=0.4,
            label=label, color=color, density=True, edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Δ Sharpe (vs S1)"); ax.set_ylabel("density")
ax.set_title("Bootstrap ΔSharpe — S1 대비", fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ④ Bootstrap ΔCAGR
ax = fig.add_subplot(gs[2, 1])
for label, bs, color in [
    ("Anchor", bs_s4a, "#9CA3AF"),
    ("Posterior", bs_post, "#DC2626"),
    ("S9 Master", bs_s9, "#2563EB"),
]:
    ax.hist(np.array(bs["raw"]["delta_cagr"])*100, bins=30, alpha=0.4,
            label=label, color=color, density=True, edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Δ CAGR (vs S1, %)"); ax.set_ylabel("density")
ax.set_title("Bootstrap ΔCAGR — S1 대비", fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ⑤ S9 vs Posterior (이게 진짜 macro overlay 가치)
ax = fig.add_subplot(gs[3, 0])
ax.hist(np.array(bs_s9_vs_post["raw"]["delta_sharpe"]), bins=30, alpha=0.65,
        color="#7C3AED", density=True, edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
prob = bs_s9_vs_post["delta_sharpe"]["prob_better"]
median = bs_s9_vs_post["delta_sharpe"]["median"]
ax.set_xlabel("Δ Sharpe (S9 − Posterior)"); ax.set_ylabel("density")
ax.set_title(f"Macro overlay 가치  ΔSh median={median:+.3f}  P(S9>Post)={prob*100:.0f}%",
             fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3)

# ⑥ Macro signal coverage (OOS)
ax = fig.add_subplot(gs[3, 1])
m_oos = macro_df.loc[macro_df.index >= OOS_START]
ax.plot(m_oos.index, m_oos["vix"], color="#374151", lw=0.7, alpha=0.7, label="VIX")
guard = m_oos["vix_guard"].astype(bool)
ax.fill_between(m_oos.index, 0, 80, where=guard, color="red",
                alpha=0.25, label="VIX guard active")
ax2 = ax.twinx()
ax2.plot(m_oos.index, m_oos["yld_spread"], color="#10B981", lw=0.8,
         alpha=0.7, label="10Y-3M spread")
ax2.axhline(0, color="#EF4444", ls="--", lw=0.8, alpha=0.7)
ax.set_ylabel("VIX", color="#374151"); ax2.set_ylabel("Yield spread (%)", color="#10B981")
ax.set_title("Macro signals (OOS)", fontsize=10, fontweight="bold")
ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

plt.savefig(OUT_DIR / "stage5_s9_master.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── JSON 저장 ────────────────────────────────────────────────────────────────
out = {
    "design": "S9 = 5 sub (Posterior anchor) + VIX 99-pct guard + Yield curve tilt",
    "sub_anchors": SUB_ANCHORS,
    "macro_overlay": {
        "vix_guard_pct": 0.99,
        "yield_inverted_hf_mult": 0.5,
        "yield_inverted_trail_mult": 1.5,
    },
    "s1":   {"cv": m_s1[0],   "oos": m_s1[1],   "full": m_s1[2]},
    "s4_anchor":   {"cv": m_s4a[0],  "oos": m_s4a[1],  "full": m_s4a[2]},
    "s4_posterior":{"cv": m_post[0], "oos": m_post[1], "full": m_post[2]},
    "s9_master":   {"cv": m_s9[0],   "oos": m_s9[1],   "full": m_s9[2]},
    "bootstrap_oos": {
        "s4_anchor_vs_s1":     {k: v for k, v in bs_s4a.items() if k != "raw"},
        "s4_posterior_vs_s1":  {k: v for k, v in bs_post.items() if k != "raw"},
        "s9_master_vs_s1":     {k: v for k, v in bs_s9.items() if k != "raw"},
        "s9_master_vs_post":   {k: v for k, v in bs_s9_vs_post.items() if k != "raw"},
    },
}

def _to_json(o):
    if isinstance(o, dict): return {k: _to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_to_json(x) for x in o]
    if isinstance(o, (np.floating, np.float64, np.float32)): return float(o)
    if isinstance(o, (np.integer, np.int64)): return int(o)
    return o

with open(OUT_DIR / "stage5_s9_master.json", "w") as f:
    json.dump(_to_json(out), f, indent=2)

print(f"\n  PNG → {OUT_DIR / 'stage5_s9_master.png'}")
print(f"  JSON → {OUT_DIR / 'stage5_s9_master.json'}")
print("\n[STAGE 5 완료]")
