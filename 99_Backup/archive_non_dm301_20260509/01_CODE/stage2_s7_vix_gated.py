"""
stage2_s7_vix_gated.py
======================
[단계 2] S7: VIX-Gated S4 — vol regime에 따라 진입 비중 조절.

전략 설계
--------
S4 anchor (b=-5%, r1=2, r2=2, trail=-15%)를 기본으로 사용.
진입 시 VIX의 252일 percentile에 따라 half_frac 동적 조절:

    if VIX_now > VIX.rolling(252).quantile(0.70):
        → half_frac = 0.25  (high vol = uncertainty, 보수적 진입)
        → trail     = -7%   (빠른 청산, 손실 제한)
    elif VIX_now < VIX.rolling(252).quantile(0.30):
        → half_frac = 1.00  (low vol = clean trend, 공격적)
        → trail     = -15%  (큰 추세 추적)
    else:
        → half_frac = 0.50  (anchor)
        → trail     = -15%  (anchor)

근거
----
• VIX = 30일 implied vol. "high VIX → high realized vol → 잡음 ↑ → trade 보수적"
• "low VIX → 추세장에서의 mean reversion 강력 → 공격적 진입"
• VIX는 QLD drawdown과 직교 — 새로운 정보원
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

from backtest_switching import (
    load_extended_daily, strategy_buy_and_hold, strategy_s4_trailing,
)
from evaluation_metrics import full_metrics, paired_bootstrap_compare

OUT_DIR = Path("03_RESULT/sensitivity")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_vix() -> pd.DataFrame:
    df = pd.read_csv("02_DATA/yahoo_extended/VIX/VIX_daily.csv",
                     index_col="Date", parse_dates=True)
    df.index.name = "Date"
    return df


def compute_vix_signal(vix: pd.Series, lookback: int = 252,
                       lo_q: float = 0.30, hi_q: float = 0.70) -> pd.DataFrame:
    """VIX rolling percentile → regime label."""
    rank = vix.rolling(lookback, min_periods=lookback//4).rank(pct=True)
    regime = pd.Series("MID", index=vix.index)
    regime[rank > hi_q] = "HIGH"
    regime[rank < lo_q] = "LOW"
    return pd.DataFrame({"vix": vix, "rank": rank, "regime": regime})


def strategy_s7_vix_gated(
    qqq: pd.DataFrame, qld: pd.DataFrame, tqqq: pd.DataFrame,
    vix_df: pd.DataFrame,
    initial_capital: float = 100_000,
    # base S4 params
    b: float = -0.05, r1: float = 2.0, r2: float = 2.0,
    # vol-regime overrides
    hf_low: float = 1.00, hf_mid: float = 0.50, hf_high: float = 0.25,
    trail_low: float = -0.15, trail_mid: float = -0.15, trail_high: float = -0.07,
    full_frac: float = 1.00,
):
    """Vol-regime conditional S4 trailing.

    매 진입 결정 시점의 VIX regime에 따라 half_frac, trail 동적 선택.
    HALF_ATTACK 진입 후의 trail은 "TRAILING 상태로 들어갈 때" regime 사용.
    """
    dates = qqq.index
    portfolio_values = []
    switch_events = []

    def_shares  = initial_capital / qqq["Close"].iloc[0]
    tqqq_shares = 0.0
    state       = "NORMAL"
    ath         = qld["Close"].iloc[0]
    touched_10  = False
    touched_20  = False

    tqqq_trail_peak  = 0.0
    tqqq_trail_entry = 0.0
    cur_trail        = trail_mid   # active trailing 사용 값

    sd_base = b * r1
    db_base = b * r2
    dd_base = b * r1 * r2
    hs_base = (sd_base + dd_base) / 2

    for i, date in enumerate(dates):
        qld_close  = qld["Close"].iloc[i]
        qqq_close  = qqq["Close"].iloc[i]
        tqqq_close = tqqq["Close"].iloc[i]

        # VIX regime 조회 (forward fill)
        vix_loc = vix_df.index.searchsorted(date, "right") - 1
        if vix_loc < 0:
            regime = "MID"
        else:
            regime = vix_df["regime"].iloc[vix_loc]

        if regime == "LOW":
            cur_hf, cur_tr = hf_low,  trail_low
        elif regime == "HIGH":
            cur_hf, cur_tr = hf_high, trail_high
        else:
            cur_hf, cur_tr = hf_mid,  trail_mid

        if qld_close > ath:
            ath = qld_close
        dd = (qld_close - ath) / ath

        if dd <= b * r1:
            touched_10 = True
        if dd <= b * r1 * r2:
            touched_20 = True

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
                switch_events.append({"Date": date, "type": "TRAIL_FLOOR", "value": total_value, "regime": regime})
            elif tqqq_close <= tqqq_trail_peak * (1.0 + cur_trail):
                def_shares  = total_value / qqq_close
                tqqq_shares = 0.0
                state       = "NORMAL"
                touched_10 = touched_20 = False
                switch_events.append({"Date": date, "type": "TRAIL_EXIT", "value": total_value, "regime": regime})

        # ── HALF / FULL ATTACK ──
        elif state in ("HALF_ATTACK", "FULL_ATTACK"):
            if dd >= 0:
                tqqq_trail_peak  = tqqq_close
                tqqq_trail_entry = tqqq_close
                cur_trail        = cur_tr   # TRAILING 진입 시점의 regime 기준 trail 고정
                state = "TRAILING"
                switch_events.append({"Date": date, "type": "TO_TRAILING", "value": total_value, "regime": regime})
            elif state == "HALF_ATTACK" and dd <= hs_base:
                def_shares  = total_value / qqq_close
                tqqq_shares = 0.0
                state       = "NORMAL"
                switch_events.append({"Date": date, "type": "HALF_STOP", "value": total_value, "regime": regime})
            elif state == "HALF_ATTACK" and touched_20 and dd >= db_base:
                tqqq_shares = total_value * full_frac / tqqq_close
                def_shares  = total_value * (1.0 - full_frac) / qqq_close
                state       = "FULL_ATTACK"
                switch_events.append({"Date": date, "type": "TO_FULL_ATTACK", "value": total_value, "regime": regime})

        # ── NORMAL ──
        elif state == "NORMAL":
            if dd >= 0:
                touched_10 = touched_20 = False
            elif touched_10 and not touched_20 and dd >= b:
                # vol regime별 hf 사용
                tqqq_shares = total_value * cur_hf / tqqq_close
                def_shares  = total_value * (1.0 - cur_hf) / qqq_close
                state       = "HALF_ATTACK"
                switch_events.append({"Date": date, "type": "TO_HALF_ATTACK", "value": total_value, "regime": regime})
            elif touched_20 and dd >= db_base:
                tqqq_shares = total_value * full_frac / tqqq_close
                def_shares  = total_value * (1.0 - full_frac) / qqq_close
                state       = "FULL_ATTACK"
                switch_events.append({"Date": date, "type": "TO_FULL_ATTACK", "value": total_value, "regime": regime})

        portfolio_values.append(def_shares * qqq_close + tqqq_shares * tqqq_close)

    portfolio = pd.Series(portfolio_values, index=dates, name="S7")
    events_df = pd.DataFrame(switch_events) if switch_events else pd.DataFrame(
        columns=["Date", "type", "value", "regime"])
    return portfolio, events_df


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print("=" * 78)
print("[STAGE 2] S7: VIX-Gated S4")
print("=" * 78)

qqq  = load_extended_daily("QQQ")
qld  = load_extended_daily("QLD")
tqqq = load_extended_daily("TQQQ")
common = qqq.index.intersection(qld.index).intersection(tqqq.index)
qqq, qld, tqqq = qqq.loc[common], qld.loc[common], tqqq.loc[common]

vix_raw = load_vix()
print(f"  QQQ/QLD/TQQQ: {qqq.index[0].date()} ~ {qqq.index[-1].date()}")
print(f"  VIX        : {vix_raw.index[0].date()} ~ {vix_raw.index[-1].date()}")

vix_df = compute_vix_signal(vix_raw["Close"], lookback=252, lo_q=0.30, hi_q=0.70)

# regime 분포
print(f"\n  VIX regime 분포 (252일 rolling percentile):")
for r in ["LOW", "MID", "HIGH"]:
    n = (vix_df["regime"] == r).sum()
    print(f"    {r:5s}: {n:>5,}일 ({n/len(vix_df)*100:>5.1f}%)")

CV_START = pd.Timestamp("2002-01-01")
OOS_START = pd.Timestamp("2017-01-01")
END = qqq.index[-1]

def slice_data(start, end):
    m = (qqq.index >= start) & (qqq.index <= end)
    return qqq[m], qld[m], tqqq[m]

Q_cv, L_cv, T_cv    = slice_data(CV_START, OOS_START - pd.Timedelta(days=1))
Q_oos, L_oos, T_oos = slice_data(OOS_START, END)
Q_full, L_full, T_full = slice_data(CV_START, END)

# ── 백테스트 ────────────────────────────────────────────────────────────────
print("\n  백테스트 실행...")
t0 = time.time()
s7_cv,   ev_cv   = strategy_s7_vix_gated(Q_cv,   L_cv,   T_cv,   vix_df)
s7_oos,  ev_oos  = strategy_s7_vix_gated(Q_oos,  L_oos,  T_oos,  vix_df)
s7_full, ev_full = strategy_s7_vix_gated(Q_full, L_full, T_full, vix_df)
print(f"  완료 ({time.time()-t0:.1f}s)")
print(f"  이벤트 수: CV={len(ev_cv)} | OOS={len(ev_oos)} | Full={len(ev_full)}")

# 비교용
s1_cv   = strategy_buy_and_hold(Q_cv,   100_000)
s1_oos  = strategy_buy_and_hold(Q_oos,  100_000)
s1_full = strategy_buy_and_hold(Q_full, 100_000)

# S4 anchor
s4_cv,  _ = strategy_s4_trailing(Q_cv,  L_cv,  T_cv,  100_000,
    shallow_drop=-0.10, deep_drop=-0.20, shallow_bounce=-0.05, deep_bounce=-0.10,
    half_stop=-0.15, trailing_stop_pct=-0.15, half_frac=0.50, full_frac=1.00)
s4_oos, _ = strategy_s4_trailing(Q_oos, L_oos, T_oos, 100_000,
    shallow_drop=-0.10, deep_drop=-0.20, shallow_bounce=-0.05, deep_bounce=-0.10,
    half_stop=-0.15, trailing_stop_pct=-0.15, half_frac=0.50, full_frac=1.00)
s4_full,_ = strategy_s4_trailing(Q_full, L_full, T_full, 100_000,
    shallow_drop=-0.10, deep_drop=-0.20, shallow_bounce=-0.05, deep_bounce=-0.10,
    half_stop=-0.15, trailing_stop_pct=-0.15, half_frac=0.50, full_frac=1.00)

# ── 메트릭 ───────────────────────────────────────────────────────────────────
def show_metrics(label, p_cv, p_oos, p_full):
    m_cv, m_oos, m_full = full_metrics(p_cv), full_metrics(p_oos), full_metrics(p_full)
    print(f"  {label:24s}  CV: CAGR={m_cv['cagr']*100:>+5.1f}% Sh={m_cv['sharpe']:.2f} MDD={m_cv['mdd']*100:>+6.1f}% Ulcer={m_cv['ulcer']:>5.2f}  "
          f"|  OOS: CAGR={m_oos['cagr']*100:>+5.1f}% Sh={m_oos['sharpe']:.2f} MDD={m_oos['mdd']*100:>+6.1f}% Ulcer={m_oos['ulcer']:>5.2f}  "
          f"|  Full: CAGR={m_full['cagr']*100:>+5.1f}% Sh={m_full['sharpe']:.2f}")
    return m_cv, m_oos, m_full

print("\n" + "─" * 168)
m_s1 = show_metrics("S1 (QQQ B&H)",  s1_cv,  s1_oos,  s1_full)
m_s4 = show_metrics("S4 Anchor",     s4_cv,  s4_oos,  s4_full)
m_s7 = show_metrics("S7 VIX-Gated",  s7_cv,  s7_oos,  s7_full)

# 진입 이벤트 regime 분포
print("\n  S7 OOS 진입 이벤트의 regime 분포 (TO_HALF_ATTACK 기준):")
ev_half = ev_oos[ev_oos["type"] == "TO_HALF_ATTACK"]
if not ev_half.empty:
    for r in ["LOW", "MID", "HIGH"]:
        n = (ev_half["regime"] == r).sum()
        print(f"    {r:5s}: {n}건")

# ── Bootstrap ────────────────────────────────────────────────────────────────
print("\n" + "─" * 168)
s1_rets_oos = s1_oos.pct_change().dropna()
s4_rets_oos = s4_oos.pct_change().dropna()
s7_rets_oos = s7_oos.pct_change().dropna()

bs_s4 = paired_bootstrap_compare(s4_rets_oos, s1_rets_oos, 60, 500)
bs_s7 = paired_bootstrap_compare(s7_rets_oos, s1_rets_oos, 60, 500)
bs_s7_vs_s4 = paired_bootstrap_compare(s7_rets_oos, s4_rets_oos, 60, 500)

def show_bs(label, bs):
    def cell(key, fmt="{:+.3f}"):
        x = bs[key]; sig = "★" if x["p_value"]<0.05 else ("·" if x["p_value"]<0.20 else " ")
        return f"{fmt.format(x['median'])}{sig} ({x['prob_better']*100:>3.0f}%)"
    print(f"  {label:24s}  ΔSharpe={cell('delta_sharpe'):>20s}  "
          f"ΔCAGR={cell('delta_cagr','{:+.1%}'):>22s}  "
          f"ΔSortino={cell('delta_sortino'):>20s}  "
          f"ΔUlcer={cell('delta_ulcer'):>20s}")

print("  [Paired Bootstrap, OOS, 60-day blocks × 500]")
show_bs("S4 Anchor vs S1",   bs_s4)
show_bs("S7 VIX-Gate vs S1", bs_s7)
show_bs("S7 VIX-Gate vs S4", bs_s7_vs_s4)

# ── 시각화 ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 14))
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.28)
fig.suptitle("S7: VIX-Gated S4  (vol regime conditional half_frac & trail)",
             fontsize=12, fontweight="bold")

# ① OOS 자산곡선
ax = fig.add_subplot(gs[0, :])
ax.plot(s1_oos.index,  s1_oos.values  / s1_oos.iloc[0]  * 100, label="S1 QQQ", color="#000", lw=1.4, alpha=0.7)
ax.plot(s4_oos.index,  s4_oos.values  / s4_oos.iloc[0]  * 100, label="S4 Anchor",   color="#DC2626", lw=1.6)
ax.plot(s7_oos.index,  s7_oos.values  / s7_oos.iloc[0]  * 100, label="S7 VIX-Gate", color="#2563EB", lw=2.0)
ax.set_yscale("log")
ax.set_ylabel("Index (start=100, log)")
ax.set_title(f"OOS 자산곡선  ({s7_oos.index[0].date()} ~ {s7_oos.index[-1].date()})",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ② VIX with regime band (OOS)
ax = fig.add_subplot(gs[1, :])
vix_oos_mask = (vix_df.index >= OOS_START) & (vix_df.index <= END)
v = vix_df.loc[vix_oos_mask]
ax.plot(v.index, v["vix"], color="#374151", lw=0.8, alpha=0.8)
# regime shading
for r, color in [("LOW", "#10B981"), ("HIGH", "#EF4444")]:
    mask = (v["regime"] == r).values
    if mask.any():
        ax.fill_between(v.index, 0, v["vix"], where=mask,
                        color=color, alpha=0.15, label=f"VIX {r}")
ax.set_ylabel("VIX")
ax.set_title("VIX with regime shading (LOW=green, HIGH=red, MID=white)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

# ③ Bootstrap ΔSharpe
ax = fig.add_subplot(gs[2, 0])
ax.hist(np.array(bs_s4["raw"]["delta_sharpe"]), bins=30, alpha=0.45,
        label="S4 vs S1", color="#DC2626", density=True, edgecolor="white")
ax.hist(np.array(bs_s7["raw"]["delta_sharpe"]), bins=30, alpha=0.45,
        label="S7 vs S1", color="#2563EB", density=True, edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Δ Sharpe (vs S1)"); ax.set_ylabel("density")
ax.set_title("Bootstrap ΔSharpe vs S1", fontsize=9, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ④ Bootstrap ΔCAGR
ax = fig.add_subplot(gs[2, 1])
ax.hist(np.array(bs_s4["raw"]["delta_cagr"])*100, bins=30, alpha=0.45,
        label="S4 vs S1", color="#DC2626", density=True, edgecolor="white")
ax.hist(np.array(bs_s7["raw"]["delta_cagr"])*100, bins=30, alpha=0.45,
        label="S7 vs S1", color="#2563EB", density=True, edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Δ CAGR (vs S1, %)"); ax.set_ylabel("density")
ax.set_title("Bootstrap ΔCAGR vs S1", fontsize=9, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ⑤ S7 vs S4: 직접 비교 분포
ax = fig.add_subplot(gs[3, 0])
ax.hist(np.array(bs_s7_vs_s4["raw"]["delta_sharpe"]), bins=30, alpha=0.65,
        color="#7C3AED", density=True, edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Δ Sharpe (S7 − S4)"); ax.set_ylabel("density")
prob = bs_s7_vs_s4["delta_sharpe"]["prob_better"]
ax.set_title(f"Bootstrap S7 vs S4 ΔSharpe  (P(S7>S4)={prob*100:.0f}%)",
             fontsize=9, fontweight="bold")
ax.grid(True, alpha=0.3)

# ⑥ S7 진입 이벤트 막대
ax = fig.add_subplot(gs[3, 1])
ev_oos_full = ev_oos[ev_oos["type"].isin(["TO_HALF_ATTACK", "TO_FULL_ATTACK"])]
if not ev_oos_full.empty:
    counts = ev_oos_full.groupby("regime").size()
    counts = counts.reindex(["LOW", "MID", "HIGH"], fill_value=0)
    colors = ["#10B981", "#9CA3AF", "#EF4444"]
    ax.bar(counts.index, counts.values, color=colors, edgecolor="black", lw=0.5)
    for i, (r, c) in enumerate(counts.items()):
        ax.text(i, c+0.1, str(c), ha="center", fontsize=10)
    ax.set_ylabel("진입 횟수")
    ax.set_title("S7 OOS 진입 이벤트의 VIX regime 분포", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

plt.savefig(OUT_DIR / "stage2_s7_vix_gated.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── JSON 저장 ────────────────────────────────────────────────────────────────
out = {
    "vix_regime_thresholds": {"low_q": 0.30, "high_q": 0.70, "lookback": 252},
    "regime_overrides": {
        "LOW":  {"hf": 1.00, "trail": -0.15},
        "MID":  {"hf": 0.50, "trail": -0.15},
        "HIGH": {"hf": 0.25, "trail": -0.07},
    },
    "regime_distribution": {
        r: int((vix_df["regime"] == r).sum()) for r in ["LOW", "MID", "HIGH"]
    },
    "s1": {"cv": m_s1[0], "oos": m_s1[1], "full": m_s1[2]},
    "s4_anchor": {"cv": m_s4[0], "oos": m_s4[1], "full": m_s4[2]},
    "s7_vix_gated": {"cv": m_s7[0], "oos": m_s7[1], "full": m_s7[2]},
    "n_events_oos": len(ev_oos),
    "bootstrap_s7_vs_s1": {k: v for k, v in bs_s7.items() if k != "raw"},
    "bootstrap_s4_vs_s1": {k: v for k, v in bs_s4.items() if k != "raw"},
    "bootstrap_s7_vs_s4": {k: v for k, v in bs_s7_vs_s4.items() if k != "raw"},
}

def _to_json(o):
    if isinstance(o, dict): return {k: _to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_to_json(x) for x in o]
    if isinstance(o, (np.floating, np.float64, np.float32)): return float(o)
    if isinstance(o, (np.integer, np.int64)): return int(o)
    return o

with open(OUT_DIR / "stage2_s7_vix_gated.json", "w") as f:
    json.dump(_to_json(out), f, indent=2)

print(f"\n  PNG → {OUT_DIR / 'stage2_s7_vix_gated.png'}")
print(f"  JSON → {OUT_DIR / 'stage2_s7_vix_gated.json'}")
print("\n[STAGE 2 완료]")
