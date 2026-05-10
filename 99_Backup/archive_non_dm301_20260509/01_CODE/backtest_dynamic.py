"""
backtest_dynamic.py
===================
동적 A 기반 전략 (Dynamic-A Strategy)

정의: A = 직전 ATH 대비 로컬 저점 낙폭 (% 단위, 예: -20%이면 A=20)

진입:
  A > min_A 이고 QLD가 ATH 대비 낙폭 A의 k_entry 배만큼 회복 (기본 k_entry=0.5 → 중간값)
  즉 threshold = ath × (1 − k_entry × A/100)

투입량:
  min(A / invest_divisor, 1.0) 비율만큼 TQQQ (기본 invest_divisor=40)

손절:
  진입 후 QLD가 ATH 대비 낙폭 k_stop × A % 이하로 재하락 → 전량 청산 (기본 k_stop=0.75 = 3A/4)
  stop_qld = ath × (1 − k_stop × A/100)

청산:
  QLD가 ATH 회복 → TRAILING 진입
  TRAILING: TQQQ 진입가 이하 하락(TRAIL_FLOOR) 또는
            TQQQ 고점 대비 -10% 하락(TRAIL_EXIT) → NORMAL

A/40 공식 예시:
  A=10 → 25%   (경계값, A>10 미충족 → 미진입)
  A=15 → 37.5%
  A=20 → 50%
  A=30 → 75%
  A=40 → 100%  (이상부터 모두 100%)
  A=50 → 100%  (cap)

* 사용자 예시 "-20% 터치→ 1/4 투입"은 A/80에 해당하나,
  "-50% 터치→ 100% 투입" 조건을 동시에 만족하는 공식은 min(A/40, 1)임.
  본 코드는 공식 그대로 A/40 사용.
"""

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from backtest_switching import load_extended_daily, strategy_s4_trailing
from backtest_tiered import strategy_tiered
from evaluation_metrics import full_metrics

OUT_DIR = Path("03_RESULT")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic-A 전략
# ═══════════════════════════════════════════════════════════════════════════
def strategy_dynamic(
    defensive: pd.DataFrame,
    qld: pd.DataFrame,
    tqqq: pd.DataFrame,
    initial_capital: float,
    min_A: float = 10.0,        # A > min_A 일 때만 진입
    *,
    entry_retrace_k: float = 0.5,   # 진입: ATH 대비 DD = k × A (중간값 반등 = 0.5)
    invest_divisor: float = 40.0,   # 투입 비율 = min(A / div, 1)
    stop_dd_k: float = 0.75,       # 손절: ATH 대비 DD ≥ k × A (예: 0.75 → 3A/4)
    trailing_stop_pct: float = -0.15,
    series_name: str = "Dynamic-A",
) -> tuple:
    """
    Dynamic-A 전략.
    A = QLD ATH 대비 현재 에피소드 최대 낙폭 (%, 0-100 범위).

    저점↔진입 비율: entry_retrace_k (0.5 = 저점~ATH 거리의 절반까지 반등 시 진입).
    투입 규모: invest_divisor (작을수록 공격적).
    손절선: stop_dd_k × A % (ATH 기준 하방 낙폭이 이 값에 도달하면 청산).
    """
    dates = defensive.index
    dc    = defensive["Close"].values
    qc    = qld["Close"].values
    tc    = tqqq["Close"].values

    def_shares   = initial_capital / dc[0]
    tqqq_shares  = 0.0
    state        = "NORMAL"   # NORMAL / ATTACK / TRAILING

    ath           = qc[0]
    local_low     = qc[0]   # 현재 에피소드 최저점 (ATH 갱신시 리셋)

    entry_A         = 0.0   # 진입 시점 A (손절 기준)
    entry_stop_qld  = 0.0   # 손절 QLD 가격
    entry_frac      = 0.0   # 투입 비율 (기록용)

    tqqq_trail_peak  = 0.0
    tqqq_trail_entry = 0.0

    portfolio = []
    events    = []

    for i in range(len(dates)):
        date = dates[i]
        dcp  = dc[i]
        qcp  = qc[i]
        tcp  = tc[i]

        # ── QLD ATH 갱신 ──────────────────────────────────────────────────
        if qcp > ath:
            ath       = qcp
            local_low = qcp   # 새 ATH → 로컬저점 리셋

        tv = def_shares * dcp + tqqq_shares * tcp

        # ══════════════════════════════════════════════════════════════════
        # TRAILING
        # ══════════════════════════════════════════════════════════════════
        if state == "TRAILING":
            if tcp > tqqq_trail_peak:
                tqqq_trail_peak = tcp

            if tcp <= tqqq_trail_entry:
                def_shares  = tv / dcp
                tqqq_shares = 0.0
                state       = "NORMAL"
                local_low   = qcp   # 현재가로 리셋
                events.append({"Date": date, "type": "TRAIL_FLOOR",
                               "value": tv, "A": entry_A})

            elif tcp <= tqqq_trail_peak * (1.0 + trailing_stop_pct):
                def_shares  = tv / dcp
                tqqq_shares = 0.0
                state       = "NORMAL"
                local_low   = qcp
                events.append({"Date": date, "type": "TRAIL_EXIT",
                               "value": tv, "A": entry_A})

        # ══════════════════════════════════════════════════════════════════
        # ATTACK
        # ══════════════════════════════════════════════════════════════════
        elif state == "ATTACK":
            # QLD ATH 회복 → TRAILING
            if qcp >= ath:
                tqqq_trail_peak  = tcp
                tqqq_trail_entry = tcp
                state = "TRAILING"
                events.append({"Date": date, "type": "TO_TRAILING",
                               "value": tv, "A": entry_A})

            # 손절: QLD가 stop_dd_k × A % (ATH 대비) 이하로 하락
            elif qcp <= entry_stop_qld:
                def_shares  = tv / dcp
                tqqq_shares = 0.0
                state       = "NORMAL"
                local_low   = qcp   # 현재가로 리셋 → 새 에피소드 시작
                events.append({"Date": date, "type": "DYN_STOP",
                               "value": tv, "A": entry_A,
                               "frac": entry_frac,
                               "stop_qld": entry_stop_qld})

        # ══════════════════════════════════════════════════════════════════
        # NORMAL
        # ══════════════════════════════════════════════════════════════════
        elif state == "NORMAL":
            # 로컬저점 갱신
            if qcp < local_low:
                local_low = qcp

            # A 계산
            A = (1.0 - local_low / ath) * 100.0  # 낙폭 % (양수)

            # QLD ATH 회복 시 에피소드 리셋
            if qcp >= ath:
                local_low = qcp

            # 진입: A > min_A AND QLD가 ATH 대비 DD ≤ k_entry × A (= k × 낙폭 만큼 회복)
            elif A > min_A:
                entry_px_thr = ath * (1.0 - entry_retrace_k * A / 100.0)
                if qcp >= entry_px_thr:
                    frac = min(A / invest_divisor, 1.0)
                    tqqq_shares = tv * frac / tcp
                    def_shares  = tv * (1.0 - frac) / dcp
                    state       = "ATTACK"
                    entry_A     = A
                    entry_frac  = frac
                    entry_stop_qld = ath * (1.0 - stop_dd_k * A / 100.0)
                    events.append({"Date": date, "type": "ENTRY",
                                   "value": tv, "A": round(A, 2),
                                   "frac": round(frac * 100, 1),
                                   "entry_dd": round((qcp / ath - 1) * 100, 2),
                                   "stop_dd": round((entry_stop_qld / ath - 1) * 100, 2),
                                   "stop_qld": entry_stop_qld})

        portfolio.append(def_shares * dcp + tqqq_shares * tcp)

    port = pd.Series(portfolio, index=dates, name=series_name)
    ev   = (pd.DataFrame(events)
            if events
            else pd.DataFrame(columns=["Date", "type", "value", "A"]))
    return port, ev


def strategy_dynamic_schedule(
    defensive: pd.DataFrame,
    qld: pd.DataFrame,
    tqqq: pd.DataFrame,
    initial_capital: float,
    param_fn: Callable[[pd.Timestamp], tuple[float, float, float]],
    min_A: float = 10.0,
    trailing_stop_pct: float = -0.15,
    series_name: str = "Dynamic-A schedule",
) -> tuple:
    """
    일자별로 (entry_retrace_k, invest_divisor, stop_dd_k)를 param_fn(date)로 조회.
    ATTACK/TRAILING 중에는 진입 시점에 고정된 손절가를 유지 (연초 파라미터 변경과 무관).
    """
    dates = defensive.index
    dc    = defensive["Close"].values
    qc    = qld["Close"].values
    tc    = tqqq["Close"].values

    def_shares   = initial_capital / dc[0]
    tqqq_shares  = 0.0
    state        = "NORMAL"

    ath           = qc[0]
    local_low     = qc[0]

    entry_A         = 0.0
    entry_stop_qld  = 0.0
    entry_frac      = 0.0

    tqqq_trail_peak  = 0.0
    tqqq_trail_entry = 0.0

    portfolio = []
    events    = []

    for i in range(len(dates)):
        date = dates[i]
        dcp  = dc[i]
        qcp  = qc[i]
        tcp  = tc[i]

        if qcp > ath:
            ath       = qcp
            local_low = qcp

        tv = def_shares * dcp + tqqq_shares * tcp

        if state == "TRAILING":
            if tcp > tqqq_trail_peak:
                tqqq_trail_peak = tcp

            if tcp <= tqqq_trail_entry:
                def_shares  = tv / dcp
                tqqq_shares = 0.0
                state       = "NORMAL"
                local_low   = qcp
                events.append({"Date": date, "type": "TRAIL_FLOOR",
                               "value": tv, "A": entry_A})

            elif tcp <= tqqq_trail_peak * (1.0 + trailing_stop_pct):
                def_shares  = tv / dcp
                tqqq_shares = 0.0
                state       = "NORMAL"
                local_low   = qcp
                events.append({"Date": date, "type": "TRAIL_EXIT",
                               "value": tv, "A": entry_A})

        elif state == "ATTACK":
            if qcp >= ath:
                tqqq_trail_peak  = tcp
                tqqq_trail_entry = tcp
                state = "TRAILING"
                events.append({"Date": date, "type": "TO_TRAILING",
                               "value": tv, "A": entry_A})

            elif qcp <= entry_stop_qld:
                def_shares  = tv / dcp
                tqqq_shares = 0.0
                state       = "NORMAL"
                local_low   = qcp
                events.append({"Date": date, "type": "DYN_STOP",
                               "value": tv, "A": entry_A,
                               "frac": entry_frac,
                               "stop_qld": entry_stop_qld})

        elif state == "NORMAL":
            if qcp < local_low:
                local_low = qcp

            A = (1.0 - local_low / ath) * 100.0

            if qcp >= ath:
                local_low = qcp

            elif A > min_A:
                entry_retrace_k, invest_divisor, stop_dd_k = param_fn(date)
                invest_divisor = max(float(invest_divisor), 1e-9)
                entry_px_thr = ath * (1.0 - entry_retrace_k * A / 100.0)
                if qcp >= entry_px_thr:
                    frac = min(A / invest_divisor, 1.0)
                    tqqq_shares = tv * frac / tcp
                    def_shares  = tv * (1.0 - frac) / dcp
                    state       = "ATTACK"
                    entry_A     = A
                    entry_frac  = frac
                    entry_stop_qld = ath * (1.0 - stop_dd_k * A / 100.0)
                    events.append({"Date": date, "type": "ENTRY",
                                   "value": tv, "A": round(A, 2),
                                   "frac": round(frac * 100, 1),
                                   "entry_dd": round((qcp / ath - 1) * 100, 2),
                                   "stop_dd": round((entry_stop_qld / ath - 1) * 100, 2),
                                   "stop_qld": entry_stop_qld,
                                   "k_ent": round(entry_retrace_k, 4),
                                   "div": round(invest_divisor, 2),
                                   "k_stop": round(stop_dd_k, 4)})

        portfolio.append(def_shares * dcp + tqqq_shares * tcp)

    port = pd.Series(portfolio, index=dates, name=series_name)
    ev   = (pd.DataFrame(events)
            if events
            else pd.DataFrame(columns=["Date", "type", "value", "A"]))
    return port, ev


# ═══════════════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════════════
def run(start="2017-01-01", end="2025-12-31", out_tag="2017_2025"):
    qqq  = load_extended_daily("QQQ")
    qld  = load_extended_daily("QLD")
    tqqq = load_extended_daily("TQQQ")
    common = qqq.index.intersection(qld.index).intersection(tqqq.index)
    common = common[(common >= start) & (common <= end)]
    Q = qqq.loc[common]; L = qld.loc[common]; T = tqqq.loc[common]
    INIT = 100_000

    print("=" * 80)
    print(f"  기간: {common[0].date()} ~ {common[-1].date()}  ({len(common):,}일)")
    print("=" * 80)

    # ── 전략 실행 ──────────────────────────────────────────────────────────
    port_dyn,  ev_dyn  = strategy_dynamic(Q, L, T, INIT)

    port_tier, ev_tier = strategy_tiered(Q, L, T, INIT)

    port_base, _ = strategy_s4_trailing(
        Q, L, T, INIT,
        shallow_drop=-0.10, deep_drop=-0.20,
        shallow_bounce=-0.05, deep_bounce=-0.10,
        half_stop=-0.15, trailing_stop_pct=-0.15,
        half_frac=0.50, full_frac=1.00,
        series_name="2-Tier hf50%",
    )
    port_hf25, _ = strategy_s4_trailing(
        Q, L, T, INIT,
        shallow_drop=-0.10, deep_drop=-0.20,
        shallow_bounce=-0.05, deep_bounce=-0.10,
        half_stop=-0.15, trailing_stop_pct=-0.15,
        half_frac=0.25, full_frac=1.00,
        series_name="2-Tier hf25%",
    )

    b_qqq    = Q["Close"] / Q["Close"].iloc[0] * INIT
    b_qqtqqq = (Q["Close"] / Q["Close"].iloc[0] * 0.5
                + T["Close"] / T["Close"].iloc[0] * 0.5) * INIT
    b_tqqq   = T["Close"] / T["Close"].iloc[0] * INIT

    strats = {
        "Dynamic-A (신규)":         (port_dyn,  ev_dyn),
        "3-Tier":                   (port_tier, ev_tier),
        "2-Tier Anchor (hf=50%)":   (port_base, None),
        "2-Tier Anchor (hf=25%)":   (port_hf25, None),
        "QQQ only":                 (b_qqq,     None),
        "QQQ/TQQQ 50/50":           (b_qqtqqq,  None),
        "TQQQ only":                (b_tqqq,    None),
    }

    # ── 메트릭 출력 ──────────────────────────────────────────────────────────
    print(f"\n  {'전략':30s}  {'CAGR':>7}  {'Sharpe':>7}  {'Sortino':>8}  "
          f"{'MDD':>8}  {'Ulcer':>7}")
    print("  " + "-"*79)
    for label, (port, _) in strats.items():
        m  = full_metrics(port)
        mk = " ◀" if label.startswith("Dynamic") else ""
        print(f"  {label:30s}  {m['cagr']*100:>+6.2f}%  {m['sharpe']:>7.3f}  "
              f"{m['sortino']:>8.3f}  {m['mdd']*100:>+7.2f}%  {m['ulcer']:>7.2f}{mk}")

    # ── Dynamic-A 이벤트 상세 ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  [Dynamic-A] 이벤트 상세")
    print(f"{'='*80}")
    if not ev_dyn.empty:
        for _, r in ev_dyn.iterrows():
            t = str(r["type"])
            if t == "ENTRY":
                stop_dd = float(r["stop_dd"]) if "stop_dd" in r and pd.notna(r.get("stop_dd")) else 0
                print(f"  {str(r['Date'].date()):>10}  {t:>12}  "
                      f"A={float(r['A']):>5.1f}%  투입={float(r['frac']):>5.1f}%  "
                      f"진입={float(r['entry_dd']):>+6.2f}%  손절={stop_dd:>+6.2f}%  "
                      f"자산={r['value']/1000:>8.1f}K")
            elif t == "DYN_STOP":
                print(f"  {str(r['Date'].date()):>10}  {t:>12}  "
                      f"A={float(r['A']):>5.1f}%  손절발동  자산={r['value']/1000:>8.1f}K")
            else:
                print(f"  {str(r['Date'].date()):>10}  {t:>12}  "
                      f"자산={r['value']/1000:>8.1f}K")

    # ── 에피소드 분석 ────────────────────────────────────────────────────────
    def build_ep(ev_df, entry_t, exit_ts):
        if ev_df is None or ev_df.empty:
            return pd.DataFrame()
        ENTRY = {t for t in ev_df["type"] if t in entry_t}
        EXIT  = set(exit_ts)
        rows, ep, entry_r = [], 0, None
        for _, r in ev_df.iterrows():
            if r["type"] in ENTRY and entry_r is None:
                entry_r = r; ep += 1
            elif r["type"] in EXIT and entry_r is not None:
                rows.append({
                    "ep": ep,
                    "entry_dt":  entry_r["Date"],
                    "exit_dt":   r["Date"],
                    "exit_type": r["type"],
                    "A": float(entry_r.get("A", 0)) if "A" in entry_r.index else 0,
                    "frac": float(entry_r.get("frac", 0)) if "frac" in entry_r.index else 0,
                    "dur_days": (r["Date"] - entry_r["Date"]).days,
                    "ret_pct": (r["value"] / entry_r["value"] - 1) * 100,
                })
                entry_r = None
        return pd.DataFrame(rows)

    ep_dyn = build_ep(ev_dyn, {"ENTRY"}, {"TRAIL_FLOOR", "TRAIL_EXIT", "DYN_STOP"})
    if not ep_dyn.empty:
        print(f"\n{'='*80}")
        print("  [Dynamic-A] 에피소드 요약")
        print(f"{'='*80}")
        print(f"\n  {'#':>3}  {'진입일':>10}  {'청산일':>10}  "
              f"{'A':>6}  {'투입':>6}  {'청산':>12}  {'기간':>6}  {'수익률':>8}")
        print("  " + "-"*78)
        for _, r in ep_dyn.iterrows():
            ret  = float(r["ret_pct"])
            mark = "★" if ret > 5 else ("▼" if ret < 0 else " ")
            print(f"  {int(r['ep']):>3}  {str(r['entry_dt'].date()):>10}  "
                  f"{str(r['exit_dt'].date()):>10}  "
                  f"{float(r['A']):>5.1f}%  {float(r['frac']):>5.1f}%  "
                  f"{str(r['exit_type']):>12}  {int(r['dur_days']):>5}일  "
                  f"{ret:>+7.2f}%{mark}")
        wins = ep_dyn[ep_dyn["ret_pct"] > 0]
        print(f"\n  총 {len(ep_dyn)}건  승={len(wins)}건({len(wins)/len(ep_dyn)*100:.0f}%)  "
              f"평균A={ep_dyn['A'].mean():.1f}%  평균투입={ep_dyn['frac'].mean():.1f}%  "
              f"평균수익={ep_dyn['ret_pct'].mean():+.2f}%  평균기간={ep_dyn['dur_days'].mean():.0f}일")

    # ── 연도별 수익률 ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  [연도별 수익률]")
    print(f"{'='*80}")
    years = sorted({d.year for d in common})
    print(f"\n  {'연도':>5}  {'DynA':>8}  {'3-Tier':>8}  {'2T hf50%':>9}  "
          f"{'QQQ':>7}  {'QQ/TQ':>7}  {'TQQQ':>7}")
    print("  " + "-"*62)
    port_list = [port_dyn, port_tier, port_base, b_qqq, b_qqtqqq, b_tqqq]
    for yr in years:
        mask = common.year == yr
        rets = [(p[mask].iloc[-1]/p[mask].iloc[0]-1)*100
                if mask.sum()>=2 else float("nan") for p in port_list]
        dyn_r, t3_r, b50_r, qqq_r, qqtq_r, tqqq_r = rets
        m1 = "★" if dyn_r > t3_r + 0.5 else ("▼" if dyn_r < t3_r - 0.5 else " ")
        m2 = "★" if t3_r > b50_r + 0.5 else ("▼" if t3_r < b50_r - 0.5 else " ")
        print(f"  {yr:>5}  {dyn_r:>+7.1f}%{m1}  {t3_r:>+6.1f}%{m2}  {b50_r:>+7.1f}%  "
              f"{qqq_r:>+5.1f}%  {qqtq_r:>+5.1f}%  {tqqq_r:>+5.1f}%")

    # ── 시각화 ──────────────────────────────────────────────────────────────
    COLORS = {
        "Dynamic-A (신규)":        "#7C3AED",
        "3-Tier":                  "#DC2626",
        "2-Tier Anchor (hf=50%)":  "#6B7280",
        "2-Tier Anchor (hf=25%)":  "#2563EB",
        "QQQ only":                "#111827",
        "QQQ/TQQQ 50/50":          "#D97706",
        "TQQQ only":               "#9CA3AF",
    }

    fig = plt.figure(figsize=(17, 22))
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.52, wspace=0.30,
                            height_ratios=[1.8, 1.0, 1.1, 1.1, 1.1])
    period_label = f"{start[:4]}-{end[:4]}"
    fig.suptitle(
        f"Dynamic-A vs 3-Tier vs 2-Tier vs Benchmarks  |  {period_label}\n"
        f"Dynamic-A: A=낙폭%, 진입=-A/2%(중간값), 투입=min(A/40,100%), 손절=-3A/4%, Trail-10%",
        fontsize=10, fontweight="bold"
    )

    # ① 자산곡선
    ax = fig.add_subplot(gs[0, :])
    for label, (port, _) in strats.items():
        is_bench = label in ("QQQ only", "QQQ/TQQQ 50/50", "TQQQ only")
        ls = "--" if is_bench else "-"
        lw = 2.5 if label.startswith("Dynamic") else (2.0 if "Tier" in label else 1.0)
        alpha = 1.0 if not is_bench else 0.55
        ax.plot(port.index, port / port.iloc[0] * 100,
                label=label, color=COLORS[label], lw=lw, ls=ls, alpha=alpha)
    ax.set_yscale("log"); ax.set_ylabel("Index (start=100, log)")
    ax.set_title("자산곡선 (log)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)

    # ② Drawdown
    ax = fig.add_subplot(gs[1, :])
    for label, port, ck in [
        ("Dynamic-A (신규)", port_dyn,  "Dynamic-A (신규)"),
        ("3-Tier",           port_tier, "3-Tier"),
        ("2-Tier hf50%",     port_base, "2-Tier Anchor (hf=50%)"),
        ("QQQ",              b_qqq,     "QQQ only"),
    ]:
        dd = (port / port.cummax() - 1) * 100
        ax.fill_between(dd.index, dd, 0, alpha=0.18, color=COLORS[ck])
        ax.plot(dd.index, dd, lw=0.9, color=COLORS[ck], label=label)
    ax.set_ylabel("Drawdown (%)"); ax.set_title("Drawdown 비교", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ③ 연도별 수익률
    ax = fig.add_subplot(gs[2, :])
    yr_arr = np.array(years)
    w = 0.22
    bars = [
        ("Dynamic-A (신규)", port_dyn,  "Dynamic-A (신규)"),
        ("3-Tier",           port_tier, "3-Tier"),
        ("2-Tier hf50%",     port_base, "2-Tier Anchor (hf=50%)"),
        ("QQQ only",         b_qqq,     "QQQ only"),
    ]
    for ki, (lbl, port, ck) in enumerate(bars):
        rets = [(port[common.year==yr].iloc[-1]/port[common.year==yr].iloc[0]-1)*100
                if (common.year==yr).sum()>=2 else 0 for yr in years]
        ax.bar(yr_arr + (ki - 1.5) * w, rets, width=w * 0.9,
               color=COLORS[ck], alpha=0.85, label=lbl, edgecolor="white", lw=0.3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("연간 수익률 (%)"); ax.set_title("연도별 수익률", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # ④ Dynamic-A 에피소드 수익률
    ax = fig.add_subplot(gs[3, 0])
    if not ep_dyn.empty:
        colors_ep = ["#7C3AED" if r > 0 else "#9CA3AF" for r in ep_dyn["ret_pct"]]
        ax.bar(range(len(ep_dyn)), ep_dyn["ret_pct"], color=colors_ep,
               alpha=0.85, edgecolor="white", lw=0.3)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("에피소드"); ax.set_ylabel("수익률 (%)")
        ax.set_title(f"Dynamic-A 에피소드 수익률\n"
                     f"{len(ep_dyn)}건  승={int((ep_dyn['ret_pct']>0).sum())}  "
                     f"avg={ep_dyn['ret_pct'].mean():+.1f}%",
                     fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    # ⑤ Dynamic-A vs 3-Tier A별 투입비율 산점도
    ax = fig.add_subplot(gs[3, 1])
    A_range = np.linspace(10, 60, 200)
    frac_dyn  = np.minimum(A_range / 40, 1.0) * 100
    # 3-tier 비교용 (기준점)
    tier3_alloc = {10: 25, 15: 50, 20: 100}
    ax.plot(A_range, frac_dyn, color="#7C3AED", lw=2.5, label="Dynamic-A: min(A/40, 1)")
    ax.axhline(25,  color="#DC2626", lw=1.2, ls="--", alpha=0.7, label="3-Tier L1=25% (A≈10)")
    ax.axhline(50,  color="#DC2626", lw=1.2, ls="-.", alpha=0.7, label="3-Tier L2=50% (A≈15)")
    ax.axhline(100, color="#DC2626", lw=1.2, ls=":",  alpha=0.7, label="3-Tier L3=100% (A≈20)")
    ax.set_xlabel("A (낙폭 %)"); ax.set_ylabel("TQQQ 투입 비율 (%)")
    ax.set_title("A에 따른 TQQQ 투입 비율 비교", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(10, 60); ax.set_ylim(0, 110)

    # ⑥ 메트릭 바 비교
    ax = fig.add_subplot(gs[4, :])
    metric_keys   = ["cagr", "sharpe", "sortino", "mdd", "ulcer"]
    metric_labels = ["CAGR (%)", "Sharpe", "Sortino", "MDD (%)", "Ulcer"]
    # 정규화: 각 메트릭을 분리 표현하기 위해 서브플롯 대신 grouped bar
    compare = [
        ("Dynamic-A",   port_dyn,  "#7C3AED"),
        ("3-Tier",       port_tier, "#DC2626"),
        ("2-Tier hf50%", port_base, "#6B7280"),
        ("QQQ",          b_qqq,     "#111827"),
    ]
    x = np.arange(len(metric_labels))
    w_bar = 0.20
    for ki, (lbl, port, color) in enumerate(compare):
        m = full_metrics(port)
        vals = [
            m["cagr"] * 100,
            m["sharpe"] * 10,
            m["sortino"] * 10,
            m["mdd"] * 100,    # 음수
            -m["ulcer"],       # 음수로 변환
        ]
        ax.bar(x + ki * w_bar, vals, width=w_bar * 0.9,
               color=color, alpha=0.85, label=lbl, edgecolor="white", lw=0.3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x + w_bar * 1.5)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_title("핵심 메트릭 비교 (Sharpe/Sortino ×10, MDD/Ulcer: 높을수록 양호)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    out_path = OUT_DIR / f"dynamic_a_{out_tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  PNG → {out_path}")
    print("\n[완료]")


if __name__ == "__main__":
    # 2017-2025
    run("2017-01-01", "2025-12-31", "2017_2025")
    print()
    # 전체 기간
    run("2002-01-01", "2026-04-30", "full")
