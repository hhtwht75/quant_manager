"""
backtest_tiered.py  (v2)
========================
**사용자 명칭: S4-2** — S4(2단 트레일링)와 구분.

QLD ATH 기준 3-티어 진입 + 트레일링 청산 (구현: strategy_tiered).

티어 (사용자 스펙과 동일):
  L1 : QLD ≤ -10% 후 ≥ -5%까지 반등  → 포트의 **25%** TQQQ
  L2 : QLD ≤ -15% 후 ≥ -7.5%까지 반등 → 포트의 **50%** TQQQ
  L3 : QLD ≤ -20% 후 ≥ -10%까지 반등 → 포트의 **100%** TQQQ

청산 / 트레일: QLD 전고점 회복 시 TRAILING, TQQQ 바닥·-15% 트레일 등 (아래 상세)

기존 2단(참고): HALF(-10%→-5% → 50%), FULL(-20%→-10% → 100%).

청산:
  QLD ATH 회복(dd ≥ 0) → TRAILING 상태
  TRAILING: TQQQ가 진입가 이하 하락(TRAIL_FLOOR) 또는
            TQQQ 고점 대비 -15%(TRAIL_EXIT) → NORMAL + 플래그 리셋

손절:
  L1 STOP: L1(25%)에서 QLD dd ≤ l1_stop (기본 -15%) → QQQ 청산, 플래그 유지
  L2 STOP: L2(50%)에서 QLD dd 악화 시 청산. l2_stop 기본값은 drop_l3(-20%)와 같고,
    이때만 기존과 동일하게 dd < drop_l3 (엄격). l2_stop 을 바꾸면 dd ≤ l2_stop 사용.
  L3: QLD 중간 손절 없음 (FULL_ATTACK 과 동일; Exit 는 ATH→TRAILING 등)

  attack_stops=False: L1_STOP·L2_STOP 비활성 → 깊은 스윙은 L3 업그레이드·회복까지 보유(TRAILING 규칙 동일).

오버랩 처리:
  [A] L1 진입 후 QLD가 -15% 도달 → L1_STOP 발동 (NORMAL 복귀)
      이후 QLD가 -7.5%까지 반등 → L2 신규 진입 (NORMAL→L2)
  [B] L1 진입 후 QLD가 -15% 도달 전 QLD가 -7.5% 이상 반등 → L2 업그레이드
  [C] L1 또는 L2 상태에서 QLD가 -20% 도달 후 -10% 반등 → L3 업그레이드
  [D] L1_STOP과 업그레이드 동시 불가: 손절 dd ≤ -0.15 발동 시 당일 업그레이드 스킵
  [E] NORMAL에서 touched_20 상태에서는 L3 조건(-10% 반등)만 진입 가능
  [F] TRAILING 청산 시 전체 플래그 리셋
"""

from __future__ import annotations

import sys
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


# ═══════════════════════════════════════════════════════════════════════════
# 신규 3-티어 전략
# ═══════════════════════════════════════════════════════════════════════════
def strategy_s4_2_tiered(*args, **kwargs):
    """별칭: 사용자 명명 **S4-2** (= strategy_tiered 동일)."""
    return strategy_tiered(*args, **kwargs)


def strategy_tiered(
    defensive: pd.DataFrame,
    qld: pd.DataFrame,
    tqqq: pd.DataFrame,
    initial_capital: float,
    # QLD ATH 대비 하락 임계값 (기존과 동일한 방식)
    drop_l1:   float = -0.10,   # QLD -10% from ATH
    drop_l2:   float = -0.15,   # QLD -15% from ATH (신규)
    drop_l3:   float = -0.20,   # QLD -20% from ATH
    # QLD ATH 대비 반등 임계값 (기존과 동일한 방식)
    bounce_l1: float = -0.05,   # QLD 반등 → -5%  from ATH
    bounce_l2: float = -0.075,  # QLD 반등 → -7.5% from ATH (신규)
    bounce_l3: float = -0.10,   # QLD 반등 → -10% from ATH
    # TQQQ 투입 비율
    frac_l1: float = 0.25,
    frac_l2: float = 0.50,
    frac_l3: float = 1.00,
    # L1 손절 (QLD가 L1 상태에서 이 깊이 이하로 악화 시 청산, 플래그 유지)
    l1_stop: float = -0.15,
    # L2 손절: l2_stop == drop_l3 이면 dd < drop_l3 (레거시). 다르면 dd <= l2_stop
    l2_stop: float = -0.20,
    # L1/L2 공격·업그레이드 구간의 QLD 악화 손절 (False면 L1_STOP·L2_STOP 비활성)
    attack_stops: bool = True,
    # 트레일링 스탑 (TRAILING 상태에서 TQQQ 고점 대비)
    trailing_stop_pct: float = -0.15,
    series_name: str = "Tiered",
    daily_tqqq_weight_out: list | None = None,
) -> tuple:
    dates  = defensive.index
    dc     = defensive["Close"].values
    qc     = qld["Close"].values
    tc     = tqqq["Close"].values

    def_shares   = initial_capital / dc[0]
    tqqq_shares  = 0.0
    state        = "NORMAL"   # NORMAL / L1 / L2 / L3 / TRAILING

    ath            = qc[0]
    touched_10     = touched_15 = touched_20 = False
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
            ath = qcp
        dd = qcp / ath - 1.0    # QLD drawdown from ATH

        # ── 터치 플래그 갱신 ──────────────────────────────────────────────
        if dd <= drop_l1:
            touched_10 = True
        if dd <= drop_l2:
            touched_15 = True
        if dd <= drop_l3:
            touched_20 = True

        tv = def_shares * dcp + tqqq_shares * tcp

        # ══════════════════════════════════════════════════════════════════
        # TRAILING 상태
        # ══════════════════════════════════════════════════════════════════
        if state == "TRAILING":
            if tcp > tqqq_trail_peak:
                tqqq_trail_peak = tcp

            # Rule 1: TQQQ가 진입가 이하 → 수익 확정
            if tcp <= tqqq_trail_entry:
                def_shares  = tv / dcp
                tqqq_shares = 0.0
                state       = "NORMAL"
                touched_10  = touched_15 = touched_20 = False
                events.append({"Date": date, "type": "TRAIL_FLOOR", "value": tv})

            # Rule 2: TQQQ 고점 대비 -10% → 청산
            elif tcp <= tqqq_trail_peak * (1.0 + trailing_stop_pct):
                def_shares  = tv / dcp
                tqqq_shares = 0.0
                state       = "NORMAL"
                touched_10  = touched_15 = touched_20 = False
                events.append({"Date": date, "type": "TRAIL_EXIT", "value": tv})

        # ══════════════════════════════════════════════════════════════════
        # L1 상태 (25% TQQQ)
        # ══════════════════════════════════════════════════════════════════
        elif state == "L1":
            # QLD ATH 회복 → TRAILING 진입
            if dd >= 0:
                tqqq_trail_peak  = tcp
                tqqq_trail_entry = tcp
                state = "TRAILING"
                events.append({"Date": date, "type": "TO_TRAILING_from_L1", "value": tv})

            # L1 손절: QLD -15% 도달 → NORMAL 복귀, 플래그 유지
            elif attack_stops and dd <= l1_stop:
                def_shares  = tv / dcp
                tqqq_shares = 0.0
                state = "NORMAL"
                events.append({"Date": date, "type": "L1_STOP", "value": tv})

            # L3 업그레이드: QLD -20% 터치 후 -10%까지 반등
            elif touched_20 and dd >= bounce_l3:
                tqqq_shares = tv * frac_l3 / tcp
                def_shares  = tv * (1.0 - frac_l3) / dcp
                state = "L3"
                events.append({"Date": date, "type": "TO_L3_from_L1", "value": tv})

            # L2 업그레이드: QLD -15% 터치 후 -7.5%까지 반등 (아직 -20% 미도달)
            elif touched_15 and not touched_20 and dd >= bounce_l2:
                tqqq_shares = tv * frac_l2 / tcp
                def_shares  = tv * (1.0 - frac_l2) / dcp
                state = "L2"
                events.append({"Date": date, "type": "TO_L2_from_L1", "value": tv})

        # ══════════════════════════════════════════════════════════════════
        # L2 상태 (50% TQQQ)
        # ══════════════════════════════════════════════════════════════════
        elif state == "L2":
            # QLD ATH 회복 → TRAILING 진입
            if dd >= 0:
                tqqq_trail_peak  = tcp
                tqqq_trail_entry = tcp
                state = "TRAILING"
                events.append({"Date": date, "type": "TO_TRAILING_from_L2", "value": tv})

            # L2 손절 (레거시: l2_stop==drop_l3 일 때만 dd < drop_l3, 그 외 dd <= l2_stop)
            elif attack_stops and (
                (l2_stop == drop_l3 and dd < drop_l3)
                or (l2_stop != drop_l3 and dd <= l2_stop)
            ):
                def_shares  = tv / dcp
                tqqq_shares = 0.0
                state = "NORMAL"
                events.append({"Date": date, "type": "L2_STOP", "value": tv})

            # L3 업그레이드: QLD -20% 터치 후 -10%까지 반등
            elif touched_20 and dd >= bounce_l3:
                tqqq_shares = tv * frac_l3 / tcp
                def_shares  = tv * (1.0 - frac_l3) / dcp
                state = "L3"
                events.append({"Date": date, "type": "TO_L3_from_L2", "value": tv})

        # ══════════════════════════════════════════════════════════════════
        # L3 상태 (100% TQQQ)
        # ══════════════════════════════════════════════════════════════════
        elif state == "L3":
            # QLD ATH 회복 → TRAILING 진입
            if dd >= 0:
                tqqq_trail_peak  = tcp
                tqqq_trail_entry = tcp
                state = "TRAILING"
                events.append({"Date": date, "type": "TO_TRAILING_from_L3", "value": tv})

        # ══════════════════════════════════════════════════════════════════
        # NORMAL 상태
        # ══════════════════════════════════════════════════════════════════
        elif state == "NORMAL":
            # QLD ATH → 모든 플래그 리셋
            if dd >= 0:
                touched_10 = touched_15 = touched_20 = False

            # L3 신규 진입 (touched_20 + -10% 반등)
            elif touched_20 and dd >= bounce_l3:
                tqqq_shares = tv * frac_l3 / tcp
                def_shares  = tv * (1.0 - frac_l3) / dcp
                state = "L3"
                events.append({"Date": date, "type": "TO_L3_from_NORMAL", "value": tv})

            # L2 신규 진입 (touched_15, NOT touched_20, -7.5% 반등)
            elif touched_15 and not touched_20 and dd >= bounce_l2:
                tqqq_shares = tv * frac_l2 / tcp
                def_shares  = tv * (1.0 - frac_l2) / dcp
                state = "L2"
                events.append({"Date": date, "type": "TO_L2_from_NORMAL", "value": tv})

            # L1 신규 진입 (touched_10, NOT touched_15, -5% 반등)
            elif touched_10 and not touched_15 and dd >= bounce_l1:
                tqqq_shares = tv * frac_l1 / tcp
                def_shares  = tv * (1.0 - frac_l1) / dcp
                state = "L1"
                events.append({"Date": date, "type": "TO_L1_from_NORMAL", "value": tv})

        tv_end = def_shares * dcp + tqqq_shares * tcp
        if daily_tqqq_weight_out is not None:
            daily_tqqq_weight_out.append(
                (tqqq_shares * tcp) / tv_end if tv_end > 1e-18 else 0.0
            )
        portfolio.append(tv_end)

    port_series = pd.Series(portfolio, index=dates, name=series_name)
    ev_df = (pd.DataFrame(events)
             if events
             else pd.DataFrame(columns=["Date", "type", "value"]))
    return port_series, ev_df


def strategy_tiered_s42_chain_sl(
    defensive: pd.DataFrame,
    qld: pd.DataFrame,
    tqqq: pd.DataFrame,
    initial_capital: float,
    drop_l1: float = -0.10,
    drop_l2: float = -0.15,
    drop_l3: float = -0.20,
    bounce_l1: float = -0.05,
    bounce_l2: float = -0.075,
    bounce_l3: float = -0.10,
    l1_stop: float = -0.15,
    l3_stop: float = -0.25,
    frac_l1: float = 0.25,
    frac_l2: float = 0.50,
    frac_l3: float = 1.00,
    trailing_stop_pct: float = -0.15,
    series_name: str = "S4-2 chain SL",
    daily_tqqq_weight_out: list | None = None,
) -> tuple:
    """S4-2 변형: L1/L2/L3 각각 단계 손절 + 손절 후에도 터치 플래그 유지(재진입 가능).

    • L1: -10% 터치 후 -5% 반등 → 25% TQQQ. dd ≤ -15% → L1_STOP → QQQ, 터치 플래그 유지.
    • L2: -15% 터치 후 -7.5% 반등 → 50% TQQQ. dd ≤ -20% → L2_STOP → QQQ, 플래그 유지.
      (L2에서 L3로의 업그레이드 없음: -20% 도달 시 즉시 L2_STOP.)
    • L3: -20% 터치 후 -10% 반등 → 100% TQQQ. dd ≤ -25% → L3_STOP → QQQ, 플래그 유지.
    • L1에서 L3 직행 없음(-15% 손절이 -20% 이전에 선행).
    • TRAILING 청산(TRAIL_FLOOR/EXIT) 시에는 기존과 같이 터치 플래그 전체 리셋.
    """
    dates = defensive.index
    dc = defensive["Close"].values
    qc = qld["Close"].values
    tc = tqqq["Close"].values

    def_shares = initial_capital / dc[0]
    tqqq_shares = 0.0
    state = "NORMAL"

    ath = qc[0]
    touched_10 = touched_15 = touched_20 = False
    tqqq_trail_peak = 0.0
    tqqq_trail_entry = 0.0

    portfolio = []
    events = []

    for i in range(len(dates)):
        date = dates[i]
        dcp = dc[i]
        qcp = qc[i]
        tcp = tc[i]

        if qcp > ath:
            ath = qcp
        dd = qcp / ath - 1.0

        if dd <= drop_l1:
            touched_10 = True
        if dd <= drop_l2:
            touched_15 = True
        if dd <= drop_l3:
            touched_20 = True

        tv = def_shares * dcp + tqqq_shares * tcp

        if state == "TRAILING":
            if tcp > tqqq_trail_peak:
                tqqq_trail_peak = tcp

            if tcp <= tqqq_trail_entry:
                def_shares = tv / dcp
                tqqq_shares = 0.0
                state = "NORMAL"
                touched_10 = touched_15 = touched_20 = False
                events.append({"Date": date, "type": "TRAIL_FLOOR", "value": tv})

            elif tcp <= tqqq_trail_peak * (1.0 + trailing_stop_pct):
                def_shares = tv / dcp
                tqqq_shares = 0.0
                state = "NORMAL"
                touched_10 = touched_15 = touched_20 = False
                events.append({"Date": date, "type": "TRAIL_EXIT", "value": tv})

        elif state == "L1":
            if dd >= 0:
                tqqq_trail_peak = tcp
                tqqq_trail_entry = tcp
                state = "TRAILING"
                events.append({"Date": date, "type": "TO_TRAILING_from_L1", "value": tv})

            elif dd <= l1_stop:
                def_shares = tv / dcp
                tqqq_shares = 0.0
                state = "NORMAL"
                events.append({"Date": date, "type": "L1_STOP", "value": tv})

            elif touched_15 and not touched_20 and dd >= bounce_l2:
                tqqq_shares = tv * frac_l2 / tcp
                def_shares = tv * (1.0 - frac_l2) / dcp
                state = "L2"
                events.append({"Date": date, "type": "TO_L2_from_L1", "value": tv})

        elif state == "L2":
            if dd >= 0:
                tqqq_trail_peak = tcp
                tqqq_trail_entry = tcp
                state = "TRAILING"
                events.append({"Date": date, "type": "TO_TRAILING_from_L2", "value": tv})

            elif dd <= drop_l3:
                def_shares = tv / dcp
                tqqq_shares = 0.0
                state = "NORMAL"
                events.append({"Date": date, "type": "L2_STOP", "value": tv})

        elif state == "L3":
            if dd >= 0:
                tqqq_trail_peak = tcp
                tqqq_trail_entry = tcp
                state = "TRAILING"
                events.append({"Date": date, "type": "TO_TRAILING_from_L3", "value": tv})

            elif dd <= l3_stop:
                def_shares = tv / dcp
                tqqq_shares = 0.0
                state = "NORMAL"
                events.append({"Date": date, "type": "L3_STOP", "value": tv})

        elif state == "NORMAL":
            if dd >= 0:
                touched_10 = touched_15 = touched_20 = False

            elif touched_20 and dd >= bounce_l3:
                tqqq_shares = tv * frac_l3 / tcp
                def_shares = tv * (1.0 - frac_l3) / dcp
                state = "L3"
                events.append({"Date": date, "type": "TO_L3_from_NORMAL", "value": tv})

            elif touched_15 and not touched_20 and dd >= bounce_l2:
                tqqq_shares = tv * frac_l2 / tcp
                def_shares = tv * (1.0 - frac_l2) / dcp
                state = "L2"
                events.append({"Date": date, "type": "TO_L2_from_NORMAL", "value": tv})

            elif touched_10 and not touched_15 and dd >= bounce_l1:
                tqqq_shares = tv * frac_l1 / tcp
                def_shares = tv * (1.0 - frac_l1) / dcp
                state = "L1"
                events.append({"Date": date, "type": "TO_L1_from_NORMAL", "value": tv})

        tv_end = def_shares * dcp + tqqq_shares * tcp
        if daily_tqqq_weight_out is not None:
            daily_tqqq_weight_out.append(
                (tqqq_shares * tcp) / tv_end if tv_end > 1e-18 else 0.0
            )
        portfolio.append(tv_end)

    port_series = pd.Series(portfolio, index=dates, name=series_name)
    ev_df = (
        pd.DataFrame(events)
        if events
        else pd.DataFrame(columns=["Date", "type", "value"])
    )
    return port_series, ev_df


# ═══════════════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # ── 데이터 ──────────────────────────────────────────────────────────────
    qqq  = load_extended_daily("QQQ")
    qld  = load_extended_daily("QLD")
    tqqq = load_extended_daily("TQQQ")
    common = (qqq.index.intersection(qld.index).intersection(tqqq.index))
    common = common[(common >= "2002-01-01") & (common <= "2026-04-30")]
    Q = qqq.loc[common]; L = qld.loc[common]; T = tqqq.loc[common]
    N = len(common)

    print("=" * 76)
    print(f"  기간: {common[0].date()} ~ {common[-1].date()}  ({N:,}일)")
    print("=" * 76)

    INIT = 100_000

    # ── 전략 실행 ────────────────────────────────────────────────────────────
    # 신규 3-티어 (L1=25%, L2=50%, L3=100%)
    port_tier, ev_tier = strategy_tiered(Q, L, T, INIT, series_name="Tiered")

    # 기존 앵커 hf=50% (비교 기준)
    port_base, ev_base = strategy_s4_trailing(
        Q, L, T, INIT,
        shallow_drop=-0.10, deep_drop=-0.20,
        shallow_bounce=-0.05, deep_bounce=-0.10,
        half_stop=-0.15, trailing_stop_pct=-0.15,
        half_frac=0.50, full_frac=1.00,
        series_name="Base (hf=50%)",
    )

    # 기존 앵커 hf=25% (Monte Carlo 최적 케이스)
    port_hf25, ev_hf25 = strategy_s4_trailing(
        Q, L, T, INIT,
        shallow_drop=-0.10, deep_drop=-0.20,
        shallow_bounce=-0.05, deep_bounce=-0.10,
        half_stop=-0.15, trailing_stop_pct=-0.15,
        half_frac=0.25, full_frac=1.00,
        series_name="Base (hf=25%)",
    )

    # 벤치마크
    b_qqq  = Q["Close"] / Q["Close"].iloc[0] * INIT
    b_tqqq = T["Close"] / T["Close"].iloc[0] * INIT

    # ── 메트릭 요약 ─────────────────────────────────────────────────────────
    strats = {
        "Tiered (L1=25%,L2=50%,L3=100%)": (port_tier, ev_tier),
        "Base Anchor (hf=50%)":            (port_base, ev_base),
        "Base Anchor (hf=25%)":            (port_hf25, ev_hf25),
        "B/M: QQQ":                        (b_qqq,    None),
        "B/M: TQQQ":                       (b_tqqq,   None),
    }

    print(f"\n  {'전략':35s}  {'CAGR':>7}  {'Sharpe':>7}  {'Sortino':>8}  "
          f"{'MDD':>8}  {'Ulcer':>7}")
    print("  " + "-" * 82)
    for label, (port, ev) in strats.items():
        m = full_metrics(port)
        print(f"  {label:35s}  {m['cagr']*100:>+6.2f}%  {m['sharpe']:>7.3f}  "
              f"{m['sortino']:>8.3f}  {m['mdd']*100:>+7.2f}%  {m['ulcer']:>7.2f}")

    # ── 이벤트 상세 출력 ─────────────────────────────────────────────────────
    def print_events(label, ev_df):
        print(f"\n{'='*76}")
        print(f"  [{label}] 이벤트 목록")
        print(f"{'='*76}")
        if ev_df is None or ev_df.empty:
            print("  (없음)")
            return
        type_counts = ev_df["type"].value_counts().to_dict()
        cnt_str = "  ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
        print(f"  {cnt_str}")
        print()
        for _, r in ev_df.iterrows():
            print(f"  {str(r['Date'].date()):>10}  {str(r['type']):>28}  "
                  f"자산={r['value']/1000:>8.1f}K")

    print_events("Tiered", ev_tier)
    print_events("Base Anchor hf=50%", ev_base)

    # ── 연도별 수익률 비교 ────────────────────────────────────────────────────
    print(f"\n{'='*76}")
    print("  [연도별 수익률]")
    print(f"{'='*76}")
    years = sorted({d.year for d in common})
    print(f"\n  {'연도':>5}  {'Tiered':>8}  {'Base hf50%':>10}  "
          f"{'Base hf25%':>10}  {'QQQ':>8}  {'TQQQ':>8}")
    print("  " + "-" * 62)
    for yr in years:
        mask = common.year == yr
        ports = [port_tier, port_base, port_hf25, b_qqq, b_tqqq]
        rets = []
        for p in ports:
            sub = p[mask]
            rets.append((sub.iloc[-1] / sub.iloc[0] - 1) * 100 if len(sub) >= 2 else float("nan"))
        tier_r, base_r, hf25_r, qqq_r, tqqq_r = rets
        if not np.isnan(tier_r):
            mark = "★" if tier_r > base_r + 1 else ("▼" if tier_r < base_r - 1 else " ")
            print(f"  {yr:>5}  {tier_r:>+7.1f}%{mark}  {base_r:>+8.1f}%  "
                  f"{hf25_r:>+8.1f}%  {qqq_r:>+6.1f}%  {tqqq_r:>+6.1f}%")

    # ── 에피소드 요약 ────────────────────────────────────────────────────────
    def build_episodes(ev_df: pd.DataFrame) -> pd.DataFrame:
        if ev_df is None or ev_df.empty:
            return pd.DataFrame()
        ENTRY = {t for t in ev_df["type"] if t.startswith("TO_L")}
        EXIT  = {"TRAIL_FLOOR", "TRAIL_EXIT", "L1_STOP", "L2_STOP"}
        rows, ep, entry_r = [], 0, None
        for _, r in ev_df.iterrows():
            if r["type"] in ENTRY and entry_r is None:
                entry_r = r; ep += 1
            elif r["type"] in EXIT and entry_r is not None:
                rows.append({
                    "ep": ep,
                    "entry_dt": entry_r["Date"],
                    "entry_type": entry_r["type"],
                    "exit_dt":  r["Date"],
                    "exit_type": r["type"],
                    "dur_days": (r["Date"] - entry_r["Date"]).days,
                    "ret_pct": (r["value"] / entry_r["value"] - 1) * 100,
                })
                entry_r = None
        return pd.DataFrame(rows)

    ep_tier = build_episodes(ev_tier)
    if not ep_tier.empty:
        print(f"\n{'='*76}")
        print("  [Tiered] 에피소드 요약")
        print(f"{'='*76}")
        print(f"\n  {'#':>3}  {'진입일':>10}  {'청산일':>10}  {'유형':>28}  "
              f"{'청산유형':>15}  {'기간(일)':>7}  {'수익률':>7}")
        print("  " + "-" * 95)
        for _, r in ep_tier.iterrows():
            ret = float(r["ret_pct"])
            mark = "★" if ret > 5 else ("▼" if ret < 0 else " ")
            print(f"  {int(r['ep']):>3}  {str(r['entry_dt'].date()):>10}  "
                  f"{str(r['exit_dt'].date()):>10}  {str(r['entry_type']):>28}  "
                  f"{str(r['exit_type']):>15}  {int(r['dur_days']):>7}  "
                  f"{ret:>+6.1f}%{mark}")
        wins  = ep_tier[ep_tier["ret_pct"] > 0]
        print(f"\n  총 {len(ep_tier)}건  승={len(wins)}건({len(wins)/len(ep_tier)*100:.0f}%)  "
              f"평균수익={ep_tier['ret_pct'].mean():+.1f}%  "
              f"평균기간={ep_tier['dur_days'].mean():.0f}일")

    # ── 시각화 ────────────────────────────────────────────────────────────────
    COLORS = {
        "Tiered":      "#DC2626",
        "Base hf50%":  "#6B7280",
        "Base hf25%":  "#2563EB",
        "QQQ":         "#111827",
        "TQQQ":        "#9CA3AF",
    }

    fig = plt.figure(figsize=(16, 18))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.30,
                            height_ratios=[1.8, 1.0, 1.0, 1.2])
    fig.suptitle(
        "3-티어 진입 전략 vs Anchor  |  2002-2026\n"
        "L1: QLD≤-10%→-5%반등→25%TQQQ  |  "
        "L2: QLD≤-15%→-7.5%반등→50%TQQQ  |  "
        "L3: QLD≤-20%→-10%반등→100%TQQQ  |  TrailStop-10%",
        fontsize=10, fontweight="bold"
    )

    # ① 자산곡선
    ax = fig.add_subplot(gs[0, :])
    plot_map = [
        ("Tiered (L1=25%,L2=50%,L3=100%)", port_tier, "Tiered", "-", 2.0),
        ("Base Anchor (hf=50%)",            port_base, "Base hf50%", "-", 2.0),
        ("Base Anchor (hf=25%)",            port_hf25, "Base hf25%", "--", 1.2),
        ("B/M: QQQ",                        b_qqq,     "QQQ", ":", 1.0),
        ("B/M: TQQQ",                       b_tqqq,    "TQQQ", ":", 0.8),
    ]
    for label, port, ck, ls, lw in plot_map:
        ax.plot(port.index, port / port.iloc[0] * 100,
                label=label, color=COLORS[ck], lw=lw, ls=ls, alpha=0.9)
    ax.set_yscale("log"); ax.set_ylabel("Index (start=100, log)")
    ax.set_title("자산곡선 비교 (log scale)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)

    # ② Drawdown
    ax = fig.add_subplot(gs[1, :])
    for label, port, ck in [
        ("Tiered", port_tier, "Tiered"),
        ("Base hf50%", port_base, "Base hf50%"),
        ("QQQ", b_qqq, "QQQ"),
    ]:
        dd_ser = (port / port.cummax() - 1) * 100
        ax.fill_between(dd_ser.index, dd_ser, 0, alpha=0.20, color=COLORS[ck])
        ax.plot(dd_ser.index, dd_ser, lw=0.8, color=COLORS[ck], label=label)
    ax.set_ylabel("Drawdown (%)"); ax.set_title("Drawdown 비교", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ③ 연도별 수익률 바
    ax = fig.add_subplot(gs[2, :])
    yr_arr = np.array(years)
    w = 0.3
    bars = [
        ("Tiered", port_tier, "Tiered"),
        ("Base hf50%", port_base, "Base hf50%"),
        ("QQQ", b_qqq, "QQQ"),
    ]
    for ki, (lbl, port, ck) in enumerate(bars):
        rets = [(port[common.year == yr].iloc[-1] / port[common.year == yr].iloc[0] - 1) * 100
                if (common.year == yr).sum() >= 2 else 0 for yr in years]
        ax.bar(yr_arr + (ki - 1) * w, rets, width=w * 0.9,
               color=COLORS[ck], alpha=0.8, label=lbl, edgecolor="white", lw=0.3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("연간 수익률 (%)"); ax.set_title("연도별 수익률", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # ④ 에피소드 수익률
    ax = fig.add_subplot(gs[3, 0])
    if not ep_tier.empty:
        colors_ep = ["#DC2626" if r > 0 else "#9CA3AF" for r in ep_tier["ret_pct"]]
        ax.bar(range(len(ep_tier)), ep_tier["ret_pct"], color=colors_ep,
               alpha=0.85, edgecolor="white", lw=0.3)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("에피소드 번호"); ax.set_ylabel("수익률 (%)")
        ax.set_title("Tiered 에피소드별 수익률", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    # ⑤ 진입 유형 분포
    ax = fig.add_subplot(gs[3, 1])
    if not ev_tier.empty:
        entry_types = {t: int((ev_tier["type"] == t).sum())
                       for t in ev_tier["type"].unique()
                       if not t.startswith("TRAIL") and t not in ("L1_STOP", "L2_STOP")}
        entry_types = {k: v for k, v in entry_types.items() if v > 0}
        if entry_types:
            ax.barh(list(entry_types.keys()), list(entry_types.values()),
                    color="#DC2626", alpha=0.7, edgecolor="white")
            ax.set_xlabel("건수")
            ax.set_title("진입/업그레이드 유형", fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")

    plt.savefig(OUT_DIR / "tiered_backtest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  PNG → {OUT_DIR}/tiered_backtest.png")
    print("\n[완료]")


if __name__ == "__main__":
    main()
