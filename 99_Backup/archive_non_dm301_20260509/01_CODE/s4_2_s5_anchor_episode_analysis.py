"""
S4-2 vs S5 앵커 — 에피소드별 정밀 비교.

1) QLD 종가 cummax가 갱신되는 날을 경계로 **ATH 사이클**(피크→다음 피크)을 정의.
   각 사이클에서 두 전략의 구간 수익률·차이·QLD 최대 낙폭을 산출.

2) S4-2 이벤트 로그 기준 **레버 진입~완전 청산** 에피소-d(TO_L* → TRAIL*/STOP)마다
   동일 캘린더 구간의 S5 수익률을 붙여 해석.

출력: 콘솔 요약 + 03_RESULT/sensitivity/s4_2_s5_episodes_*.csv/json

실행: python3 01_CODE/s4_2_s5_anchor_episode_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import load_extended_daily, strategy_s5  # noqa: E402
from backtest_tiered import strategy_tiered  # noqa: E402

OUT_DIR = _ROOT / "03_RESULT" / "sensitivity"
START = pd.Timestamp("2002-10-01")
CAP = 100_000
S5_ANCHOR = dict(
    beta=0.5,
    min_drop=0.10,
    max_drop=0.20,
    trailing_stop_pct=-0.15,
    use_stop_loss=True,
    position_mode="exp",
    exp_frac_lo=0.25,
    exp_frac_hi=1.00,
    exp_base=2.0,
)


def ath_step_dates(close: pd.Series) -> pd.DatetimeIndex:
    """cummax가 커지는 모든 일자(구간 시작점). 첫 날 포함."""
    rm = close.cummax().values
    inc = np.r_[True, rm[1:] > rm[:-1]]
    return close.index[inc]


def period_return(port: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    s = port.index.get_indexer([start], method="ffill")[0]
    e = port.index.get_indexer([end], method="ffill")[0]
    if s < 0 or e < 0:
        return float("nan")
    return float(port.iloc[e] / port.iloc[s] - 1.0)


def qld_cycle_mdd(qld_close: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    sub = qld_close.loc[start:end]
    if sub.empty:
        return float("nan")
    peak0 = float(sub.iloc[0])
    peaks = sub.cummax()
    dd = sub / peaks - 1.0
    return float(dd.min())


def build_s42_episodes(ev_df: pd.DataFrame) -> pd.DataFrame:
    if ev_df is None or ev_df.empty:
        return pd.DataFrame()
    ENTRY = {t for t in ev_df["type"].unique() if str(t).startswith("TO_L")}
    EXIT = {"TRAIL_FLOOR", "TRAIL_EXIT", "L1_STOP", "L2_STOP"}
    rows, ep, entry_r = [], 0, None
    for _, r in ev_df.iterrows():
        if r["type"] in ENTRY and entry_r is None:
            entry_r = r
            ep += 1
        elif r["type"] in EXIT and entry_r is not None:
            rows.append({
                "ep": ep,
                "entry_dt": pd.Timestamp(entry_r["Date"]),
                "entry_type": entry_r["type"],
                "exit_dt": pd.Timestamp(r["Date"]),
                "exit_type": r["type"],
                "dur_days": (pd.Timestamp(r["Date"]) - pd.Timestamp(entry_r["Date"])).days,
                "s42_ep_ret": float(r["value"] / entry_r["value"] - 1.0),
            })
            entry_r = None
    return pd.DataFrame(rows)


def events_in_range(ev: pd.DataFrame, a: pd.Timestamp, b: pd.Timestamp) -> dict:
    if ev.empty:
        return {}
    m = (ev["Date"] >= a) & (ev["Date"] <= b)
    sub = ev.loc[m]
    return sub["type"].value_counts().to_dict()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qqq = load_extended_daily("QQQ")
    qld = load_extended_daily("QLD")
    tqqq = load_extended_daily("TQQQ")
    c = qqq.index.intersection(qld.index).intersection(tqqq.index)
    Q, L, T = qqq.loc[c], qld.loc[c], tqqq.loc[c]
    m = Q.index >= START
    Q, L, T = Q.loc[m], L.loc[m], T.loc[m]
    qld_c = L["Close"]

    p42, ev42 = strategy_tiered(Q, L, T, CAP)
    p5, ev5 = strategy_s5(Q, L, T, CAP, stop_factor=0.75, **S5_ANCHOR)

    peaks = ath_step_dates(qld_c)
    cycle_rows = []
    for i in range(len(peaks) - 1):
        s, e = peaks[i], peaks[i + 1]
        r42 = period_return(p42, s, e)
        r5 = period_return(p5, s, e)
        cycle_rows.append({
            "cyc": i + 1,
            "start": str(s.date()),
            "end": str(e.date()),
            "days": int((e - s).days),
            "qld_mdd": round(qld_cycle_mdd(qld_c, s, e) * 100, 2),
            "s42_ret": round(r42 * 100, 3),
            "s5_ret": round(r5 * 100, 3),
            "s5_minus_s42_pct": round((r5 - r42) * 100, 3),
        })
    cyc_df = pd.DataFrame(cycle_rows)

    ep42 = build_s42_episodes(ev42)
    if not ep42.empty:
        ep42["s5_same_win_ret"] = [
            round(period_return(p5, row.entry_dt, row.exit_dt) * 100, 3)
            for _, row in ep42.iterrows()
        ]
        ep42["edge_s5_minus_s42_pp"] = round(
            ep42["s5_same_win_ret"] - ep42["s42_ep_ret"] * 100, 3
        )

    # 요약 통계
    tot42 = float(p42.iloc[-1] / p42.iloc[0] - 1.0)
    tot5 = float(p5.iloc[-1] / p5.iloc[0] - 1.0)
    cyc_df["s5_wins_cycle"] = cyc_df["s5_minus_s42_pct"] > 0

    summary = {
        "period": [str(Q.index[0].date()), str(Q.index[-1].date())],
        "n_days": int(len(Q)),
        "total_return_pct_s42": round(tot42 * 100, 4),
        "total_return_pct_s5": round(tot5 * 100, 4),
        "n_ath_cycles": int(len(cyc_df)),
        "cycles_s5_better": int(cyc_df["s5_wins_cycle"].sum()),
        "mean_cycle_diff_s5_minus_s42_pp": float(cyc_df["s5_minus_s42_pct"].mean()),
        "median_cycle_diff_pp": float(cyc_df["s5_minus_s42_pct"].median()),
        "correlation_qld_mdd_vs_diff": float(
            np.corrcoef(cyc_df["qld_mdd"], cyc_df["s5_minus_s42_pct"])[0, 1]
        ) if len(cyc_df) > 2 else None,
        "largest_s5_advantage_pp": float(cyc_df["s5_minus_s42_pct"].max()),
        "largest_s42_advantage_pp": float(-cyc_df["s5_minus_s42_pct"].min()),
        "s42_episode_count": int(len(ep42)),
    }

    if summary["correlation_qld_mdd_vs_diff"] is not None:
        summary["corr_note"] = (
            "QLD 사이클 MDD(%)와 (S5−S4-2) p.p.차 : 양(+)이면 깊은 조정 사이클에서 "
            "차이가 더 내려가 S4-2 상대 우위가 커지는 경향(약한 선형관계)."
        )

    tag = f"{Q.index[0].strftime('%Y%m%d')}_{Q.index[-1].strftime('%Y%m%d')}"
    csv_c = OUT_DIR / f"s4_2_s5_ath_cycles_{tag}.csv"
    cyc_df.to_csv(csv_c, index=False, encoding="utf-8-sig")
    if not ep42.empty:
        csv_e = OUT_DIR / f"s4_2_s5_s42_episodes_{tag}.csv"
        ep42.to_csv(csv_e, index=False, encoding="utf-8-sig")

    json_path = OUT_DIR / f"s4_2_s5_episode_analysis_{tag}.json"
    payload = {
        "summary": summary,
        "interpretation": [
            "S5 앵커는 같은 QLD 스윙에 대해 진입 시 연속(exp) 비중으로 종종 더 높은 TQQQ 노출을 가져가고, "
            "손절은 '스윙 저점 추가 악화'라 S4-2의 고정 -15%/-20% QLD dd 스탑과 타점이 다름.",
            "ATH 사이클 전체로 보면 장기 상승장에서는 높은 평균 레버로 S5 누적이 커지기 쉽고, "
            "특정 깊은 조정 사이클에서는 S4-2의 단계적/손절 구조가 상대 수익에 유리할 수 있음.",
        ],
        "worst_cycles_for_s42_vs_s5": cyc_df.nsmallest(5, "s5_minus_s42_pct").to_dict(
            orient="records"
        ),
        "best_cycles_for_s5": cyc_df.nlargest(5, "s5_minus_s42_pct").to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # 콘솔
    W = 88
    print("=" * W)
    print(f"  S4-2 vs S5 앵커 에피소드 분석  |  {Q.index[0].date()} ~ {Q.index[-1].date()}  ({len(Q):,}일)")
    print("=" * W)
    print(f"  전체 누적: S4-2 {summary['total_return_pct_s42']:.2f}%  |  S5 {summary['total_return_pct_s5']:.2f}%")
    print(f"  QLD ATH 사이클 수: {summary['n_ath_cycles']}  (S5가 더 나은 사이클: {summary['cycles_s5_better']})")
    print(f"  사이클별 평균 (S5−S4-2): {summary['mean_cycle_diff_s5_minus_s42_pp']:+.2f} p.p.  "
          f"중앙값: {summary['median_cycle_diff_pp']:+.2f} p.p.")
    cr = summary.get("correlation_qld_mdd_vs_diff")
    if cr is not None:
        print(f"  corr(QLD 사이클 MDD, S5−S4-2 차이): {cr:.3f}  ({summary.get('corr_note', '')})")
    print(f"  S4-2 이벤트 에피소드 수: {summary['s42_episode_count']}")
    print("=" * W)
    print("  S4-2 레버 에피소드별 — 동일 구간 S5 수익률 (상위 12개 |edge| )")
    if not ep42.empty:
        ep2 = ep42.copy()
        ep2["abs_edge"] = ep2["edge_s5_minus_s42_pp"].abs()
        for _, r in ep2.nlargest(12, "abs_edge").iterrows():
            print(
                f"    ep{int(r['ep'])} {str(r['entry_dt'].date())}→{str(r['exit_dt'].date())} "
                f"{r['entry_type'][:16]:<16} S4-2 {r['s42_ep_ret']*100:+.2f}%  "
                f"S5동기 {r['s5_same_win_ret']:+.2f}%  Δ {r['edge_s5_minus_s42_pp']:+.2f} p.p."
            )
    print("=" * W)
    print("  QLD ATH 사이클 중 S5가 가장 앞선 5개 (p.p.)")
    for r in payload["best_cycles_for_s5"]:
        print(f"    cyc{r['cyc']} {r['start']}→{r['end']}  QLD MDD {r['qld_mdd']:.1f}%  "
              f"Δ {r['s5_minus_s42_pct']:+.2f} (S5 {r['s5_ret']:+.2f}% vs S4-2 {r['s42_ret']:+.2f}%)")
    print("  QLD ATH 사이클 중 S4-2가 가장 앞선 5개 (p.p.)")
    for r in payload["worst_cycles_for_s42_vs_s5"]:
        print(f"    cyc{r['cyc']} {r['start']}→{r['end']}  QLD MDD {r['qld_mdd']:.1f}%  "
              f"Δ {r['s5_minus_s42_pct']:+.2f} (S5 {r['s5_ret']:+.2f}% vs S4-2 {r['s42_ret']:+.2f}%)")
    print("=" * W)
    print(f"  저장: {csv_c}")
    if not ep42.empty:
        print(f"          {csv_e}")
    print(f"          {json_path}")


if __name__ == "__main__":
    main()
