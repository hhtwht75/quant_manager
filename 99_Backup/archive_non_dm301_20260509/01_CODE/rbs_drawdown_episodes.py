"""
전 구간 RMABS 백테스트 후, 순자산 기준 ATH 대비 최대 낙폭(MDD)이 큰 에피소드 상위 N개 추출·출력.

실행:
  python3 01_CODE/rbs_drawdown_episodes.py [--top N] [--root 2002-10-01]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import load_extended_daily, strategy_rsi_ma_based_switching  # noqa: E402

OUT_DIR = _ROOT / "03_RESULT"
CAP = 100_000.0


def ath_episodes(port: pd.Series) -> list[dict]:
    """연속 사이 새 ATH 발생 구간별로 그 전 ATH로부터 다음 ATH 직전까지의 최대 하락."""
    idx = port.index
    w = port.astype(float).values
    ath_i: list[int] = []
    cur_m = -np.inf
    for i, v in enumerate(w):
        if v > cur_m:
            cur_m = v
            ath_i.append(i)

    episodes: list[dict] = []
    for j in range(len(ath_i) - 1):
        p, pend = ath_i[j], ath_i[j + 1]
        peak_val = w[p]
        seg = w[p : pend + 1]
        if len(seg) < 2:
            continue
        tl = int(seg.argmin())
        t_i = p + tl
        trough_val = w[t_i]
        mdd = float(trough_val / peak_val - 1.0)
        episodes.append(
            {
                "peak_date": str(idx[p].date()),
                "trough_date": str(idx[t_i].date()),
                "recovery_to_new_ath_date": str(idx[pend].date()),
                "dd_days_peak_to_trough": int(t_i - p),
                "dd_days_trough_to_recovery": int(pend - t_i),
                "mdd_pct": mdd * 100.0,
                "peak_nav": float(peak_val),
                "trough_nav": float(trough_val),
                "recovery_nav": float(w[pend]),
            }
        )

    if ath_i:
        p = ath_i[-1]
        seg = w[p:]
        peak_val = w[p]
        tl = int(seg.argmin())
        t_i = p + tl
        trough_val = w[t_i]
        mdd = float(trough_val / peak_val - 1.0)
        last_dt = idx[-1]
        if mdd < -0.005 and trough_val < peak_val * 0.995:
            episodes.append(
                {
                    "peak_date": str(idx[p].date()),
                    "trough_date": str(idx[t_i].date()),
                    "recovery_to_new_ath_date": None,
                    "dd_days_peak_to_trough": int(t_i - p),
                    "dd_days_trough_to_recovery": None,
                    "mdd_pct": mdd * 100.0,
                    "peak_nav": float(peak_val),
                    "trough_nav": float(trough_val),
                    "recovery_nav": float(w[-1]),
                    "as_of_note": str(last_dt.date()),
                }
            )

    return episodes


EVT_KR = {
    "RMABS_INIT_RULE0_QLD": "규칙0 시작→QLD100%",
    "RMABS_INIT_RULE0_QQQ": "규칙0 시작→QQQ100%",
    "RMABS_TO_TQQQ": "RSI·MA충족→TQQQ",
    "RMABS_EXIT_BELOW_COST": "진입가↓→QLD",
    "RMABS_EXIT_TRAIL": "트레일-15%→QLD",
    "RMABS_QLD_TO_QQQ_MA_DOWN": "QLD중 QQQ≤MA×0.97→QQQ",
    "RMABS_QLD_TO_QQQ_MA_LE": "(구버전 이벤트명)",
}


def events_in_window(ev: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    if ev.empty:
        return ev
    d = ev.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    m = (d["Date"] >= t0) & (d["Date"] <= t1)
    return d.loc[m]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="2002-10-01")
    ap.add_argument("--top", type=int, default=2, help="손실 깊이 상위 N개 에피소드")
    args = ap.parse_args()

    qqq = load_extended_daily("QQQ")
    qld = load_extended_daily("QLD")
    tqqq = load_extended_daily("TQQQ")
    common = qqq.index.intersection(qld.index).intersection(tqqq.index)
    root_ts = pd.Timestamp(args.root.strip())
    common = common[common >= root_ts]
    Q, L, T = qqq.loc[common], qld.loc[common], tqqq.loc[common]

    port, ev = strategy_rsi_ma_based_switching(Q, L, T, CAP, series_name="RMABS")
    eps = ath_episodes(port)
    eps_sorted = sorted(eps, key=lambda x: x["mdd_pct"])[: args.top]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "sensitivity/rmabs_two_worst_drawdown_episodes.json"

    report: dict = {"full_range": [str(common[0].date()), str(common[-1].date())], "episodes": []}

    print("=" * 72)
    print(f"RMABS 전 구간 순자산 MDD 에피소드 (깊은 순 상위 {args.top}개)")
    print(f"  기간 {common[0].date()} ~ {common[-1].date()}  |  시작자본 ${CAP:,.0f}")
    print("=" * 72)

    full_m = float((port / port.cummax() - 1.0).min() * 100)
    print(f"전체 구간 순자산 max drawdown(일봉): {full_m:.2f}%")
    print()

    rank = 0
    for ep in eps_sorted:
        rank += 1
        t_peak = pd.Timestamp(ep["peak_date"])
        t_trough = pd.Timestamp(ep["trough_date"])
        sub = events_in_window(ev, t_peak, t_trough)
        lines = []
        for _, row in sub.iterrows():
            t = EVT_KR.get(str(row["type"]), str(row["type"]))
            lines.append(f"    • {pd.Timestamp(row['Date']).date()}  {t}")
        episode_dict = {
            "rank_by_depth": rank,
            **ep,
            "events_peak_to_trough": [
                {"date": str(pd.Timestamp(r["Date"]).date()), "type": str(r["type"]), "label_ko": EVT_KR.get(str(r["type"]), str(r["type"]))}
                for _, r in sub.iterrows()
            ],
        }
        report["episodes"].append(episode_dict)

        print(f"--- 에피소드 #{rank} (ATH 대비 최저점 {ep['mdd_pct']:.2f}%) ---")
        print(f"  고점일:      {ep['peak_date']}  NAV ${ep['peak_nav']:,.2f}")
        print(f"  최저점일:    {ep['trough_date']}  NAV ${ep['trough_nav']:,.2f}")
        print(f"  새 ATH 회복일: {ep.get('recovery_to_new_ath_date') or '(샘플 끝까지 미회복)'}")
        if ep.get("as_of_note"):
            print(f"  ※ 샘플 말일: {ep['as_of_note']} (아직 새 고점 회복 안 될 수 있음)")
        print(f"  피크→저점 거래일: {ep['dd_days_peak_to_trough']}일")
        print("  해당 구간(피크~저점) 거래 스위치·이벤트:")
        if sub.empty:
            print("    (없음 또는 저장된 이벤트만 해당)")
        else:
            print("\n".join(lines))
        print()

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("JSON:", out_json)


if __name__ == "__main__":
    main()
