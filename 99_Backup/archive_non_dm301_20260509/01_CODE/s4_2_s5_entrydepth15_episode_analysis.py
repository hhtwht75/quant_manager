"""
S4-2 vs S5(entry_depth_stop_mult=1.5) 에피소드 분석 + 기존 drop_deepen S5 대비 변화.

• S4-2 레버 에피소드(TO_L* → 청산)마다 동일 구간 수익률: S4-2, S5(1.5×), S5(구 drop_deepen)
• QLD ATH 사이클 피크→피크 구간 수익률 동일 비교
• JSON/CSV → 03_RESULT/sensitivity/

실행: python3 01_CODE/s4_2_s5_entrydepth15_episode_analysis.py
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


def period_return(port: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    s = port.index.get_indexer([start], method="ffill")[0]
    e = port.index.get_indexer([end], method="ffill")[0]
    return float(port.iloc[e] / port.iloc[s] - 1.0)


def ath_step_dates(close: pd.Series) -> pd.DatetimeIndex:
    rm = close.cummax().values
    inc = np.r_[True, rm[1:] > rm[:-1]]
    return close.index[inc]


def events_between(ev: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    if ev is None or ev.empty:
        return pd.DataFrame()
    return ev.loc[(ev["Date"] >= t0) & (ev["Date"] <= t1)].copy()


def summarize_s5_ev(sub: pd.DataFrame) -> str:
    if sub.empty:
        return "(이벤트 없음)"
    parts = []
    for _, r in sub.iterrows():
        t = r["type"]
        if t == "TO_ATTACK":
            parts.append(
                f"{r['Date'].date()} ATK d={r.get('drop_pct','?')}% f={r.get('atk_frac','?')}"
            )
        elif t == "STOP_LOSS":
            parts.append(f"{r['Date'].date()} SL {r.get('reason','')}")
        else:
            parts.append(f"{r['Date'].date()} {t}")
    return " | ".join(parts[:8]) + (" ..." if len(parts) > 8 else "")


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
    p5_old, ev5_old = strategy_s5(Q, L, T, CAP, stop_factor=0.75, **S5_ANCHOR)
    p5_new, ev5_new = strategy_s5(
        Q, L, T, CAP,
        stop_factor=0.75,
        entry_depth_stop_mult=1.5,
        **S5_ANCHOR,
    )

    ep = build_s42_episodes(ev42)
    rows = []
    for _, row in ep.iterrows():
        t0, t1 = row["entry_dt"], row["exit_dt"]
        r42 = float(row["s42_ep_ret"])
        ro = period_return(p5_old, t0, t1)
        rn = period_return(p5_new, t0, t1)
        rows.append({
            "ep": int(row["ep"]),
            "entry_dt": str(t0.date()),
            "exit_dt": str(t1.date()),
            "entry_type": row["entry_type"],
            "exit_type": row["exit_type"],
            "dur_days": int(row["dur_days"]),
            "s42_ret_pct": round(r42 * 100, 2),
            "s5_dropd_ret_pct": round(ro * 100, 2),
            "s5_d15_ret_pct": round(rn * 100, 2),
            "edge_old_pp": round((ro - r42) * 100, 2),
            "edge_new_pp": round((rn - r42) * 100, 2),
            "delta_new_minus_old_pp": round((rn - ro) * 100, 2),
            "s5_new_timeline": summarize_s5_ev(events_between(ev5_new, t0, t1)),
        })

    ep_df = pd.DataFrame(rows)
    n_ep = len(ep_df)
    s42_wins_new = (ep_df["edge_new_pp"] < -0.05).sum()
    s5_new_wins = (ep_df["edge_new_pp"] > 0.05).sum()
    better_vs_old = (ep_df["delta_new_minus_old_pp"] > 0.05).sum()
    worse_vs_old = (ep_df["delta_new_minus_old_pp"] < -0.05).sum()

    sl_old = int((ev5_old["type"] == "STOP_LOSS").sum())
    sl_new = int((ev5_new["type"] == "STOP_LOSS").sum())

    peaks = ath_step_dates(qld_c)
    cyc_rows = []
    for i in range(len(peaks) - 1):
        s, e = peaks[i], peaks[i + 1]
        r42 = period_return(p42, s, e)
        ro = period_return(p5_old, s, e)
        rn = period_return(p5_new, s, e)
        cyc_rows.append({
            "cyc": i + 1,
            "start": str(s.date()),
            "end": str(e.date()),
            "s42_ret_pct": round(r42 * 100, 2),
            "s5_old_ret_pct": round(ro * 100, 2),
            "s5_new_ret_pct": round(rn * 100, 2),
            "edge_new_vs_s42_pp": round((rn - r42) * 100, 2),
            "delta_new_vs_old_pp": round((rn - ro) * 100, 2),
        })
    cyc_df = pd.DataFrame(cyc_rows)

    tot42 = float(p42.iloc[-1] / p42.iloc[0] - 1)
    toto = float(p5_old.iloc[-1] / p5_old.iloc[0] - 1)
    totn = float(p5_new.iloc[-1] / p5_new.iloc[0] - 1)

    tag = f"{Q.index[0].strftime('%Y%m%d')}_{Q.index[-1].strftime('%Y%m%d')}"
    payload = {
        "period": tag,
        "totals_pct": {
            "s42": round(tot42 * 100, 2),
            "s5_drop_deepen": round(toto * 100, 2),
            "s5_entry_depth_1_5": round(totn * 100, 2),
        },
        "s4_2_episodes": {
            "n": int(n_ep),
            "s42_better_than_s5_new": int(s42_wins_new),
            "s5_new_better_than_s42": int(s5_new_wins),
            "episodes_new_better_than_old": int(better_vs_old),
            "episodes_new_worse_than_old": int(worse_vs_old),
        },
        "stop_loss_counts": {"s5_drop_deepen": sl_old, "s5_entry_depth_1_5": sl_new},
        "interpretation": [
            "1.5× 손절은 스윙 ‘조금 더 깎임’보다 **ATH dd가 진입 스윙의 1.5배까지** 가야 발동해, "
            "drop_deepen 대비 **같은 S4-2 창에서 레버 유지가 길어질 수 있음**.",
            "창 수익은 그 창 안 반등·재진입 패턴에 민감: 새 규칙이 유리한 에피소드는 ‘조기 손절만 손해’였던 곳, "
            "불리한 곳은 ‘더 깊게 물린 뒤’ 청산이 늦어진 경우로 나뉨.",
            "전체 누적(이 표본)에서는 새 S5가 구 S5보다 크게 낮음 — 에피소드 일부 개선·악화가 상쇄되지 않고 "
            "긴 구간·고레버 MDD에서 손실이 커진 결과와 일치.",
        ],
        "episodes_largest_new_improvement_vs_old": ep_df.nlargest(5, "delta_new_minus_old_pp").to_dict(
            orient="records"
        ),
        "episodes_largest_new_damage_vs_old": ep_df.nsmallest(5, "delta_new_minus_old_pp").to_dict(
            orient="records"
        ),
        "episodes_s42_crushes_new_s5": ep_df.nsmallest(5, "edge_new_pp").to_dict(orient="records"),
        "cycles_largest_new_damage": cyc_df.nsmallest(5, "delta_new_vs_old_pp").to_dict(orient="records"),
    }

    out_json = OUT_DIR / f"s4_2_s5_entrydepth15_analysis_{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    ep_df.to_csv(OUT_DIR / f"s4_2_s5_entrydepth15_episodes_{tag}.csv", index=False, encoding="utf-8-sig")
    cyc_df.to_csv(OUT_DIR / f"s4_2_s5_entrydepth15_cycles_{tag}.csv", index=False, encoding="utf-8-sig")

    W = 102
    print("=" * W)
    print(f"  S4-2 vs S5(진입스윙×1.5 ATH dd 손절)  |  {Q.index[0].date()} ~ {Q.index[-1].date()}")
    print("=" * W)
    print(f"  전체 누적%  S4-2 {payload['totals_pct']['s42']:.2f}  "
          f"S5 drop_deepen {payload['totals_pct']['s5_drop_deepen']:.2f}  "
          f"S5 1.5× {payload['totals_pct']['s5_entry_depth_1_5']:.2f}")
    print(f"  S4-2 레버 에피소드 {n_ep}건: S4-2가 새 S5보다 유리 {s42_wins_new}건, "
          f"새 S5가 유리 {s5_new_wins}건  |  "
          f"구 대비 새가 나은 창 {better_vs_old}건, 구가 나은 창 {worse_vs_old}건")
    print(f"  STOP_LOSS  drop_deepen {sl_old}회 →  entry_depth 1.5× {sl_new}회")
    print("-" * W)
    print("  【S4-2 에피소드】 새 S5가 구 S5보다 **많이** 나아진 3건 (delta_new_minus_old_pp)")
    for r in ep_df.nlargest(3, "delta_new_minus_old_pp").itertuples(index=False):
        print(f"    ep{r.ep} {r.entry_dt}→{r.exit_dt}  S4-2 {r.s42_ret_pct:+.2f}%  "
              f"구S5 {r.s5_dropd_ret_pct:+.2f}%  새S5 {r.s5_d15_ret_pct:+.2f}%  "
              f"Δ구간(새-구) {r.delta_new_minus_old_pp:+.2f}p.p.")
        print(f"      {r.s5_new_timeline}")
    print("-" * W)
    print("  【S4-2 에피소드】 새 S5가 구 S5보다 **많이** 나빠진 3건")
    for r in ep_df.nsmallest(3, "delta_new_minus_old_pp").itertuples(index=False):
        print(f"    ep{r.ep} {r.entry_dt}→{r.exit_dt}  S4-2 {r.s42_ret_pct:+.2f}%  "
              f"구S5 {r.s5_dropd_ret_pct:+.2f}%  새S5 {r.s5_d15_ret_pct:+.2f}%  "
              f"Δ구간(새-구) {r.delta_new_minus_old_pp:+.2f}p.p.")
        print(f"      {r.s5_new_timeline}")
    print("-" * W)
    print("  【QLD ATH 사이클】 새 S5가 구 S5보다 손해 큰 사이클 3개 (전구간 수익 차)")
    for r in cyc_df.nsmallest(3, "delta_new_vs_old_pp").itertuples(index=False):
        print(f"    cyc{r.cyc} {r.start}→{r.end}  구S5 {r.s5_old_ret_pct:+.2f}%  새S5 {r.s5_new_ret_pct:+.2f}%  "
              f"Δ(새-구) {r.delta_new_vs_old_pp:+.2f}p.p.  (vs S4-2 {r.edge_new_vs_s42_pp:+.2f}p.p.)")
    print("=" * W)
    print(f"  JSON {out_json}")
    print("=" * W)


if __name__ == "__main__":
    main()
