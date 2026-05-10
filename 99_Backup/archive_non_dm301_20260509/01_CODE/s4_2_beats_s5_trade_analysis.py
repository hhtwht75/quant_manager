"""
S4-2가 S5 앵커보다 유리했던 레버 구간(한 번의 매수~매도) 분석.

• 에피소드: S4-2 이벤트 TO_L* → TRAIL*/L1_STOP/L2_STOP (같은 정의 as tiered main)
• 각 에피소드에서 동일 캘린더 구간 S5 수익률 비교 후, S4-2가 앞선 건만 상세 기록.
• 해당 구간 S4-2/S5 **전체 이벤트** 열거 + QLD ATH 대비 dd / 최소 dd + 규칙 기반 원인 태그.

실행: python3 01_CODE/s4_2_beats_s5_trade_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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


def events_between(ev: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    if ev is None or ev.empty:
        return pd.DataFrame()
    m = (ev["Date"] >= t0) & (ev["Date"] <= t1)
    return ev.loc[m].copy()


def format_ev_row(r: pd.Series) -> str:
    t = r["type"]
    extra = []
    if t == "TO_ATTACK":
        for k in ("drop_pct", "rebound_pct", "atk_frac", "sizing"):
            if k in r and pd.notna(r[k]):
                extra.append(f"{k}={r[k]}")
    if t == "STOP_LOSS" and "reason" in r and pd.notna(r["reason"]):
        extra.append(f"reason={r['reason']}")
    tail = ("  " + " ".join(extra)) if extra else ""
    return f"{str(r['Date'].date())}  {t}{tail}"


def qld_context(qld_close: pd.Series, t0: pd.Timestamp, t1: pd.Timestamp) -> dict:
    sub = qld_close.loc[t0:t1]
    if sub.empty:
        return {}
    pre = qld_close.loc[:t0]
    ath0 = float(pre.max()) if len(pre) else float(sub.iloc[0])
    q0 = float(qld_close.loc[t0])
    dd0 = q0 / ath0 - 1.0
    dd_series = sub / sub.cummax() - 1.0
    dd_min = float(dd_series.min())
    return {
        "dd_at_s42_entry_pct": round(dd0 * 100, 2),
        "min_dd_in_window_pct": round(dd_min * 100, 2),
    }


def tag_reason(s5_sub: pd.DataFrame, s42_ret: float, s5_ret: float) -> list[str]:
    tags = []
    n_atk = len(s5_sub[s5_sub["type"] == "TO_ATTACK"])
    n_sl = len(s5_sub[s5_sub["type"] == "STOP_LOSS"])
    n_tr = len(s5_sub[s5_sub["type"] == "TO_TRAILING"])
    n_tex = len(s5_sub[s5_sub["type"].isin(["TRAIL_EXIT", "TRAIL_FLOOR"])])

    if n_atk == 0:
        if n_sl > 0:
            tags.append("창내TO_ATTACK없음_창밖ATTACK이후손절가능")
        else:
            tags.append("S5_해당구간레버이벤트없음")
    if n_sl > 0:
        reasons = s5_sub.loc[s5_sub["type"] == "STOP_LOSS", "reason"].dropna().tolist()
        tags.append(f"S5_STOP_LOSS×{n_sl}" + (f"({reasons})" if reasons else ""))
    if n_atk >= 2:
        tags.append("S5_다회진입")
    if n_atk == 1 and n_sl == 0 and n_tr == 0:
        tags.append("S5_단일ATTACK후ATH미도달로청산없음")
    if s5_ret < 0 < s42_ret:
        tags.append("S5구간손실_S4-2이익")
    elif s42_ret > s5_ret + 0.02:
        tags.append("S4-2초과수익큼")
    if not tags:
        tags.append("구조혼합_이벤트대조필요")
    return tags


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

    ep = build_s42_episodes(ev42)
    records = []
    for _, row in ep.iterrows():
        t0, t1 = row["entry_dt"], row["exit_dt"]
        s5r = period_return(p5, t0, t1)
        s42r = float(row["s42_ep_ret"])
        edge_pp = (s5r - s42r) * 100
        if edge_pp >= -0.05:
            continue

        s42_ev = events_between(ev42, t0, t1)
        s5_ev = events_between(ev5, t0, t1)
        ctx = qld_context(qld_c, t0, t1)

        s42_lines = [format_ev_row(s42_ev.loc[i]) for i in s42_ev.index]
        s5_lines = [format_ev_row(s5_ev.loc[i]) for i in s5_ev.index]

        rec = {
            "ep": int(row["ep"]),
            "entry_dt": str(t0.date()),
            "exit_dt": str(t1.date()),
            "entry_type": row["entry_type"],
            "exit_type": row["exit_type"],
            "dur_days": int(row["dur_days"]),
            "s42_ret_pct": round(s42r * 100, 2),
            "s5_same_win_ret_pct": round(s5r * 100, 2),
            "s5_minus_s42_pp": round(edge_pp, 2),
            **ctx,
            "tags": tag_reason(s5_ev, s42r, s5r),
            "s42_events": s42_lines,
            "s5_events": s5_lines,
        }
        records.append(rec)

    records.sort(key=lambda x: x["s5_minus_s42_pp"])
    tag = f"{Q.index[0].strftime('%Y%m%d')}_{Q.index[-1].strftime('%Y%m%d')}"
    out_json = OUT_DIR / f"s4_2_beats_s5_trades_{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_cases": len(records),
                "note": "S4-2 레버 에피소드 중 동일 구간 S5 수익이 더 나쁜 경우(edge S5−S4-2 < 0, 약간의 반올림 허용 -0.05pp)",
                "cases": records,
                "pattern_summary": _pattern_summary(records),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 평탄 CSV (이벤트는 생략)
    flat = []
    for r in records:
        x = {k: v for k, v in r.items() if k not in ("s42_events", "s5_events")}
        x["tags"] = ";".join(r["tags"])
        flat.append(x)
    pd.DataFrame(flat).to_csv(
        OUT_DIR / f"s4_2_beats_s5_trades_{tag}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    W = 100
    print("=" * W)
    print(f"  S4-2 > S5 앵커 (동일 창)  —  {len(records)}건  |  {Q.index[0].date()} ~ {Q.index[-1].date()}")
    print("=" * W)
    for x in _pattern_summary(records):
        print(f"  • {x}")
    print("-" * W)
    for r in records[:15]:
        print(
            f"  ep{r['ep']} {r['entry_dt']} → {r['exit_dt']}  "
            f"S4-2 {r['s42_ret_pct']:+.2f}%  S5 {r['s5_same_win_ret_pct']:+.2f}%  "
            f"Δ{r['s5_minus_s42_pp']:+.1f}pp  |  {r['entry_type'][:18]} → {r['exit_type']}"
        )
        print(f"      QLD dd₀={r.get('dd_at_s42_entry_pct', '?')}%  창내최소≈{r.get('min_dd_in_window_pct', '?')}%")
        print(f"      태그: {'; '.join(r['tags'])}")
        print("      S4-2:" + "".join(f"\n        {ln}" for ln in r["s42_events"][:12]))
        if len(r["s42_events"]) > 12:
            print(f"        ... +{len(r['s42_events'])-12}행")
        print("      S5:")
        for ln in r["s5_events"][:14]:
            print(f"        {ln}")
        if len(r["s5_events"]) > 14:
            print(f"        ... +{len(r['s5_events'])-14}행")
        print("-" * W)
    print(f"  JSON: {out_json}")
    print("=" * W)


def _pattern_summary(records: list[dict]) -> list[str]:
    if not records:
        return ["(해당 없음)"]
    no_atk = sum(1 for r in records if any("창내TO_ATTACK없음" in t for t in r["tags"]))
    sl = sum(1 for r in records if any("STOP_LOSS" in t for t in r["tags"]))
    lines = [
        f"총 {len(records)}건 중 창 안에 TO_ATTACK 기록이 없는 경우: {no_atk}건 "
        f"(S5는 그 전에 ATTACK에 들어갔다가 창 안에서만 손절된 케이스 포함)",
        f"STOP_LOSS 포함: {sl}건 — exp 앵커는 스윙 저점이 진입보다 깎이면 청산; "
        f"S4-2는 티어/트레일·스탑 타이밍이 달라 같은 조정에서 더 오래 버티거나 빨리 QQQ로 갈 수 있음.",
        "L3 등 장기 보유 후 TRAIL 청산 구간: S4-2는 단계적 레버·플래그로 이미 노출이 정교; "
        "S5는 진입 시점 한 번의 frac으로 고정되어 조정 중 스윙이 한 번 더 깎이면 STOP_LOSS로 탈락하기 쉬움.",
    ]
    return lines


if __name__ == "__main__":
    main()
