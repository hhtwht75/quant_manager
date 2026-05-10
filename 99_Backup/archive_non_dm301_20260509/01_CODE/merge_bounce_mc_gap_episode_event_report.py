#!/usr/bin/env python3
"""
sticky_defense 대비 merge_bounce_simple_mc MC Sortino가 가장 많이 깨졌던(= sticky 우세 차가 최대) 창을 찾고,
그 무작위 3년 구간 안에서 둘 중 하나라도 전이 이벤트가 있었던 영업일·보유비중표를 출력한다.

  python3 01_CODE/merge_bounce_mc_gap_episode_event_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from evaluation_metrics import full_metrics  # noqa: E402

import merge_bounce_simple_mc as mbase  # noqa: E402
import merge_bounce_simple_mc_sticky_defense as mstick  # noqa: E402

IDLE = mbase.IDLE
BOUNCE = mbase.BOUNCE
DEFENSE = mbase.DEFENSE
SAFE = mbase.SAFE
SAFE_BOUNCE = mbase.SAFE_BOUNCE
_REG_NAMES = ("IDLE", "BOUNCE", "DEFENSE", "SAFE", "SAFE_BOUNCE")


def _weights(
    r: int, units: float, u_safe: float, u_sb: float, sfx: float, bv: float, dv: float, nav: float
) -> tuple[float, float, float, float]:
    """안전·반등·방어·현금(나머지) 비중, EOD 레짐 기준."""
    if nav <= 1e-15:
        return (float("nan"),) * 4
    if r == IDLE:
        return 0.0, 0.0, 0.0, 1.0
    if r == DEFENSE:
        return 0.0, 0.0, (units * dv) / nav, 0.0
    if r == BOUNCE:
        return 0.0, (units * bv) / nav, 0.0, 0.0
    ws = u_safe * sfx / nav
    wb = u_sb * bv / nav
    wc = max(0.0, 1.0 - ws - wb)
    return ws, wb, 0.0, wc


def run_fsm_logged(
    *,
    sticky_defense: bool,
    sig: pd.Series,
    safe: pd.Series,
    bounce: pd.Series,
    defense: pd.Series,
    ma: pd.Series,
    trail: float,
    capital: float,
) -> pd.DataFrame:
    ix = sig.index
    n = len(ix)
    sgv = sig.astype(float).to_numpy()
    m5a = pd.Series(sgv, index=ix).rolling(mbase.MA_FAST, min_periods=mbase.MA_FAST).mean().to_numpy()
    m12a = pd.Series(sgv, index=ix).rolling(mbase.MA_SLOW, min_periods=mbase.MA_SLOW).mean().to_numpy()
    mav = ma.astype(float).to_numpy()

    r = IDLE
    units = float(capital)
    u_safe = u_sb = 0.0
    agg_i = agg_h = 0.0
    prev_sv = prev_mah = float("nan")
    prev_ma_ok = False
    prev_m5 = prev_m120 = float("nan")

    rows: list[dict[str, object]] = []

    for k in range(n):
        sv = float(sgv[k])
        sfx = float(safe.iloc[k])
        bv = float(bounce.iloc[k])
        dv = float(defense.iloc[k])
        mah = float(mav[k])
        ma_ok = not np.isnan(mah)
        m5 = float(m5a[k])
        m12 = float(m12a[k])
        ts = ix[k]

        nav_pre = mbase._nav(r, units, u_safe, u_sb, sfx, bv, dv)
        trans: list[str] = []

        cross_dn = (
            ma_ok
            and prev_ma_ok
            and sv < mbase.MULT_DN * mah
            and prev_sv >= mbase.MULT_DN * prev_mah
        )

        if cross_dn:
            trans.append("RULE4_TO_SAFE")
            agg_i = agg_h = 0.0
            if sfx <= 0:
                raise ValueError("규칙4: safe<=0")
            r = SAFE
            u_safe = nav_pre / sfx
            u_sb = 0.0
            units = 0.0
        elif r == BOUNCE and agg_i > 1e-12:
            agg_h = max(agg_h, bv)
            if bv < agg_i - 1e-12:
                trans.append("BOUNCE_LT_INIT_DEFENSE")
                r = DEFENSE
                units = nav_pre / dv
                u_safe = u_sb = 0.0
                agg_i = agg_h = 0.0
            elif agg_h > 1e-12 and bv < agg_h * trail - 1e-12:
                trans.append("BOUNCE_TRAIL_DEFENSE")
                r = DEFENSE
                units = nav_pre / dv
                u_safe = u_sb = 0.0
                agg_i = agg_h = 0.0
        elif not sticky_defense and r == DEFENSE:
            ma_dual = mbase._ma_dual_ok(m5, m12, prev_m5, prev_m120)
            sig_break = mbase._cross_up_103_ma(sv, mah, ma_ok, prev_sv, prev_mah, prev_ma_ok)
            if sig_break and bv <= 0:
                raise ValueError("DEFENSE→BOUNCE: bounce<=0")
            if sig_break:
                trans.append("DEFENSE_SIG_TO_BOUNCE")
            elif ma_dual and bv > 0 and units > 1e-12:
                trans.append("DEFENSE_MA5_TO_BOUNCE")
            if sig_break or (ma_dual and bv > 0 and units > 1e-12):
                r = BOUNCE
                units = nav_pre / bv
                agg_i = agg_h = bv
                u_safe = u_sb = 0.0
        elif r in (SAFE, SAFE_BOUNCE):
            if mbase._cross_up_103_ma(sv, mah, ma_ok, prev_sv, prev_mah, prev_ma_ok):
                trans.append("SAFE_SIG_TO_BOUNCE")
                if bv <= 0:
                    raise ValueError("SAFE→BOUNCE: bounce<=0")
                r = BOUNCE
                units = nav_pre / bv
                agg_i = agg_h = bv
                u_safe = u_sb = 0.0
            else:
                ma_dual = mbase._ma_dual_ok(m5, m12, prev_m5, prev_m120)
                if ma_dual and u_safe > 1e-12 and bv > 0:
                    trans.append("SAFE_MA5_TO_BOUNCE_FULL")
                    u_sb = nav_pre / bv
                    u_safe = 0.0
                r = SAFE_BOUNCE if u_sb > 1e-12 else SAFE
        elif r == BOUNCE and mbase._cross_up_103_ma(sv, mah, ma_ok, prev_sv, prev_mah, prev_ma_ok):
            trans.append("BOUNCE_RESEED_103MA")
            if bv <= 0:
                raise ValueError("reseed: bounce<=0")
            agg_i = agg_h = bv
        elif r == IDLE:
            if ma_ok:
                trans.append("IDLE_TO_BOUNCE")
                if bv <= 0:
                    raise ValueError("IDLE→BOUNCE: bounce<=0")
                r = BOUNCE
                units = nav_pre / bv
                u_safe = u_sb = 0.0
                agg_i = agg_h = bv

        nav_eod = mbase._nav(r, units, u_safe, u_sb, sfx, bv, dv)
        ws, wb, wd, wc = _weights(r, units, u_safe, u_sb, sfx, bv, dv, nav_eod)
        rows.append(
            {
                "Date": ts,
                "events": "; ".join(trans) if trans else "",
                "regime": _REG_NAMES[r],
                "nav": nav_eod,
                "w_safe": ws,
                "w_bounce": wb,
                "w_defense": wd,
                "w_cash": wc,
            }
        )

        prev_sv, prev_mah, prev_ma_ok = sv, mah, ma_ok
        prev_m5, prev_m120 = m5, m12

    return pd.DataFrame(rows)


def main() -> None:
    root_ts = pd.Timestamp("1999-03-10")
    mc_years = 3.0
    mc_iters = 3000
    mc_seed = 42
    trail = mbase.TRAIL_DEFAULT
    capital = mbase.CAP0
    bt, dt, st = "TQQQ", "QQQ", "Gold"

    sg = mbase.load_extended_daily("QQQ")
    ql = mbase.load_extended_daily("QLD")
    tg = mbase.load_extended_daily("TQQQ")
    ix0 = sg.index.intersection(ql.index).intersection(tg.index).sort_values()
    ix0 = ix0[ix0 >= root_ts]
    gold_df, gold_note = mbase.load_gold_series(ix0[0], ix0[-1])
    ix = ix0.intersection(gold_df.index).sort_values()

    sig_s = sg["Close"].astype(float).reindex(ix)
    mav = mbase.ma200(sig_s).reindex(ix)
    lo = mbase.mc_min_days(mc_years)
    n = len(ix)
    rng = np.random.default_rng(mc_seed)

    windows: list[tuple[int, int]] = []
    sort_base: list[float] = []
    sort_sticky: list[float] = []

    nb = mbase.run_merge_fsm(
        sig_s,
        mbase.pick(st, sg.reindex(ix), ql.reindex(ix), tg.reindex(ix), gold_df.reindex(ix)),
        mbase.pick(bt, sg.reindex(ix), ql.reindex(ix), tg.reindex(ix), gold_df.reindex(ix)),
        mbase.pick(dt, sg.reindex(ix), ql.reindex(ix), tg.reindex(ix), gold_df.reindex(ix)),
        mav,
        trail=trail,
        capital=capital,
    )
    ns = mstick.run_merge_fsm(
        sig_s,
        mbase.pick(st, sg.reindex(ix), ql.reindex(ix), tg.reindex(ix), gold_df.reindex(ix)),
        mbase.pick(bt, sg.reindex(ix), ql.reindex(ix), tg.reindex(ix), gold_df.reindex(ix)),
        mbase.pick(dt, sg.reindex(ix), ql.reindex(ix), tg.reindex(ix), gold_df.reindex(ix)),
        mav,
        trail=trail,
        capital=capital,
    )

    best_i = -1
    best_gap = -np.inf

    for i in range(mc_iters):
        s0 = int(rng.integers(0, n - lo + 1))
        win = int(rng.integers(lo, (n - s0) + 1))
        windows.append((s0, win))
        sl = slice(s0, s0 + win)
        sb = float(full_metrics(nb.iloc[sl])["sortino"])
        ss = float(full_metrics(ns.iloc[sl])["sortino"])
        sort_base.append(sb)
        sort_sticky.append(ss)
        if ss > sb:
            gap = ss - sb
            if gap > best_gap:
                best_gap = gap
                best_i = i

    if best_i < 0:
        raise SystemExit("sticky 우세 에피소드가 한 건도 없음.")

    s0, win = windows[best_i]
    ix_win = ix[s0 : s0 + win]

    log_b = run_fsm_logged(
        sticky_defense=False,
        sig=sig_s,
        safe=mbase.pick(st, sg.reindex(ix), ql.reindex(ix), tg.reindex(ix), gold_df.reindex(ix)),
        bounce=mbase.pick(bt, sg.reindex(ix), ql.reindex(ix), tg.reindex(ix), gold_df.reindex(ix)),
        defense=mbase.pick(dt, sg.reindex(ix), ql.reindex(ix), tg.reindex(ix), gold_df.reindex(ix)),
        ma=mav,
        trail=trail,
        capital=capital,
    )
    log_s = run_fsm_logged(
        sticky_defense=True,
        sig=sig_s,
        safe=mbase.pick(st, sg.reindex(ix), ql.reindex(ix), tg.reindex(ix), gold_df.reindex(ix)),
        bounce=mbase.pick(bt, sg.reindex(ix), ql.reindex(ix), tg.reindex(ix), gold_df.reindex(ix)),
        defense=mbase.pick(dt, sg.reindex(ix), ql.reindex(ix), tg.reindex(ix), gold_df.reindex(ix)),
        ma=mav,
        trail=trail,
        capital=capital,
    )

    lb = log_b[log_b["Date"].isin(ix_win)].copy()
    ls = log_s[log_s["Date"].isin(ix_win)].copy()
    lb = lb.rename(
        columns={
            "events": "events_base",
            "regime": "regime_base",
            "w_safe": "w_safe_base",
            "w_bounce": "w_bounce_base",
            "w_defense": "w_defense_base",
            "w_cash": "w_cash_base",
        }
    )
    ls = ls.rename(
        columns={
            "events": "events_sticky",
            "regime": "regime_sticky",
            "w_safe": "w_safe_sticky",
            "w_bounce": "w_bounce_sticky",
            "w_defense": "w_defense_sticky",
            "w_cash": "w_cash_sticky",
        }
    )
    merged = lb.merge(ls.drop(columns=["nav"]), on="Date", how="inner").sort_values("Date")
    evb = merged["events_base"].fillna("") != ""
    evs = merged["events_sticky"].fillna("") != ""
    merged = merged[evb | evs]

    out_md = (
        _ROOT
        / "03_RESULT"
        / "sensitivity"
        / f"mc_gap_episode_events_i{best_i}_s{s0}_w{win}_seed{mc_seed}_table.md"
    )
    out_csv = out_md.with_suffix(".csv")

    header = (
        f"# 최대 Sortino 격차 에피소드 (sticky − base 우세 중 최대)\n\n"
        f"- MC repetition index: **{best_i + 1}** / {mc_iters} (0-based `{best_i}`)\n"
        f"- 무작위 창: **`{ix_win[0].date()}` ~ `{ix_win[-1].date()}`** ({win}일)\n"
        f"- 금 데이터: {gold_note}\n"
        f"- base Sortino window: **{sort_base[best_i]:.4f}**\n"
        f"- sticky Sortino window: **{sort_sticky[best_i]:.4f}**\n"
        f"- 격차 (sticky − base): **{best_gap:.4f}**\n"
        f"- 비중 안전=`{st}` 반등=`{bt}` 방어=`{dt}` · 시그널=QQQ\n\n"
    )

    tbl = merged.copy()
    for c in tbl.columns:
        if c.startswith("w_"):
            tbl[c] = tbl[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    tbl["events_base"] = tbl["events_base"].fillna("")
    tbl["events_sticky"] = tbl["events_sticky"].fillna("")

    lines = ["| 날짜 | base 이벤트 | sticky 이벤트 | base 레짐 | sticky 레짐 | base w_safe/w_bnce/w_def/cash | sticky w_* |"]
    lines.append("|------|-------------|---------------|-----------|---------------|-------------------------------|-------------|")

    def pack_w(row: pd.Series, side: str) -> str:
        return (
            f"{row[f'w_safe_{side}']}/{row[f'w_bounce_{side}']}/"
            f"{row[f'w_defense_{side}']}/{row[f'w_cash_{side}']}"
        )

    for _, row in tbl.iterrows():
        lines.append(
            "| {d} | {eb} | {es} | {rb} | {rs} | {wb} | {ws} |".format(
                d=row["Date"].date(),
                eb=row["events_base"].replace("|", "\\|"),
                es=row["events_sticky"].replace("|", "\\|"),
                rb=row["regime_base"],
                rs=row["regime_sticky"],
                wb=pack_w(row, "base"),
                ws=pack_w(row, "sticky"),
            )
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    merged.to_csv(out_csv, index=False)

    print(header)
    print(f"CSV: {out_csv.resolve()}")
    print(f"MD:  {out_md.resolve()}")


if __name__ == "__main__":
    main()
