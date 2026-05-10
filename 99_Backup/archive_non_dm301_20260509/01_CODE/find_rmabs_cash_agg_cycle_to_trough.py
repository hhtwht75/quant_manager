"""RMABS-4tier-Cash 전구간 NAV·레짐: AGG 퇴장→전역 최대 DD 트루→AGG 재진입 최단달력 요약."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import load_extended_daily  # noqa: E402
from fsm_four_asset_strategy import (  # noqa: E402
    CAPITAL_START,
    DEFAULT_TRAIL,
    MA_WINDOW,
    Regime,
    _align_five,
    get_or_build_ma200,
    run_fsm_backtest,
)
from rmabs_gold_simulation import load_gold_series  # noqa: E402


def _flat_cash(ix: pd.DatetimeIndex) -> pd.DataFrame:
    ix = ix.sort_values()
    one = pd.Series(1.0, index=ix, dtype=float)
    return pd.DataFrame({"Open": one, "High": one, "Low": one, "Close": one})


def main() -> None:
    root = pd.Timestamp("2002-10-01")
    sg = load_extended_daily("QQQ")
    ng = load_extended_daily("QLD")
    ag = load_extended_daily("TQQQ")
    ma_full, _ = get_or_build_ma200(sg, "QQQ", force_rebuild=False)
    ix0 = sg.index.intersection(ng.index).intersection(ag.index).sort_values()
    ix0 = ix0[ix0 >= root]
    gold_df, _ = load_gold_series(ix0[0], ix0[-1])
    ix = ix0.intersection(gold_df.index).sort_values()

    mav = ma_full.reindex(ix)
    sg_i, ng_i, ag_i = sg.reindex(ix), ng.reindex(ix), ag.reindex(ix)
    cash_i = _flat_cash(ix)
    aligned = _align_five(sg_i, cash_i, sg_i, ng_i, ag_i)

    nav, ev = run_fsm_backtest(
        aligned,
        mav,
        trail_stop=DEFAULT_TRAIL,
        initial_capital=CAPITAL_START,
        use_safe_ma_rule=True,
    )

    ev = ev.copy()
    ev["Date"] = pd.to_datetime(ev["Date"])
    nav_sr = pd.Series(nav.values, index=ev["Date"])

    cumax = nav_sr.cummax()
    dd = nav_sr / cumax - 1.0
    glob_mdd = float(dd.min())
    t_trough = dd.idxmin()

    exit_idx = []  # i: 전일 종가 후 AGG, 당일은 비AGG
    re_idx = []
    er = ev.reset_index(drop=True)
    for i in range(1, len(er)):
        if er.loc[i - 1, "regime_after"] == Regime.AGG.name and er.loc[i, "regime_after"] != Regime.AGG.name:
            exit_idx.append(i)
        if er.loc[i - 1, "regime_after"] != Regime.AGG.name and er.loc[i, "regime_after"] == Regime.AGG.name:
            re_idx.append(i)

    t_star = pd.Timestamp(t_trough)
    best_trad = None
    best_pack = None
    for ix_ex in exit_idx:
        de = pd.Timestamp(er.loc[ix_ex, "Date"])
        if de >= t_star:
            continue
        for ix_r in re_idx:
            if ix_r <= ix_ex:
                continue
            dr = pd.Timestamp(er.loc[ix_r, "Date"])
            if dr <= t_star:
                continue
            dwin = dd.loc[de:dr]
            if len(dwin) == 0:
                continue
            if abs(float(dwin.min()) - glob_mdd) > 1e-6:
                continue
            td = ix_r - ix_ex + 1
            cd = int((dr - de).days)
            pack = (
                td,
                cd,
                de.date(),
                dr.date(),
                str(er.loc[ix_ex, "transitions"]),
                str(er.loc[ix_r, "transitions"]),
                er.loc[ix_ex : ix_r]["regime_after"].iloc[:5].tolist(),
            )
            if best_trad is None or td < best_trad or (td == best_trad and cd < best_pack[1]):
                best_trad = td
                best_pack = pack

    peak_until_trough = cumax.loc[t_trough]
    last_peak_dates = cumax[cumax == peak_until_trough].index
    peak_start = last_peak_dates[last_peak_dates <= t_trough][0]

    print("glob_mdd", glob_mdd)
    print("t_trough", t_trough.date(), "peak_nav@", peak_start.date(), float(peak_until_trough))
    print(
        "min_cycle (AGG출발일=비AGG 시작일, AGG복귀):",
        "\n ",
        best_pack[:6] if best_pack else "(없음)",
    )
    if best_pack:
        print(" 거래일(출발~복귀):", best_pack[0], "달력일:", best_pack[1])

    seq = er.set_index("Date").loc[t_star :]
    rr = seq[seq["regime_after"].eq(Regime.AGG.name) & seq["regime_after"].shift(1).ne(Regime.AGG.name)].head(
        1
    )
    if len(rr):
        fd = rr.index[0].date()
        print(" 트루 이후 최초 AGG 재진입일(참조):", fd)


if __name__ == "__main__":
    main()
