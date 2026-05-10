#!/usr/bin/env python3
"""에피소드1: 시작총액 동일(첫 거래일 NAV 스케일) 재현 후 이벤트일별 보유 금액·비중 MD."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import load_extended_daily  # noqa: E402
from rmabs_gold_simulation import load_gold_series  # noqa: E402
from fsm_four_asset_strategy import (
    CAPITAL_START,
    DEFAULT_TRAIL,
    Regime,
    get_or_build_ma200,
    run_fsm_backtest,
    _align_four,
    _align_five,
)

ROOT_BT = pd.Timestamp("2002-10-01")
EP0 = pd.Timestamp("2006-06-15")
EP1 = pd.Timestamp("2009-03-06")
NAV0 = 100_000.0
OUT_MD = _ROOT / "03_RESULT/sensitivity/episode1_holdings_RM_vs_MABS_Gold_plain_ko.md"


def tx_lab(s: object) -> str:
    if pd.isna(s):
        return "—"
    t = str(s).strip()
    return "—" if not t else t


def pct(v: float, tot: float) -> str:
    if abs(tot) < 1e-12:
        return "—"
    return "{:.2f}%".format(100.0 * max(v, 0.0) / tot)


def holding_bucket(
    nav: float,
    rg: Regime,
    u_sf: float,
    u_dx: float,
    px_au: float,
    px_df: float,
    *,
    rmabs_four: bool,
) -> dict[str, float]:
    cash = au = qqq = qld = tqq = 0.0
    if rg == Regime.IDLE:
        cash = nav
    elif rg == Regime.BOUNCE:
        if rmabs_four:
            qqq = nav
        else:
            au = nav
    elif rg == Regime.DEFENSE:
        qld = nav
    elif rg == Regime.AGG:
        tqq = nav
    elif rg == Regime.SAFE:
        au = u_sf * px_au
    elif rg == Regime.SAFE_BOUNCE:
        au = u_sf * px_au
        qqq = u_dx * px_df
    labs = [
        ("현금(아직 종목 미보유 상태)", cash),
        ("금(안전종목·종가 레그 평가)", au),
        ("QQQ 종가 레그 분(반등·시그널)", qqq),
        ("QLD 종가 레그 분", qld),
        ("TQQQ 종가 레그 분", tqq),
    ]
    vals = dict(labs)
    gap = nav - sum(vals.values())
    if abs(gap) > max(10.0, 1e-5 * nav):
        vals["(숫자 맞춤용 잔차)"] = gap
    return vals


def event_dates(ep: pd.DataFrame) -> list[pd.Timestamp]:
    ts = pd.to_datetime(
        ep.loc[ep["transitions"].astype(str).str.strip().str.len() > 0, "Date"], errors="coerce"
    ).dt.normalize()
    return sorted(set(ts.dropna()))


def last_row(ev: pd.DataFrame, d: pd.Timestamp) -> pd.Series:
    return ev.loc[ev["Date"] == d].iloc[-1]


def main() -> None:
    q = load_extended_daily("QQQ")
    l = load_extended_daily("QLD")
    t = load_extended_daily("TQQQ")
    ma_f, _ = get_or_build_ma200(q, "QQQ")
    ix0 = q.index.intersection(l.index).intersection(t.index).sort_values()
    ix0 = ix0[ix0 >= ROOT_BT]
    gold_df, gnote = load_gold_series(ix0[0], ix0[-1])
    ix = ix0.intersection(gold_df.index).sort_values()

    qi, lj, tk = q.reindex(ix), l.reindex(ix), t.reindex(ix)
    gx = gold_df.reindex(ix)
    ma = ma_f.reindex(ix)
    alt = _align_five(qi, gx, qi, lj, tk)
    alm = _align_four(qi, gx, lj, tk)

    nav_t, evt = run_fsm_backtest(
        alt, ma, trail_stop=DEFAULT_TRAIL, initial_capital=CAPITAL_START, use_safe_ma_rule=True
    )
    nav_m, evm = run_fsm_backtest(
        alm, ma, trail_stop=DEFAULT_TRAIL, initial_capital=CAPITAL_START, use_safe_ma_rule=False
    )

    evt = evt.assign(Date=pd.to_datetime(evt["Date"]).dt.normalize())
    evm = evm.assign(Date=pd.to_datetime(evm["Date"]).dt.normalize())

    i0 = int(ix.get_indexer_for([pd.Timestamp(EP0)])[0])
    st = NAV0 / float(nav_t.iloc[i0])
    sm = NAV0 / float(nav_m.iloc[i0])

    et = evt[(evt["Date"] >= EP0.normalize()) & (evt["Date"] <= EP1.normalize())]
    em = evm[(evm["Date"] >= EP0.normalize()) & (evm["Date"] <= EP1.normalize())]
    days = sorted(set(event_dates(et)) | set(event_dates(em)))

    ln: list[str] = []
    ln.append("# 에피소드 1 — 보유를 쉽게 적은 표")
    ln.append("")
    ln.append("설명부터 짧게: **레짐** 같은 말 안 씀. 각 칸은 ‘그 날 종가로 계산했을 때 이 자산 이름으로 몇 퍼 몇 원인지’.")
    ln.append("")
    ln.append("| 항목 | 내용 |")
    ln.append("|---|---|")
    ln.append("| 기간 | **{} ~ {}** (에피소드 1 과 동일) |".format(EP0.date(), EP1.date()))
    ln.append(
        "| 상태 추적 시작 | `{}`부터 QQQ 종가 전 구간 로드 후 MA200 포함 (에피소드 첫날 상태가 과거까지 이어져야 해서 이렇게 함) |".format(
            ROOT_BT.date()
        )
    )
    ln.append("| 동일 시작 총액 | 에피소드 첫 거래일(`{}`) **종가 기준 순자산**을 각각 **{:,.0f}** 원으로 나누는 배수로 맞춤 |".format(EP0.date(), NAV0))
    ln.append("| 목록 기준 일 | 두 전략 중 **아무쪽이나 ‘오늘 뭘 했다’ 문자열 나온 날**만 |")
    ln.append("| 금 가격 | {} |".format(gnote.replace("|", "/")))
    ln.append("")
    ln.append("| 전략 | 에피소드 첫날 실제 순자산(미스케일) | 시작 맞춤 배수 |")
    ln.append("|:---:|:---:|:---:|")
    ln.append("| RMABS-4tier | {:,.2f} | {:.6f} |".format(nav_t.iloc[i0], st))
    ln.append("| MABS-Gold | {:,.2f} | {:.6f} |".format(nav_m.iloc[i0], sm))

    alt_e = alt.loc[(alt.index >= EP0.normalize()) & (alt.index <= EP1.normalize())]
    alm_e = alm.loc[(alm.index >= EP0.normalize()) & (alm.index <= EP1.normalize())]

    for d in days:
        if d not in alt_e.index or d not in alm_e.index:
            continue
        rt = last_row(et, d)
        rm = last_row(em, d)
        nav_ts = float(rt["nav_eod"]) * st
        nav_ms = float(rm["nav_eod"]) * sm
        us_t = float(rt["u_safe"]) * st
        ud_t = float(rt["u_stress_bounce"]) * st
        us_m = float(rm["u_safe"]) * sm
        ud_m = float(rm["u_stress_bounce"]) * sm
        rgt = Regime[str(rt["regime_after"])]
        rgm = Regime[str(rm["regime_after"])]
        pxt = {
            "au": float(alt_e.loc[d, "Close_safe"]),
            "df": float(alt_e.loc[d, "Close_bounce"]),
        }
        pxm = {
            "au": float(alm_e.loc[d, "Close_safe"]),
            "df": float(alm_e.loc[d, "Close_bounce"]),
        }
        bt = holding_bucket(nav_ts, rgt, us_t, ud_t, pxt["au"], pxt["df"], rmabs_four=True)
        bm = holding_bucket(nav_ms, rgm, us_m, ud_m, pxm["au"], pxm["df"], rmabs_four=False)

        ln.append("")
        ln.append("## {} — 일어난 일".format(d.strftime("%Y-%m-%d")))
        ln.append("")
        ln.append("- **RMABS-4tier 쪽 기록:** {}".format(tx_lab(rt.get("transitions"))))
        ln.append("- **MABS-Gold 쪽 기록:** {}".format(tx_lab(rm.get("transitions"))))
        ln.append("")
        ln.append("### RMABS-4tier · 그날 종가 기준 보유 (총 **{:,.0f}** 원)".format(nav_ts))
        ln.append("| 자산 항목 | 금액(원) | 비중 |")
        ln.append("|---|---:|---:|")
        for k, v in bt.items():
            ln.append("| {} | {:,.0f} | {} |".format(k, v, pct(v, nav_ts)))
        ln.append("")
        ln.append("### MABS-Gold · 그날 종가 기준 보유 (총 **{:,.0f}** 원)".format(nav_ms))
        ln.append("| 자산 항목 | 금액(원) | 비중 |")
        ln.append("|---|---:|---:|")
        for k, v in bm.items():
            ln.append("| {} | {:,.0f} | {} |".format(k, v, pct(v, nav_ms)))

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(ln).strip() + "\n", encoding="utf-8")
    print(OUT_MD.resolve())


if __name__ == "__main__":
    main()
