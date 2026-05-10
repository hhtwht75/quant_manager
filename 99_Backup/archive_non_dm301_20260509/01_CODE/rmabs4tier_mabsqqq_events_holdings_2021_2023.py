"""
RMABS-4tier vs MABS-QQQ: 2021–2023 이벤트일 보유 비중·NAV 변화량 표.

전구간으로 FSM을 돌린 뒤(MA200·과거 시그널 일관성), 지정 구간에서
어느 한쪽이라도 transitions 가 비어 있지 않은 날만 출력.
"""

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
    _align_five,
    _align_four,
    get_or_build_ma200,
    run_fsm_backtest,
)
from rmabs_gold_simulation import load_gold_series  # noqa: E402


def _weights(
    row: pd.Series,
    sfx: float,
    bv: float,
    dv: float,
    av: float,
) -> dict[str, float]:
    nav = float(row["nav_eod"])
    reg = str(row["regime_after"])
    if nav < 1e-12:
        return {"gold": 0.0, "qqq_bounce": 0.0, "qld": 0.0, "tqqq": 0.0, "idle": 100.0}

    u_safe = float(row["u_safe"])
    u_sbounce = float(row["u_stress_bounce"])

    if reg == "SAFE":
        g = 100.0 * u_safe * sfx / nav
        qb = 100.0 * u_sbounce * bv / nav
        return {"gold": g, "qqq_bounce": qb, "qld": 0.0, "tqqq": 0.0, "idle": 0.0}
    if reg == "SAFE_BOUNCE":
        g = 100.0 * u_safe * sfx / nav
        qb = 100.0 * u_sbounce * bv / nav
        return {"gold": g, "qqq_bounce": qb, "qld": 0.0, "tqqq": 0.0, "idle": 0.0}
    if reg == "BOUNCE":
        return {"gold": 0.0, "qqq_bounce": 100.0, "qld": 0.0, "tqqq": 0.0, "idle": 0.0}
    if reg == "DEFENSE":
        return {"gold": 0.0, "qqq_bounce": 0.0, "qld": 100.0, "tqqq": 0.0, "idle": 0.0}
    if reg == "AGG":
        return {"gold": 0.0, "qqq_bounce": 0.0, "qld": 0.0, "tqqq": 100.0, "idle": 0.0}
    if reg == "IDLE":
        return {"gold": 0.0, "qqq_bounce": 0.0, "qld": 0.0, "tqqq": 0.0, "idle": 100.0}
    return {"gold": 0.0, "qqq_bounce": 0.0, "qld": 0.0, "tqqq": 0.0, "idle": 0.0}


def _fmt_dnav(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:,.2f}"


def main() -> None:
    root = pd.Timestamp("2002-10-01")
    d0 = pd.Timestamp("2021-01-01")
    d1 = pd.Timestamp("2023-12-31")

    sg = load_extended_daily("QQQ")
    ng = load_extended_daily("QLD")
    ag = load_extended_daily("TQQQ")

    ma_full, _ = get_or_build_ma200(sg, "QQQ", force_rebuild=False)

    ix0 = sg.index.intersection(ng.index).intersection(ag.index).sort_values()
    ix0 = ix0[ix0 >= root]
    gold_df, _ = load_gold_series(ix0[0], ix0[-1])
    ix = ix0.intersection(gold_df.index).sort_values()
    if len(ix) < MA_WINDOW:
        raise SystemExit(f"교집합 거래일 {len(ix)} < MA_WINDOW {MA_WINDOW}")

    mav = ma_full.reindex(ix)
    sg_i, ng_i, ag_i = sg.reindex(ix), ng.reindex(ix), ag.reindex(ix)
    g_i = gold_df.reindex(ix)

    aligned_4tier = _align_five(sg_i, g_i, sg_i, ng_i, ag_i)
    aligned_mabs = _align_four(sg_i, sg_i, ng_i, ag_i)

    _, ev_rm = run_fsm_backtest(
        aligned_4tier,
        mav,
        trail_stop=DEFAULT_TRAIL,
        initial_capital=CAPITAL_START,
        use_safe_ma_rule=True,
    )
    _, ev_mabs = run_fsm_backtest(
        aligned_mabs,
        mav,
        trail_stop=DEFAULT_TRAIL,
        initial_capital=CAPITAL_START,
        use_safe_ma_rule=False,
    )

    ix_win = ix[(ix >= d0) & (ix <= d1)]
    rm_by_date = ev_rm.set_index("Date")
    mb_by_date = ev_mabs.set_index("Date")

    rows_out: list[dict[str, object]] = []

    for d in ix_win:
        rmr = rm_by_date.loc[d]
        mbr = mb_by_date.loc[d]
        tr_rm = str(rmr["transitions"]).strip()
        tr_mb = str(mbr["transitions"]).strip()
        if not tr_rm and not tr_mb:
            continue

        pos = ix.get_loc(d)
        if not isinstance(pos, int):
            raise TypeError(f"예상치 못한 인덱스 타입: {type(pos)}")
        prev_d = ix[pos - 1] if pos > 0 else None

        sfx = float(aligned_4tier.loc[d, "Close_safe"])
        bv = float(aligned_4tier.loc[d, "Close_bounce"])
        dv = float(aligned_4tier.loc[d, "Close_defense"])
        av = float(aligned_4tier.loc[d, "Close_agg"])

        w_rm = _weights(rmr, sfx, bv, dv, av)
        w_mb = _weights(mbr, sfx, bv, dv, av)

        nav_rm_f = float(rmr["nav_eod"])
        nav_mb_f = float(mbr["nav_eod"])
        prev_rm_nav = float(rm_by_date.loc[prev_d, "nav_eod"]) if prev_d is not None else None
        prev_mb_nav = float(mb_by_date.loc[prev_d, "nav_eod"]) if prev_d is not None else None
        d_rm = None if prev_rm_nav is None else nav_rm_f - prev_rm_nav
        d_mb = None if prev_mb_nav is None else nav_mb_f - prev_mb_nav

        rows_out.append(
            {
                "Date": d,
                "evt_RMABS": tr_rm or "—",
                "evt_MABS": tr_mb or "—",
                "R_Gold%": w_rm["gold"],
                "R_QQQ%": w_rm["qqq_bounce"],
                "R_QLD%": w_rm["qld"],
                "R_TQQQ%": w_rm["tqqq"],
                "R_IDLE%": w_rm["idle"],
                "M_QQQ%": w_mb["qqq_bounce"],
                "M_QLD%": w_mb["qld"],
                "M_TQQQ%": w_mb["tqqq"],
                "M_IDLE%": w_mb["idle"],
                "NAV_RMABS": nav_rm_f,
                "ΔNAV_RMABS": d_rm,
                "NAV_MABS": nav_mb_f,
                "ΔNAV_MABS": d_mb,
            }
        )

    df = pd.DataFrame(rows_out)
    stem = (
        _ROOT / "03_RESULT/sensitivity/"
        "rmabs4tier_vs_mabsqqq_events_holdings_2021_2023"
    )
    csv_path = stem.with_suffix(".csv")
    md_path = Path(str(stem) + ".md")

    if len(df):
        df.to_csv(csv_path, index=False)

    hdr = (
        "| Date | RMABS 이벤트 | MABS 이벤트 | R 금% | R QQQ% | R QLD% | R TQQQ% | R 현금% | "
        "M QQQ% | M QLD% | M TQQQ% | M 현금% | NAV_R | ΔNAV_R | NAV_M | ΔNAV_M |"
    )
    sep = "|" + "|".join(["---"] * 16) + "|"
    lines = [
        "# RMABS-4tier vs MABS-QQQ 이벤트일 보유 (2021–2023)\n",
        "",
        "**전구간** 백테스트(교집합·MA200 동일) 후 **2021-01-01 ~ 2023-12-31** 거래일 중 "
        "**한쪽이라도** `transitions`가 비어 있지 않은 날만 기록. ",
        "**ΔNAV**는 교집합 달력에서 **직전 영업일** 대비 종가 NAV 변화(USD). ",
        "**보유 %**는 당일 종가 기준 레짐·수량 환산 노출 비중.",
        "",
        "설정: `trail_stop=DEFAULT`, 초기 자본 {:.0f}. ".format(CAPITAL_START)
        + "RMABS-4tier=금 안전+규칙4·RSI; MABS-QQQ=규칙4→방어 QQQ 100%.",
        "",
        hdr,
        sep,
    ]
    if len(df) == 0:
        lines.append("| *(해당 구간 이벤트 없음)* " + "|" + " — |" * 15)
    else:
        for _, r in df.iterrows():
            dt = pd.Timestamp(r["Date"]).strftime("%Y-%m-%d")
            evr = str(r["evt_RMABS"]).replace("|", "/")
            evm = str(r["evt_MABS"]).replace("|", "/")
            lines.append(
                f"| {dt} | {evr} | {evm} | "
                f"{r['R_Gold%']:.2f} | {r['R_QQQ%']:.2f} | {r['R_QLD%']:.2f} | {r['R_TQQQ%']:.2f} | {r['R_IDLE%']:.2f} | "
                f"{r['M_QQQ%']:.2f} | {r['M_QLD%']:.2f} | {r['M_TQQQ%']:.2f} | {r['M_IDLE%']:.2f} | "
                f"{r['NAV_RMABS']:,.2f} | {_fmt_dnav(r['ΔNAV_RMABS'])} | "
                f"{r['NAV_MABS']:,.2f} | {_fmt_dnav(r['ΔNAV_MABS'])} |"
            )

    lines.append("")
    lines.append(f"- CSV: `{csv_path.name}`")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(str(md_path))
    print(str(csv_path))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
