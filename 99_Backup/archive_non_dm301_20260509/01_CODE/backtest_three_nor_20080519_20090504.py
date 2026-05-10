"""
동일 초기 자본 · 2008-05-19 종가 후 DEFENSE(QLD 100%) 부트스트랩 → 2009-05-04까지 백테스트.

RMABS-4tier-Gold / RMABS-4tier-Cash / MABS-QQQ.
한 줄이라도 transitions 가 있는 날 → 세 전략 보유 비중·순자산 한 표로 저장.
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


def _flat_cash(ix: pd.DatetimeIndex) -> pd.DataFrame:
    ix = ix.sort_values()
    one = pd.Series(1.0, index=ix, dtype=float)
    return pd.DataFrame({"Open": one, "High": one, "Low": one, "Close": one})


def weights_rmabs(
    row: pd.Series,
    sfx: float,
    bv: float,
    dv: float,
    av: float,
) -> tuple[float, float, float, float, float]:
    """(안전레그%, QQQ 반등 stress%, QLD%, TQQQ%, 현금%)."""
    nav = float(row["nav_eod"])
    reg = str(row["regime_after"])
    if nav < 1e-12:
        return (0.0, 0.0, 0.0, 0.0, 100.0)
    us = float(row["u_safe"])
    ub = float(row["u_stress_bounce"])

    if reg in ("SAFE", "SAFE_BOUNCE"):
        return (
            100.0 * us * sfx / nav,
            100.0 * ub * bv / nav,
            0.0,
            0.0,
            0.0,
        )
    if reg == "BOUNCE":
        return (0.0, 100.0, 0.0, 0.0, 0.0)
    if reg == "DEFENSE":
        return (0.0, 0.0, 100.0, 0.0, 0.0)
    if reg == "AGG":
        return (0.0, 0.0, 0.0, 100.0, 0.0)
    if reg == "IDLE":
        return (0.0, 0.0, 0.0, 0.0, 100.0)
    return (0.0, 0.0, 0.0, 0.0, 0.0)


def weights_mabs(
    row: pd.Series,
    bv: float,
    dv: float,
    av: float,
) -> tuple[float, float, float, float]:
    """(QQQ%, QLD%, TQQQ%, 현금%)."""
    nav = float(row["nav_eod"])
    reg = str(row["regime_after"])
    if nav < 1e-12:
        return (0.0, 0.0, 0.0, 100.0)
    if reg == "BOUNCE":
        return (100.0, 0.0, 0.0, 0.0)
    if reg == "DEFENSE":
        return (0.0, 100.0, 0.0, 0.0)
    if reg == "AGG":
        return (0.0, 0.0, 100.0, 0.0)
    if reg == "IDLE":
        return (0.0, 0.0, 0.0, 100.0)
    return (0.0, 0.0, 0.0, 0.0)


def main() -> None:
    root = pd.Timestamp("2002-10-01")
    d_boot = pd.Timestamp("2008-05-19")
    d_end = pd.Timestamp("2009-05-04")
    cap = float(CAPITAL_START)

    sg = load_extended_daily("QQQ")
    ng = load_extended_daily("QLD")
    ag = load_extended_daily("TQQQ")
    ma_full, _ = get_or_build_ma200(sg, "QQQ", force_rebuild=False)

    ix0 = sg.index.intersection(ng.index).intersection(ag.index).sort_values()
    ix0 = ix0[ix0 >= root]
    gold_df, gold_note = load_gold_series(ix0[0], ix0[-1])
    ix = ix0.intersection(gold_df.index).sort_values()
    if len(ix) < MA_WINDOW:
        raise SystemExit(f"교집합 일부족 {len(ix)} < {MA_WINDOW}")
    if d_boot not in ix or d_end not in ix:
        raise SystemExit("부트스트랩·종료일이 교집합에 없음")

    mav = ma_full.reindex(ix)
    sg_i, ng_i, ag_i = sg.reindex(ix), ng.reindex(ix), ag.reindex(ix)
    g_i = gold_df.reindex(ix)
    cash_i = _flat_cash(ix)
    alg = _align_five(sg_i, g_i, sg_i, ng_i, ag_i)
    alc = _align_five(sg_i, cash_i, sg_i, ng_i, ag_i)
    am = _align_four(sg_i, sg_i, ng_i, ag_i)

    common = dict(
        trail_stop=DEFAULT_TRAIL,
        initial_capital=cap,
        bootstrap_eod_date=d_boot,
        bootstrap_capital=cap,
        bootstrap_initial_regime="DEFENSE",
        end_date=d_end,
    )

    _, ev_g = run_fsm_backtest(alg, mav, use_safe_ma_rule=True, **common)
    _, ev_c = run_fsm_backtest(alc, mav, use_safe_ma_rule=True, **common)
    _, ev_m = run_fsm_backtest(am, mav, use_safe_ma_rule=False, **common)

    ix_sim = alg.index[
        (alg.index > pd.Timestamp(d_boot)) & (alg.index <= pd.Timestamp(d_end))
    ]

    gb = ev_g.set_index("Date")
    cb = ev_c.set_index("Date")
    mb = ev_m.set_index("Date")

    rows_out: list[dict[str, object]] = []

    for d in ix_sim:
        g = gb.loc[d]
        c = cb.loc[d]
        m = mb.loc[d]
        tg = str(g["transitions"]).strip()
        tc = str(c["transitions"]).strip()
        tm = str(m["transitions"]).strip()
        if not tg and not tc and not tm:
            continue

        sfxg = float(alg.loc[d, "Close_safe"])
        bv = float(alg.loc[d, "Close_bounce"])
        dv = float(alg.loc[d, "Close_defense"])
        av = float(alg.loc[d, "Close_agg"])

        Gfx, Gqq, Gqld, Gtqq, Gid = weights_rmabs(g, sfxg, bv, dv, av)
        sfxc = float(alc.loc[d, "Close_safe"])
        Cfx, Cqq, Cqld, Ctqq, Cid = weights_rmabs(c, sfxc, bv, dv, av)
        Mqq, Mqld, Mtqq, Mid = weights_mabs(m, bv, dv, av)

        rows_out.append(
            {
                "Date": d,
                "이벤트_Gold": tg or "—",
                "이벤트_Cash": tc or "—",
                "이벤트_MABS": tm or "—",
                "G_안전%": Gfx,
                "G_QQQ%": Gqq,
                "G_QLD%": Gqld,
                "G_TQQQ%": Gtqq,
                "G_현금%": Gid,
                "C_현금1%": Cfx,
                "C_QQQ%": Cqq,
                "C_QLD%": Cqld,
                "C_TQQQ%": Ctqq,
                "C_현금%": Cid,
                "M_QQQ%": Mqq,
                "M_QLD%": Mqld,
                "M_TQQQ%": Mtqq,
                "M_현금%": Mid,
                "NAV_Gold": float(g["nav_eod"]),
                "NAV_Cash": float(c["nav_eod"]),
                "NAV_MABS": float(m["nav_eod"]),
            }
        )

    df = pd.DataFrame(rows_out)
    stem = _ROOT / "03_RESULT/sensitivity/nor_boot_20080519_20090504_triple_fsm"
    csv_p = stem.with_suffix(".csv")
    md_p = Path(str(stem) + "_events.md")
    stem.parent.mkdir(parents=True, exist_ok=True)

    if len(df):
        df.to_csv(csv_p, index=False)

    def esc(s: str) -> str:
        return str(s).replace("|", "\\|")

    intro = [
        "# DEFENSE 부트스트랩 후 이벤트일 요약 (Gold / Cash / MABS-QQQ)\n",
        "",
        f"- 초기 자본 **{cap:,.0f}** USD · **`{d_boot.date()}` 종가 직후** 세 전략 모두 **DEFENSE(QLD)** 100%",
        f"- 시뮬레이션 루프: **{ix_sim[0].date()}** ~ **{ix_sim[-1].date()}** (종료일 `{d_end.date()}` 포함)",
        f"- RMABS-4tier-Gold 안전 레그: {gold_note} · RMABS-4tier-Cash 안전: **종가 1.0** (무이자 명목)",
        "",
        "| Date | G evt | C evt | M evt | G 안전% | G QQQ% | G QLD% | G TQQQ% | G 현금% | C 현금1% | C QQQ% | C QLD% | C TQQQ% | C 현금% | M QQQ% | M QLD% | M TQQQ% | M 현금% | NAV_G | NAV_C | NAV_M |",
        "|------|-------|-------|-------|---------|--------|--------|---------|---------|----------|--------|--------|---------|--------|--------|--------|---------|--------|--------|---------|",
    ]

    md_rows: list[str] = []
    for _, r in df.iterrows():
        md_rows.append(
            f"| {pd.Timestamp(r['Date']).date()} | "
            f"{esc(r['이벤트_Gold'])} | {esc(r['이벤트_Cash'])} | {esc(r['이벤트_MABS'])} | "
            f"{r['G_안전%']:.2f} | {r['G_QQQ%']:.2f} | {r['G_QLD%']:.2f} | {r['G_TQQQ%']:.2f} | {r['G_현금%']:.2f} | "
            f"{r['C_현금1%']:.2f} | {r['C_QQQ%']:.2f} | {r['C_QLD%']:.2f} | {r['C_TQQQ%']:.2f} | {r['C_현금%']:.2f} | "
            f"{r['M_QQQ%']:.2f} | {r['M_QLD%']:.2f} | {r['M_TQQQ%']:.2f} | {r['M_현금%']:.2f} | "
            f"{r['NAV_Gold']:,.2f} | {r['NAV_Cash']:,.2f} | {r['NAV_MABS']:,.2f} |"
        )

    if not md_rows:
        md_rows.append("| *(이벤트 없음)* |" + " |" * 20)

    md_p.write_text("\n".join(intro + md_rows) + "\n", encoding="utf-8")
    print(str(md_p.resolve()))
    print(str(csv_p.resolve()))
    print(f"event_rows={len(df)}")
    print("\n".join(intro + md_rows))


if __name__ == "__main__":
    main()
