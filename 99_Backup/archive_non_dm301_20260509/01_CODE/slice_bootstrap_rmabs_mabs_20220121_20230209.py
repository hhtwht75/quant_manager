"""
2022-01-21 종가 직후 동일 자본 부트스트랩:
  · RMABS-4tier → 금(SAFE) 100%
  · MABS-QQQ   → 반등 QQQ(BOUNCE) 100%
이후 FSM 동일 룰로 2023-02-09(포함)까지 시뮬·비교표.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import load_extended_daily  # noqa: E402
from evaluation_metrics import full_metrics  # noqa: E402
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


def _weights_rmabs(row: pd.Series, sfx: float, bv: float, dv: float, av: float) -> dict[str, float]:
    nav = float(row["nav_eod"])
    reg = str(row["regime_after"])
    if nav < 1e-12:
        return {"금%": 0.0, "QQQ%": 0.0, "QLD%": 0.0, "TQQQ%": 0.0, "현금%": 100.0}
    u_safe = float(row["u_safe"])
    u_sb = float(row["u_stress_bounce"])
    if reg == "SAFE":
        return {
            "금%": 100.0 * u_safe * sfx / nav,
            "QQQ%": 100.0 * u_sb * bv / nav,
            "QLD%": 0.0,
            "TQQQ%": 0.0,
            "현금%": 0.0,
        }
    if reg == "SAFE_BOUNCE":
        return {
            "금%": 100.0 * u_safe * sfx / nav,
            "QQQ%": 100.0 * u_sb * bv / nav,
            "QLD%": 0.0,
            "TQQQ%": 0.0,
            "현금%": 0.0,
        }
    if reg == "BOUNCE":
        return {"금%": 0.0, "QQQ%": 100.0, "QLD%": 0.0, "TQQQ%": 0.0, "현금%": 0.0}
    if reg == "DEFENSE":
        return {"금%": 0.0, "QQQ%": 0.0, "QLD%": 100.0, "TQQQ%": 0.0, "현금%": 0.0}
    if reg == "AGG":
        return {"금%": 0.0, "QQQ%": 0.0, "QLD%": 0.0, "TQQQ%": 100.0, "현금%": 0.0}
    if reg == "IDLE":
        return {"금%": 0.0, "QQQ%": 0.0, "QLD%": 0.0, "TQQQ%": 0.0, "현금%": 100.0}
    return {"금%": 0.0, "QQQ%": 0.0, "QLD%": 0.0, "TQQQ%": 0.0, "현금%": 0.0}


def _weights_mabs(row: pd.Series, bv: float, dv: float, av: float) -> dict[str, float]:
    nav = float(row["nav_eod"])
    reg = str(row["regime_after"])
    if nav < 1e-12:
        return {"QQQ%": 0.0, "QLD%": 0.0, "TQQQ%": 0.0, "현금%": 100.0}
    if reg == "BOUNCE":
        return {"QQQ%": 100.0, "QLD%": 0.0, "TQQQ%": 0.0, "현금%": 0.0}
    if reg == "DEFENSE":
        return {"QQQ%": 0.0, "QLD%": 100.0, "TQQQ%": 0.0, "현금%": 0.0}
    if reg == "AGG":
        return {"QQQ%": 0.0, "QLD%": 0.0, "TQQQ%": 100.0, "현금%": 0.0}
    if reg == "IDLE":
        return {"QQQ%": 0.0, "QLD%": 0.0, "TQQQ%": 0.0, "현금%": 100.0}
    # MABS 경로에 SAFE 없음
    return {"QQQ%": 0.0, "QLD%": 0.0, "TQQQ%": 0.0, "현금%": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="2002-10-01")
    ap.add_argument("--bootstrap", default="2022-01-21")
    ap.add_argument("--end", default="2023-02-09")
    ap.add_argument("--capital", type=float, default=float(CAPITAL_START))
    ap.add_argument("--trail", type=float, default=DEFAULT_TRAIL)
    args = ap.parse_args()

    root = pd.Timestamp(args.root.strip())
    d_boot = pd.Timestamp(args.bootstrap)
    d_end = pd.Timestamp(args.end)
    cap = float(args.capital)

    sg = load_extended_daily("QQQ")
    ng = load_extended_daily("QLD")
    ag = load_extended_daily("TQQQ")
    ma_full, _ = get_or_build_ma200(sg, "QQQ", force_rebuild=False)

    ix0 = sg.index.intersection(ng.index).intersection(ag.index).sort_values()
    ix0 = ix0[ix0 >= root]
    gold_df, _ = load_gold_series(ix0[0], ix0[-1])
    ix = ix0.intersection(gold_df.index).sort_values()
    if len(ix) < MA_WINDOW:
        raise SystemExit(f"일수 부족: {len(ix)} < {MA_WINDOW}")

    mav = ma_full.reindex(ix)
    sg_i = sg.reindex(ix)
    ng_i = ng.reindex(ix)
    ag_i = ag.reindex(ix)
    g_i = gold_df.reindex(ix)
    aligned_r = _align_five(sg_i, g_i, sg_i, ng_i, ag_i)
    aligned_m = _align_four(sg_i, sg_i, ng_i, ag_i)

    common_kw = dict(
        trail_stop=args.trail,
        initial_capital=cap,
        bootstrap_eod_date=d_boot,
        bootstrap_capital=cap,
        end_date=d_end,
    )

    nav_r, ev_r = run_fsm_backtest(aligned_r, mav, use_safe_ma_rule=True, **common_kw)
    nav_m, ev_m = run_fsm_backtest(aligned_m, mav, use_safe_ma_rule=False, **common_kw)

    if len(nav_r) < 2 or len(nav_m) < 2:
        raise SystemExit("시뮬레이션 거래일이 부족함")

    m_r = full_metrics(nav_r)
    m_m = full_metrics(nav_m)

    nr0 = float(nav_r.iloc[0])
    nr1 = float(nav_r.iloc[-1])
    nm0 = float(nav_m.iloc[0])
    nm1 = float(nav_m.iloc[-1])

    ret_r_boot = nr1 / cap - 1.0
    ret_m_boot = nm1 / cap - 1.0
    ret_r_series = nr1 / nr0 - 1.0
    ret_m_series = nm1 / nm0 - 1.0

    out_md = (
        _ROOT
        / "03_RESULT/sensitivity/"
        / f"bootstrap_rmabs_mabscompare_{pd.Timestamp(d_boot).strftime('%Y%m%d')}_{pd.Timestamp(d_end).strftime('%Y%m%d')}.md"
    )
    rows_summary = [
        "| 지표 | RMABS-4tier | MABS-QQQ | 차이(R−M) |",
        "|---|---:|---:|---:|",
        f"| 부트스트랩 자본 (동일, {d_boot.date()} 종가 직후) | {cap:,.2f} | {cap:,.2f} | 0 |",
        f"| 첫 시뮬레이션 거래일 | {nav_r.index[0].date()} | {nav_m.index[0].date()} | — |",
        "| 첫 거래일 종가 NAV | "
        f"{nr0:,.2f} | {nm0:,.2f} | {nr0 - nm0:,.2f} |",
        f"| 최종 거래일 ({d_end.date()}) NAV | {nr1:,.2f} | {nm1:,.2f} | {nr1 - nm1:,.2f} |",
        f"| 총수익률 (부트스트랩 자본 대비) | {ret_r_boot * 100:.2f}% | {ret_m_boot * 100:.2f}% | "
        f"{(ret_r_boot - ret_m_boot) * 100:.2f}pp |",
        f"| 총수익률 (첫 시뮬레이션일 NAV→종료) | {ret_r_series * 100:.2f}% | {ret_m_series * 100:.2f}% | "
        f"{(ret_r_series - ret_m_series) * 100:.2f}pp |",
        f"| MDD (구간) | {float(m_r['mdd']) * 100:.2f}% | {float(m_m['mdd']) * 100:.2f}% | "
        f"{(float(m_r['mdd']) - float(m_m['mdd'])) * 100:.2f}pp |",
        f"| Sharpe (구간, 일간) | {float(m_r['sharpe']):.3f} | {float(m_m['sharpe']):.3f} | "
        f"{float(m_r['sharpe']) - float(m_m['sharpe']):+.3f} |",
        f"| Sortino (구간) | {float(m_r['sortino']):.3f} | {float(m_m['sortino']):.3f} | "
        f"{float(m_r['sortino']) - float(m_m['sortino']):+.3f} |",
        "",
        "첫 거래일 NAV는 진입 다음 영업일 종가 재평가(보유수량 불변) 결과이다.",
        "",
        "## 이벤트 발생일·종료일 보유(%)",
        "",
        "| Date | RMABS 이벤트 | R 금% | R QQQ% | R QLD% | R TQQQ% | M 이벤트 | M QQQ% | M QLD% | M TQQQ% | NAV_R | NAV_M |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]

    ix_sim = nav_r.index
    ev_r_ix = ev_r.set_index("Date")
    ev_m_ix = ev_m.set_index("Date")

    evt_dates: list[pd.Timestamp] = []
    for d in ix_sim:
        er = ev_r_ix.loc[d]
        em = ev_m_ix.loc[d]
        tr = str(er["transitions"]).strip()
        tm = str(em["transitions"]).strip()
        if tr or tm:
            evt_dates.append(d)
    evt_dates.append(ix_sim[-1])

    seen: set = set()
    for d in evt_dates:
        if d in seen:
            continue
        seen.add(d)
        er = ev_r_ix.loc[d]
        em = ev_m_ix.loc[d]
        sfx = float(aligned_r.loc[d, "Close_safe"])
        bv = float(aligned_r.loc[d, "Close_bounce"])
        dv = float(aligned_r.loc[d, "Close_defense"])
        av = float(aligned_r.loc[d, "Close_agg"])
        wr = _weights_rmabs(er, sfx, bv, dv, av)
        wm = _weights_mabs(em, bv, dv, av)
        tr = str(er["transitions"]).strip() or "—"
        tm = str(em["transitions"]).strip() or "—"
        rows_summary.append(
            f"| {d.date()} | {tr} | {wr['금%']:.1f} | {wr['QQQ%']:.1f} | {wr['QLD%']:.1f} | {wr['TQQQ%']:.1f} | "
            f"{tm} | {wm['QQQ%']:.1f} | {wm['QLD%']:.1f} | {wm['TQQQ%']:.1f} | "
            f"{float(er['nav_eod']):,.2f} | {float(em['nav_eod']):,.2f} |"
        )

    body = "\n".join(
        [
            f"# 부트스트랩 비교: {d_boot.date()} 동일자본 (R 금100% / M QQQ100%) → {d_end.date()}",
            "",
            f"- trail={args.trail}, 시뮬레이션 거래일 수: {len(nav_r)} (R·M 동일 달력)",
            "",
            *rows_summary,
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(body, encoding="utf-8")
    print(str(out_md))
    print(body)


if __name__ == "__main__":
    main()
