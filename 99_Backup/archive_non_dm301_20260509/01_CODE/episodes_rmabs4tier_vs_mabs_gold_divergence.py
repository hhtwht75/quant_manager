#!/usr/bin/env python3
"""RMABS-4tier vs MABS-Gold 갈등 상위 에피소드 + 이벤트 표 MD 작성."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import load_extended_daily  # noqa: E402
from rmabs_gold_simulation import load_gold_series  # noqa: E402
from fsm_four_asset_strategy import (  # noqa: E402
    CAPITAL_START,
    DEFAULT_TRAIL,
    get_or_build_ma200,
    run_fsm_backtest,
    _align_four,
    _align_five,
)

MIN_SPAN_DAYS = 252
TOP_K = 3


def _evt_str(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def overlaps(ai: int, aj: int, bi: int, bj: int) -> bool:
    return not (aj < bi or bj < ai)


def greedy_top_disjoint(
    d_ln_ratio: np.ndarray, min_span: int, k: int
) -> list[tuple[float, int, int]]:
    """시작별 최선 종료 한 쌍 → 점수 내림차순 비중복 탐욕. k 미만이면 brute로 비중복 보충."""

    def disjoint_from_all(si_l: int, ei_l: int, chosen: list[tuple[float, int, int]]) -> bool:
        return not any(overlaps(si_l, ei_l, u, v) for _, u, v in chosen)

    def score(si_l: int, ei_l: int) -> float:
        return abs(float(d_ln_ratio[ei_l]) - float(d_ln_ratio[si_l]))

    n = len(d_ln_ratio)
    pairs: list[tuple[float, int, int]] = []
    for si in range(n - min_span):
        sl = slice(si + min_span, n)

        dd = np.abs(d_ln_ratio[sl] - d_ln_ratio[si])


        jj = int(np.argmax(dd))


        ei = si + min_span + jj



        pairs.append((float(dd[jj]), si, ei))







    pairs.sort(key=lambda x: -x[0])



    out: list[tuple[float, int, int]] = []

    for sc, si, ei in pairs:

        if not disjoint_from_all(si, ei, out):
            continue
        out.append((sc, si, ei))



        if len(out) >= k:
            return out

    # 보충: 전체 (si,ej) brute — 서로 다른 비중복 구간까지 k건
    while len(out) < k:
        best_t: tuple[float, int, int] | None = None
        best_sc = float("-inf")
        for si in range(n - min_span):
            for ej in range(si + min_span, n):
                scv = score(si, ej)

                if scv <= best_sc:
                    continue
                if disjoint_from_all(si, ej, out):
                    best_sc = scv


                    best_t = (scv, si, ej)

        if best_t is None:


            break
        out.append(best_t)





    return out


def merged_events(
    ev_t: pd.DataFrame,
    ev_g: pd.DataFrame,
    ix: pd.DatetimeIndex,
    si: int,
    ei: int,
) -> pd.DataFrame:
    sd = ix[si].normalize()
    ed = ix[ei].normalize()
    t = ev_t.assign(Date=pd.to_datetime(ev_t["Date"]).dt.normalize())
    g = ev_g.assign(Date=pd.to_datetime(ev_g["Date"]).dt.normalize())
    tsub = t.loc[(t["Date"] >= sd) & (t["Date"] <= ed), ["Date", "regime_after", "transitions", "nav_eod"]]
    gsub = g.loc[(g["Date"] >= sd) & (g["Date"] <= ed), ["Date", "regime_after", "transitions", "nav_eod"]]
    merged = tsub.merge(
        gsub,
        on="Date",
        how="outer",
        suffixes=("_RM", "_Mg"),
        sort=True,
    )
    t_r = merged["transitions_RM"].map(_evt_str)
    tg = merged["transitions_Mg"].map(_evt_str)
    sel = (t_r.str.len() > 0) | (tg.str.len() > 0)
    tbl = merged[sel.fillna(False).values].copy()
    tbl.insert(0, "Date_str", pd.to_datetime(tbl["Date"]).dt.strftime("%Y-%m-%d"))
    tbl["nav_RM"] = pd.to_numeric(tbl["nav_eod_RM"], errors="coerce").round(2)
    tbl["nav_Mg"] = pd.to_numeric(tbl["nav_eod_Mg"], errors="coerce").round(2)
    return tbl[
        [
            "Date_str",
            "regime_after_RM",
            "regime_after_Mg",
            "transitions_RM",
            "transitions_Mg",
            "nav_RM",
            "nav_Mg",
        ]
    ]


def esc_cell(v: object) -> str:
    s = "" if pd.isna(v) else str(v).strip()
    return s.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    pa = argparse.ArgumentParser(description="RM vs Mg divergence episodes MD")
    pa.add_argument("--root", default="2002-10-01")
    pa.add_argument("--out", default="")
    pa.add_argument("--min-span-days", type=int, default=MIN_SPAN_DAYS)
    pa.add_argument("--top", type=int, default=TOP_K)
    a = pa.parse_args()

    rt = pd.Timestamp(a.root.strip())
    sg = load_extended_daily("QQQ")
    ng = load_extended_daily("QLD")
    ag = load_extended_daily("TQQQ")
    mav_full, _ = get_or_build_ma200(sg, "QQQ")
    ix0 = sg.index.intersection(ng.index).intersection(ag.index).sort_values()
    ix0 = ix0[ix0 >= rt]
    gold_df, gnote = load_gold_series(ix0[0], ix0[-1])
    ix = ix0.intersection(gold_df.index).sort_values()

    qi = sg.reindex(ix)
    li = ng.reindex(ix)
    ti = ag.reindex(ix)
    gx = gold_df.reindex(ix)
    mav = mav_full.reindex(ix)
    al4 = _align_five(qi, gx, qi, li, ti)
    alg = _align_four(qi, gx, li, ti)

    n4, et = run_fsm_backtest(
        al4, mav, trail_stop=DEFAULT_TRAIL, initial_capital=CAPITAL_START, use_safe_ma_rule=True
    )
    nmg, eg = run_fsm_backtest(
        alg, mav, trail_stop=DEFAULT_TRAIL, initial_capital=CAPITAL_START, use_safe_ma_rule=False
    )

    rl = np.log((n4 / nmg).astype(float).replace(0, np.nan)).values.astype(np.float64)
    greedy_sorted = sorted(
        greedy_top_disjoint(rl, a.min_span_days, a.top), key=lambda tri: (-tri[0], tri[1], tri[2])
    )

    outp = (
        Path(a.out.strip())
        if a.out.strip()
        else _ROOT / "03_RESULT/sensitivity/rmabs4tier_vs_mabs_gold_DIVERGE_TOP3.md"
    )
    outp.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# RMABS-4tier vs MABS-Gold 최대 분기 에피소드")
    lines.append("")
    lines.append("| # | 테스트 시작일 | 테스트 종료일 | 거래일수 | $|D_{\mathrm{e}}-D_{\mathrm{s}}|$ | 종료 NAV비(RM/Mg) | 구간 RM 배수 | 구간 Mg 배수 |")
    lines.append("|--:|---|---|---|--:|--:|--:|--:|")

    epis: list[tuple[int, float, int, int, float, float, float]] = []
    for rk, (sc, si, ei) in enumerate(greedy_sorted[: a.top], start=1):
        rte = float(n4.iloc[ei] / nmg.iloc[ei])
        rr = float(n4.iloc[ei] / n4.iloc[si])
        rm = float(nmg.iloc[ei] / nmg.iloc[si])
        epis.append((rk, sc, si, ei, rte, rr, rm))
        lines.append(
            "| {} | `{}` | `{}` | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                rk, ix[si].date(), ix[ei].date(), ei - si + 1, sc, rte, rr, rm
            )
        )

    lines.append("")
    lines.append("| 항목 | 내용 |")
    lines.append("|------|------|")
    lines.append(
        "| D(t) | ln(NAV_RM / NAV_Mg). 동일 교집합·동일 시작자본이라 상대 궤적 비교 치환치 |"
    )
    lines.append(
        "| 점수 | 시작별 최선 종료 쌍 후보를 점수로 정렬 후 비중복 탐욕, 부족 시 전체 brute로 비중복 보충, 표 순서는 점수 내림차순 |"
    )
    lines.append("| 금 가격 | {} |".format(gnote.replace("|", ";")))
    lines.append("| min 거래일 | {} |".format(a.min_span_days))
    lines.append("")
    lines.append("---")

    for rk, sc, si, ei, rte, rr, rm in epis:
        lines.extend(
            [
                "",
                "## 에피소드 {} — `{}` ~ `{}`".format(rk, ix[si].date(), ix[ei].date()),
                "",
                "**점수** {:.6f} · 종료 NAV비 RM/Mg **{:.6f}** · 구간 배수 RM **{:.6f}** Mg **{:.6f}**".format(
                    sc, rte, rr, rm
                ),
                "",
                "| 날짜 | regime RM | regime Mg | transitions RM | transitions Mg | NAV RM | NAV Mg |",
                "|------|-----------|-----------|----------------|----------------|--------:|--------:|",
            ]
        )
        mf = merged_events(et, eg, ix, si, ei)
        if mf.empty:
            lines.append("| *(전이 문자열 빈 행 없음)* | | | | | | |")
        else:
            for _, row in mf.iterrows():
                lines.append("| " + " | ".join(esc_cell(row[c]) for c in mf.columns) + " |")
        lines.append("")

    outp.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(outp.resolve())


if __name__ == "__main__":
    main()
