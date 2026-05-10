"""
DMABS 256조합 무작위 윈도 MC · Sortino 평균 집계.

4레 각각 {TQQQ, QLD, QQQ, Gold} → 4^4 = 256.
_Close_sig 및 MA200·스트레스 MA 는 **항상 QQQ 종가** (기존 DMABS 스위트와 동일).

 ``--merge-bounce-trail`` 이면 AGG 레짐 없이 반등 종가만 트레일; ``Close_agg`` 열은 반등과 동일 시계열을 쓴다
 (공격·반등 동일 역할 변형).

  python3 01_CODE/mc_dmabs_four_leg_256_sortino.py \\
    --root 1999-03-10 --mc-years 3 --mc-iters 3000 --mc-seed 42

  python3 01_CODE/mc_dmabs_four_leg_256_sortino.py \\
    --merge-bounce-trail \\
    --root 1999-03-10 --mc-years 3 --mc-iters 3000 --mc-seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import load_extended_daily  # noqa: E402
from rmabs_gold_simulation import load_gold_series  # noqa: E402

from fsm_four_asset_strategy import (
    CAPITAL_START,
    DEFAULT_TRAIL,
    MA_WINDOW,
    _core_metrics,
    _distribution_block,
    get_or_build_ma200,
    mc_min_trading_days,
    run_fsm_backtest,
    _align_five,
)

OUT_DIR_DEFAULT = _ROOT / "03_RESULT" / "sensitivity"
LEGS = ["TQQQ", "QLD", "QQQ", "Gold"]

LegKey = tuple[str, str, str, str]  # agg, bounce, defense, safe


def _pick(
    tick: str,
    *,
    sg: pd.DataFrame,
    ql: pd.DataFrame,
    tg: pd.DataFrame,
    gld: pd.DataFrame,
) -> pd.DataFrame:
    up = tick.strip().upper()
    if up == "QQQ":
        return sg
    if up == "QLD":
        return ql
    if up == "TQQQ":
        return tg
    if up == "GOLD":
        return gld
    raise ValueError(tick)


def _combo_label(t: LegKey) -> str:
    a, b, d, s = t
    return f"agg{a}_bounce{b}_defense{d}_safe{s}"


def main() -> None:
    ap = argparse.ArgumentParser(description="DMABS 4레그 256조합 MC — Sortino 등")
    ap.add_argument("--root", default="1999-03-10")
    ap.add_argument("--trail", type=float, default=DEFAULT_TRAIL)
    ap.add_argument("--mc-years", type=float, default=3.0)
    ap.add_argument("--mc-iters", type=int, default=3000)
    ap.add_argument("--mc-seed", type=int, default=42)
    ap.add_argument(
        "--merge-bounce-trail",
        action="store_true",
        help="반등 종가 트레일만·AGG 레짐 제거(agg 열=반등 틱과 동일 종가 시계열)",
    )
    ap.add_argument("--out-dir", default="", help="비우면 03_RESULT/sensitivity")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve() if args.out_dir.strip() else OUT_DIR_DEFAULT
    out_root.mkdir(parents=True, exist_ok=True)

    root_ts = pd.Timestamp(args.root.strip())
    sg_full = load_extended_daily("QQQ")
    ql_full = load_extended_daily("QLD")
    tg_full = load_extended_daily("TQQQ")
    ma_full, csv_ma = get_or_build_ma200(sg_full, "QQQ", force_rebuild=False)

    ix0_all = sg_full.index.intersection(ql_full.index).intersection(tg_full.index).sort_values()
    ix0 = ix0_all[ix0_all >= root_ts]
    gold_df, gold_note = load_gold_series(ix0[0], ix0[-1])
    ix = ix0.intersection(gold_df.index).sort_values()

    if len(ix) < MA_WINDOW:
        raise SystemExit(f"교집합 거래일 {len(ix)} < MA200 {MA_WINDOW}")
    lo = mc_min_trading_days(args.mc_years)
    if len(ix) < lo:
        raise SystemExit(f"전체 일수 {len(ix)} < 최소 윈도 거래일 {lo}")

    sg_i = sg_full.reindex(ix)
    ql_i = ql_full.reindex(ix)
    tg_i = tg_full.reindex(ix)
    g_i = gold_df.reindex(ix)
    mav = ma_full.reindex(ix)

    combos: list[LegKey] = list(product(LEGS, LEGS, LEGS, LEGS))
    assert len(combos) == 256

    nav_map: dict[LegKey, pd.Series] = {}
    mode = "merge-bounce-trail" if args.merge_bounce_trail else "DMABS(분리 AGG)"
    print(f"전구간 NAV 256조합 (Close_sig=QQQ, {mode}) …")
    for i, (agg_t, bounce_t, defense_t, safe_t) in enumerate(combos):
        if (i + 1) % 32 == 0:
            print(f"  … {i + 1}/256")
        bf = _pick(bounce_t, sg=sg_i, ql=ql_i, tg=tg_i, gld=g_i)
        af = bf if args.merge_bounce_trail else _pick(agg_t, sg=sg_i, ql=ql_i, tg=tg_i, gld=g_i)
        df_f = _pick(defense_t, sg=sg_i, ql=ql_i, tg=tg_i, gld=g_i)
        sf = _pick(safe_t, sg=sg_i, ql=ql_i, tg=tg_i, gld=g_i)
        aligned = _align_five(sg_i, sf, bf, df_f, af)
        nav, _ = run_fsm_backtest(
            aligned,
            mav,
            trail_stop=args.trail,
            initial_capital=CAPITAL_START,
            use_safe_ma_rule=True,
            stress_bleed_mode="MA5_CROSS_MA120_FULL",
            merge_agg_into_bounce_trailing=bool(args.merge_bounce_trail),
        )
        nav_map[(agg_t, bounce_t, defense_t, safe_t)] = nav.astype(float)

    n_dates = len(ix)
    rng = np.random.default_rng(args.mc_seed)
    sortino_lists: dict[LegKey, list[float]] = {k: [] for k in combos}

    print(f"MC: ≥{args.mc_years:g}년 창, {args.mc_iters}회, seed={args.mc_seed}")
    for _ in range(args.mc_iters):
        s_pos = int(rng.integers(0, n_dates - lo + 1))
        l_max = n_dates - s_pos
        l_win = int(rng.integers(lo, l_max + 1))
        for k in combos:
            sub = nav_map[k].iloc[s_pos : s_pos + l_win]
            sortino_lists[k].append(float(_core_metrics(sub)["sortino"]))

    rows: list[dict[str, object]] = []
    for k in combos:
        s = pd.Series(sortino_lists[k])
        blk = _distribution_block(s)
        agg_t, bounce_t, defense_t, safe_t = k
        rows.append(
            {
                "agg": agg_t,
                "bounce": bounce_t,
                "defense": defense_t,
                "safe": safe_t,
                "sortino_mean": blk["mean"],
                "sortino_median": blk["median"],
                "sortino_std": blk["std"],
                "combo_id": _combo_label(k),
            }
        )

    rows.sort(key=lambda r: float(r["sortino_mean"]), reverse=True)

    fx0 = pd.Timestamp(ix[0]).strftime("%Y%m%d")
    fx1 = pd.Timestamp(ix[-1]).strftime("%Y%m%d")
    merge_tag = "mergebounce_" if args.merge_bounce_trail else ""
    stem = (
        f"mc_dmabs_fourleg256_{merge_tag}sortino_{fx0}_{fx1}_"
        f"{args.mc_years:g}yr_n{args.mc_iters}_s{args.mc_seed}"
    )
    json_path = out_root / f"{stem}_summary.json"
    csv_path = out_root / f"{stem}_sortino_rank.csv"
    md_path = out_root / f"{stem}_READ_ME.md"

    desc256 = (
        "256 combos: agg,bounce,defense,safe ∈ {TQQQ,QLD,QQQ,Gold}^4 · "
        "Close_sig always QQQ · DMABS MA5>M120 · sortino_mean from random windows"
    )
    if args.merge_bounce_trail:
        desc256 += (
            " · merge_bounce_trail: no AGG regime; trailing on Close_bounce only; "
            "aligned Close_agg == Close_bounce (agg ticker label in CSV is positional only)"
        )
    meta = {
        "description": desc256,
        "merge_bounce_trail": bool(args.merge_bounce_trail),
        "gold_series": gold_note,
        "ma200_csv": str(csv_ma.resolve()),
        "--root": str(root_ts.date()),
        "index_first_last_days": {"first": str(ix[0].date()), "last": str(ix[-1].date()), "days": len(ix)},
        "trail": float(args.trail),
        "mc_years": float(args.mc_years),
        "mc_min_trading_days": lo,
        "mc_iters": int(args.mc_iters),
        "mc_seed": int(args.mc_seed),
    }

    blob = {
        "meta": meta,
        "sortino_sorted_desc": rows,
        "top20_by_sortino_mean": rows[:20],
        "bottom10_by_sortino_mean": rows[-10:],
    }
    json_path.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    lines = [
        "# DMABS 256조합 — MC Sortino 평균 (정렬: 내림차순)",
        "",
        f"- 교집합 `{ix[0].date()}` ~ `{ix[-1].date()}` (**{len(ix)}일**), 금: {gold_note}",
        f"- 무작위 창 ≥**{args.mc_years:g}년**(≥{lo}일), **{args.mc_iters}**회, seed={args.mc_seed}, trail={args.trail}",
        "",
        "**Close_sig 및 MA 로직은 전 조합 공통으로 QQQ.**",
        "",
        "## 전체 표 (CSV)",
        "",
        f"`{csv_path.name}` · Top/Bottom 요약:",
        "",
        "### Sortino 평균 Top 25",
        "",
        "| 공격 | 반등 | 방어 | 안전 | Sortino 평균 | Sortino 중앙 |",
        "|------|------|------|------|-------------:|-------------:|",
    ]
    for r in rows[:25]:
        lines.append(
            "| {a} | {b} | {d} | {s} | {m:.4f} | {md:.4f} |".format(
                a=r["agg"],
                b=r["bounce"],
                d=r["defense"],
                s=r["safe"],
                m=float(r["sortino_mean"]),
                md=float(r["sortino_median"]),
            )
        )

    lines.extend(
        [
            "",
            "### Sortino 평균 Bottom 10",
            "",
            "| 공격 | 반등 | 방어 | 안전 | Sortino 평균 | Sortino 중앙 |",
            "|------|------|------|------|-------------:|-------------:|",
        ]
    )
    for r in rows[-10:]:
        lines.append(
            "| {a} | {b} | {d} | {s} | {m:.4f} | {md:.4f} |".format(
                a=r["agg"],
                b=r["bounce"],
                d=r["defense"],
                s=r["safe"],
                m=float(r["sortino_mean"]),
                md=float(r["sortino_median"]),
            )
        )

    lines.extend(["", "## JSON", "", f"`{json_path.name}`", ""])
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    print("=== 완료 ===")
    print(csv_path.resolve())
    print(json_path.resolve())
    print(md_path.resolve())


if __name__ == "__main__":
    main()
