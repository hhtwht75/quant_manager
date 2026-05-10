"""
DMABS: 공격=TQQQ·안전 레그 종가=Gold 고정, 시그널=QQQ,
반등·방어 종가 각각 {TQQQ,QLD,QQQ,Gold}^2 = 16조합 무작위 윈도 MC.

(반등·방어에 Gold 선택 시 해당 레집에서 Close_bounce 또는 Close_defense 에 동일 금 시계열이 들어간다.)

`fsm_mc_suite_fsm3_rmabs2_bench4.py` 와 동일한 교집합·무작위 (start,length) 규약.

  python3 01_CODE/mc_dmabs_bounce_defense_16grid.py \\
      --root 2002-10-01 --mc-years 3 --mc-iters 3000 --mc-seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import product
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
    MA_WINDOW,
    _core_metrics,
    _distribution_block,
    get_or_build_ma200,
    mc_min_trading_days,
    run_fsm_backtest,
    _align_five,
)

OUT_DIR_DEFAULT = _ROOT / "03_RESULT" / "sensitivity"

LEGS_ORDER = ["TQQQ", "QLD", "QQQ", "Gold"]
METRICS = ("cagr", "mdd", "sharpe", "sortino", "ulcer")


def _leg_df(
    which: str,
    *,
    sg: pd.DataFrame,
    ql: pd.DataFrame,
    tg: pd.DataFrame,
    gld: pd.DataFrame,
) -> pd.DataFrame:
    up = which.strip().upper()
    if up == "QQQ":
        return sg
    if up == "QLD":
        return ql
    if up == "TQQQ":
        return tg
    if up == "GOLD":
        return gld
    raise ValueError(which)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="DMABS Gold안전·TQQQ공격: 반등×방어 ∈ {TQQQ,QLD,QQQ,Gold}^2 MC16 (시그 QQ)"
    )
    ap.add_argument("--root", default="2002-10-01")
    ap.add_argument("--trail", type=float, default=DEFAULT_TRAIL)
    ap.add_argument("--mc-years", type=float, default=3.0)
    ap.add_argument("--mc-iters", type=int, default=3000)
    ap.add_argument("--mc-seed", type=int, default=42)
    ap.add_argument("--out-dir", default="", help="비우면 03_RESULT/sensitivity")
    args = ap.parse_args()

    out_root = Path(args.out_dir) if args.out_dir.strip() else OUT_DIR_DEFAULT
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

    n_combo = len(LEGS_ORDER) ** 2
    combos = list(product(LEGS_ORDER, LEGS_ORDER))
    nav_full: dict[tuple[str, str], pd.Series] = {}

    print(f"전구간 NAV {n_combo}조합 계산 중 (DMABS·safe=Gold·agg=TQQQ) …")
    for b_leg, d_leg in combos:
        bdf = _leg_df(b_leg, sg=sg_i, ql=ql_i, tg=tg_i, gld=g_i)
        df_df = _leg_df(d_leg, sg=sg_i, ql=ql_i, tg=tg_i, gld=g_i)
        aligned = _align_five(sg_i, g_i, bdf, df_df, tg_i)
        nav, _ev = run_fsm_backtest(
            aligned,
            mav,
            trail_stop=args.trail,
            initial_capital=CAPITAL_START,
            use_safe_ma_rule=True,
            stress_bleed_mode="MA5_CROSS_MA120_FULL",
        )
        nav_full[(b_leg, d_leg)] = nav.astype(float)

    n_dates = len(ix)
    rng = np.random.default_rng(args.mc_seed)

    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    print(f"무작위 윈도 MC: 최소 거래연 {args.mc_years:g}, 시행 {args.mc_iters}, seed={args.mc_seed}")
    for _ in range(args.mc_iters):
        s_pos = int(rng.integers(0, n_dates - lo + 1))
        l_max = n_dates - s_pos
        l_win = int(rng.integers(lo, l_max + 1))
        for b_leg, d_leg in combos:
            sub = nav_full[(b_leg, d_leg)].iloc[s_pos : s_pos + l_win]
            m = _core_metrics(sub)
            for mk in METRICS:
                buckets[(b_leg, d_leg, mk)].append(float(m[mk]))

    table_rows: list[dict[str, object]] = []
    for b_leg, d_leg in combos:
        row: dict[str, object] = {"bounce_leg": b_leg, "defense_leg": d_leg}
        for mk in METRICS:
            vals = np.array(buckets[(b_leg, d_leg, mk)])
            blk = _distribution_block(pd.Series(vals))
            row[f"{mk}_mean"] = blk["mean"]
            row[f"{mk}_median"] = blk["median"]
        table_rows.append(row)

    fx0 = pd.Timestamp(ix[0]).strftime("%Y%m%d")
    fx1 = pd.Timestamp(ix[-1]).strftime("%Y%m%d")
    stem = (
        f"mc_dmabs_gold_aggTQQ_bouncedef16grid_{fx0}_{fx1}_"
        f"{args.mc_years:g}yr_n{args.mc_iters}_s{args.mc_seed}"
    )
    json_path = out_root / f"{stem}_summary.json"

    sorted_rows = sorted(table_rows, key=lambda r: -float(r["cagr_mean"]))

    distro_block: dict[str, dict[str, dict[str, float]]] = {}
    for b_leg, d_leg in combos:
        key = f"BOUNCE_{b_leg}_DEFENSE_{d_leg}"
        distro_block[key] = {}
        for mk in METRICS:
            vals = pd.Series(buckets[(b_leg, d_leg, mk)])
            distro_block[key][mk] = _distribution_block(vals)

    meta = {
        "description": (
            "Close_sig=QQQ, Close_safe=Gold, Close_agg=TQQQ DMABS MA5>M120; "
            "Close_bounce·Close_defense ∈ {TQQQ,QLD,QQQ,Gold} ordered 16 combos"
        ),
        "gold_series": gold_note,
        "ma200_csv": str(csv_ma.resolve()),
        "--root": str(root_ts.date()),
        "index_first_last": {"first": str(ix[0].date()), "last": str(ix[-1].date()), "days": len(ix)},
        "trail": float(args.trail),
        "mc_years": float(args.mc_years),
        "mc_min_trading_days": int(lo),
        "mc_iters": int(args.mc_iters),
        "mc_seed": int(args.mc_seed),
        "leg_choices": list(LEGS_ORDER),
        "n_combos": n_combo,
    }

    blob = {"meta": meta, "distribution_by_combo_metric": distro_block}
    json_path.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Monte Carlo: DMABS (Gold 안전 · TQQQ 공격) 반등×방어 16종",
        "",
        f"- 교집합: `{ix[0].date()}` ~ `{ix[-1].date()}` (**{len(ix)}일**) — 금: {gold_note}",
        (
            f"- 설정: 무작위 윈도 최소 **{args.mc_years:g}년**(≥{lo}거래일), "
            f"시행 **{args.mc_iters}**회, seed={args.mc_seed}, trail={args.trail}"
        ),
        "",
        "## 지표 평균·중앙 (MC 무작위 창별 1값 → 분포 통계)",
        "",
        "| 반등(bounce) | 방어(defense) | CAGR 평균 | CAGR 중앙 | MDD 평균 | MDD 중앙 | Sharpe 평균 | Sharpe 중앙 | Sortino 평균 | Sortino 중앙 | Ulcer 평균 | Ulcer 중앙 |",
        "|---------------|---------------|----------:|----------:|----------:|----------:|-------------:|-----------:|-------------:|------------:|----------:|-----------:|",
    ]

    for r in sorted_rows:
        lines.append(
            "| {b} | {d} | {c_m:.2f}% | {c_md:.2f}% | {m_m:.2f}% | {m_md:.2f}% | "
            "{s_m:.3f} | {s_md:.3f} | {so_m:.3f} | {so_md:.3f} | {u_m:.2f} | {u_md:.2f} |".format(
                b=r["bounce_leg"],
                d=r["defense_leg"],
                c_m=float(r["cagr_mean"]) * 100,
                c_md=float(r["cagr_median"]) * 100,
                m_m=float(r["mdd_mean"]) * 100,
                m_md=float(r["mdd_median"]) * 100,
                s_m=float(r["sharpe_mean"]),
                s_md=float(r["sharpe_median"]),
                so_m=float(r["sortino_mean"]),
                so_md=float(r["sortino_median"]),
                u_m=float(r["ulcer_mean"]),
                u_md=float(r["ulcer_median"]),
            )
        )

    lines.extend(
        [
            "",
            "## JSON",
            "",
            f"`{json_path.name}`",
            "",
        ]
    )
    md_path = out_root / f"{stem}_READ_ME.md"
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    print("=== 완료 ===")
    print(str(json_path.resolve()))
    print(str(md_path.resolve()))
    print()
    print("\n".join(lines[:-4]))


if __name__ == "__main__":
    main()
