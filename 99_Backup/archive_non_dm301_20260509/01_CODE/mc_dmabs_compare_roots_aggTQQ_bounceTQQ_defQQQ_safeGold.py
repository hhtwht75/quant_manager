"""
DMABS 고정 레그 MC 비교:

- Close_sig = QQQ · Close_safe = Gold · Close_agg = TQQQ · Close_bounce = TQQQ · Close_defense = QQQ

두 가지 표본 시작일(`--roots`) 각각 교집합 전체에서 무작위 3년(기본)·mc-iters 회 MC.

  python3 01_CODE/mc_dmabs_compare_roots_aggTQQ_bounceTQQ_defQQQ_safeGold.py \\
    --roots 1999-03-10 2002-10-01 --mc-years 3 --mc-iters 3000 --mc-seed 42
"""

from __future__ import annotations

import argparse
import json
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
    MA_WINDOW,
    _align_five,
    _core_metrics,
    _distribution_block,
    get_or_build_ma200,
    mc_min_trading_days,
    run_fsm_backtest,
)

OUT_DEFAULT = _ROOT / "03_RESULT" / "sensitivity"


def run_mc_for_root(
    root_ts: pd.Timestamp,
    *,
    trail: float,
    mc_years: float,
    mc_iters: int,
    mc_seed: int,
) -> tuple[
    dict[str, object],
    dict[str, dict[str, float]],
    dict[str, float],
]:
    sg_full = load_extended_daily("QQQ")
    tg_full = load_extended_daily("TQQQ")
    ma_full, csv_ma = get_or_build_ma200(sg_full, "QQQ", force_rebuild=False)

    ix0_all = sg_full.index.intersection(tg_full.index).sort_values()
    ix0 = ix0_all[ix0_all >= root_ts]
    gold_df, gold_note = load_gold_series(ix0[0], ix0[-1])
    ix = ix0.intersection(gold_df.index).sort_values()

    if len(ix) < MA_WINDOW:
        raise ValueError(f"root {root_ts.date()} 교집합 {len(ix)}일 < MA200")
    lo = mc_min_trading_days(mc_years)
    if len(ix) < lo:
        raise ValueError(f"root {root_ts.date()} 일수 {len(ix)} < 최소 윈도 {lo}")

    sg_i = sg_full.reindex(ix)
    tg_i = tg_full.reindex(ix)
    g_i = gold_df.reindex(ix)
    mav = ma_full.reindex(ix)

    aligned = _align_five(sg_i, g_i, tg_i, sg_i, tg_i)
    nav_full, _ev = run_fsm_backtest(
        aligned,
        mav,
        trail_stop=trail,
        initial_capital=CAPITAL_START,
        use_safe_ma_rule=True,
        stress_bleed_mode="MA5_CROSS_MA120_FULL",
    )
    nav_full = nav_full.astype(float)

    n_dates = len(ix)
    rng = np.random.default_rng(mc_seed)
    metrics = ("cagr", "mdd", "sharpe", "sortino", "ulcer")
    buckets: dict[str, list[float]] = {m: [] for m in metrics}

    for _ in range(mc_iters):
        s_pos = int(rng.integers(0, n_dates - lo + 1))
        l_max = n_dates - s_pos
        l_win = int(rng.integers(lo, l_max + 1))
        sub = nav_full.iloc[s_pos : s_pos + l_win]
        m = _core_metrics(sub)
        for mk in metrics:
            buckets[mk].append(float(m[mk]))

    distro = {mk: _distribution_block(pd.Series(buckets[mk])) for mk in metrics}
    mean_row = {f"{mk}_mean": distro[mk]["mean"] for mk in metrics}
    med_row = {f"{mk}_median": distro[mk]["median"] for mk in metrics}

    meta = {
        "root_requested": str(root_ts.date()),
        "index_first": str(ix[0].date()),
        "index_last": str(ix[-1].date()),
        "trading_days": len(ix),
        "gold_series": gold_note,
        "ma200_csv": str(csv_ma.resolve()),
        "legs": "sig=QQQ safe=Gold agg=TQQQ bounce=TQQQ defense=QQQ · DMABS MA5>M120",
        "trail": float(trail),
        "mc_years": float(mc_years),
        "mc_iters": int(mc_iters),
        "mc_seed": int(mc_seed),
        "mc_min_trading_days": lo,
    }
    return meta, distro, mean_row | med_row


def main() -> None:
    ap = argparse.ArgumentParser(description="DMABS 고정 레그 두 root MC 비교")
    ap.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="예: 1999-03-10 2002-10-01",
    )
    ap.add_argument("--trail", type=float, default=DEFAULT_TRAIL)
    ap.add_argument("--mc-years", type=float, default=3.0)
    ap.add_argument("--mc-iters", type=int, default=3000)
    ap.add_argument("--mc-seed", type=int, default=42)
    ap.add_argument("--out-dir", default="", help="기본 03_RESULT/sensitivity")
    args = ap.parse_args()

    out_root = Path(args.out_dir) if args.out_dir.strip() else OUT_DEFAULT
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, dict[str, object], dict[str, dict[str, float]], dict[str, float]]] = []
    for rs in args.roots:
        root_ts = pd.Timestamp(str(rs).strip())
        meta, distro, flat = run_mc_for_root(
            root_ts,
            trail=args.trail,
            mc_years=args.mc_years,
            mc_iters=args.mc_iters,
            mc_seed=args.mc_seed,
        )
        rows.append((str(root_ts.date()), meta, distro, flat))

    fx_end = rows[-1][1]["index_last"].replace("-", "")
    stem = (
        f"mc_dmabs_aggTQQ_bounceTQQ_defQQQ_safeGold_compare_roots_"
        f"{'_'.join(r[0].replace('-', '') for r in rows)}_to_{fx_end}_"
        f"{args.mc_years:g}yr_n{args.mc_iters}_s{args.mc_seed}"
    )
    json_path = out_root / f"{stem}_summary.json"

    blob = {
        "meta_bundle": {"runs": [r[1] for r in rows]},
        "distribution_by_root": {r[0]: r[2] for r in rows},
        "flat_summary_by_root": {r[0]: r[3] for r in rows},
    }
    json_path.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Monte Carlo 비교 · DMABS (금 안전 / TQQQ 공격·반등 / QQQ 방어)",
        "",
        f"- 무작위 윈도 최소 **{args.mc_years:g}년**, 시행 **{args.mc_iters}**, seed={args.mc_seed}, trail={args.trail}",
        "",
        "| 표본 시작(root) | 거래구간 종료일 | 교집합 일수 | CAGR 평균 | CAGR 중앙 | MDD 평균 | MDD 중앙 | Sharpe 평균 | Sharpe 중앙 | Sortino 평균 | Ulcer 평균 |",
        "|----------------|----------------|-------------|----------:|----------:|----------:|----------:|-------------:|-----------:|-------------:|-----------:|",
    ]
    for rdate, meta, distro, flat in rows:
        lines.append(
            "| `{r}` | {end} | {n} | {cma:.2f}% | {cmd:.2f}% | {mma:.2f}% | {mmd:.2f}% | {sma:.3f} | {smd:.3f} | {so:.3f} | {ua:.2f} |".format(
                r=rdate,
                end=meta["index_last"],
                n=int(meta["trading_days"]),
                cma=flat["cagr_mean"] * 100,
                cmd=flat["cagr_median"] * 100,
                mma=flat["mdd_mean"] * 100,
                mmd=flat["mdd_median"] * 100,
                sma=flat["sharpe_mean"],
                smd=flat["sharpe_median"],
                so=flat["sortino_mean"],
                ua=flat["ulcer_mean"],
            )
        )
    lines.extend(["", "## JSON", "", f"`{json_path.name}`", ""])
    md_path = out_root / f"{stem}_READ_ME.md"
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    print("=== 완료 ===")
    print(json_path.resolve())
    print(md_path.resolve())
    print()
    print("\n".join(lines[:-4]))


if __name__ == "__main__":
    main()
