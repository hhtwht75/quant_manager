#!/usr/bin/env python3
"""
merge_bounce_simple_mc · merge_bounce_simple_mc_sticky_defense 무작위 윈도 MC와
벤치 4종(QQQ B&H, QLD B&H, QQQ/TQQQ 50-50 B&H, TQQQ B&H)을 **동일 창**에서 비교.

각 시행마다 한 구간의 CAGR·MDD·Sharpe·Sortino·Ulcer을 계산하고, 시행 간 **평균**을 표로 저장.

  python3 01_CODE/merge_bounce_mc_benchmark_table.py \\
    --root 1999-03-10 --mc-years 3 --mc-iters 3000 --mc-seed 42

  python3 01_CODE/merge_bounce_mc_benchmark_table.py \\
    --root 2002-10-01 --mc-years 3 --mc-iters 3000 --mc-seed 42

  # OOS 창만 무작위 창 (예: 2020-01-01 이후)·QQQ 대비 Δ 포함
  python3 01_CODE/merge_bounce_mc_benchmark_table.py \\
    --pool-mode oos_only --split-oos-start 2020-01-01 --collect-delta-qqq
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

import merge_bounce_simple_mc as mbase  # noqa: E402
import merge_bounce_simple_mc_sticky_defense as mb_sticky  # noqa: E402

from backtest_switching import load_extended_daily  # noqa: E402
from evaluation_metrics import full_metrics  # noqa: E402
from fsm_four_asset_strategy import (  # noqa: E402
    benchmark_panel_for_slice,
    build_bench_source_for_index,
)
from rmabs_gold_simulation import load_gold_series  # noqa: E402

METRIC_KEYS = ("cagr", "mdd", "sharpe", "sortino", "ulcer")

BENCH_LABELS = {
    "QQQ_bh": "QQQ B&H",
    "QLD_bh": "QLD B&H",
    "mix_qq50_tqq50_bh": "QQQ/TQQQ 50-50 B&H",
    "TQQQ_bh": "TQQQ B&H",
}


def _dist_vec(a: np.ndarray) -> dict[str, float]:
    a = a.astype(float)
    if len(a) < 1:
        return {"mean": 0.0, "std": 0.0, "median": 0.0}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
    }


def _fmt_pct(x: float, nd: int = 2) -> str:
    return f"{x * 100:.{nd}f}%"


def _fmt_float(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


def _stem_pool_suffix(mode: str, is_cal: pd.Timestamp | None, oos_cal: pd.Timestamp | None) -> str:
    """full 이면 빈 문자열(레거시 파일명 호환). 그 외 풀·날짜 태그."""
    if mode == "full":
        return ""
    bits = [mode]
    if is_cal is not None:
        bits.append(f"isEnd{is_cal.strftime('%Y%m%d')}")
    if oos_cal is not None:
        bits.append(f"oosStart{oos_cal.strftime('%Y%m%d')}")
    return "_" + "_".join(bits)


PoolMode = Literal["full", "is_only", "oos_only"]


def sample_mc_window_slice(
    rng: np.random.Generator,
    n: int,
    lo: int,
    pool_mode: PoolMode,
    *,
    is_end_idx: int | None,
    oos_start_idx: int | None,
) -> slice:
    """Return slice(s0, s0+win) with win in [lo, …] respecting pool_mode."""
    if pool_mode == "full":
        s0 = int(rng.integers(0, n - lo + 1))
        win_max = n - s0
        win = int(rng.integers(lo, win_max + 1))
        return slice(s0, s0 + win)
    if pool_mode == "is_only":
        if is_end_idx is None:
            raise ValueError("is_only requires is_end_idx")
        s0_max = min(n - lo, is_end_idx - lo + 1)
        if s0_max < 0:
            raise ValueError(f"IS pool empty: is_end_idx={is_end_idx} lo={lo} n={n}")
        s0 = int(rng.integers(0, s0_max + 1))
        win_max = min(n - s0, is_end_idx - s0 + 1)
        if win_max < lo:
            raise ValueError("IS segment too short vs mc_min_days")
        win = int(rng.integers(lo, win_max + 1))
        return slice(s0, s0 + win)
    if pool_mode == "oos_only":
        if oos_start_idx is None:
            raise ValueError("oos_only requires oos_start_idx")
        min_s0 = max(0, oos_start_idx)
        max_s0 = n - lo
        if min_s0 > max_s0:
            raise ValueError(f"OOS pool empty: oos_start_idx={oos_start_idx} lo={lo} n={n}")
        s0 = int(rng.integers(min_s0, max_s0 + 1))
        win_max = n - s0
        win = int(rng.integers(lo, win_max + 1))
        return slice(s0, s0 + win)
    raise ValueError(pool_mode)


def run_benchmark_mc(
    ix: pd.DatetimeIndex,
    nav_base: pd.Series,
    nav_sticky: pd.Series,
    bench_source: pd.DataFrame,
    capital: float,
    *,
    mc_years: float,
    mc_iters: int,
    mc_seed: int,
    pool_mode: PoolMode = "full",
    is_end_calendar: pd.Timestamp | None = None,
    oos_start_calendar: pd.Timestamp | None = None,
    collect_vs_qqq_delta: bool = False,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, object]]:
    """
    무작위 창별 full_metrics 평균/표준편차/중앙값 반환.

    NAV는 반드시 **전역 시계열**에서 계산된 뒤, 창만 잘라 쓸 것 (풀모드별 가산 위치 일치).
    """
    lo = mbase.mc_min_days(mc_years)
    n = len(ix)
    is_end_idx: int | None = None
    oos_start_idx: int | None = None
    if is_end_calendar is not None:
        is_end_idx = int(ix.searchsorted(is_end_calendar, side="right")) - 1
    if oos_start_calendar is not None:
        oos_start_idx = int(ix.searchsorted(oos_start_calendar, side="left"))

    row_order_keys = (
        ["merge_bounce_simple_mc", "merge_bounce_sticky_defense"]
        + list(BENCH_LABELS.keys())
    )
    accum: dict[str, dict[str, list[float]]] = {k: {m: [] for m in METRIC_KEYS} for k in row_order_keys}
    deltas_b: dict[str, list[float]] = {"delta_sortino": [], "delta_cagr": []}
    deltas_s: dict[str, list[float]] = {"delta_sortino": [], "delta_cagr": []}

    rng = np.random.default_rng(mc_seed)

    for _ in range(mc_iters):
        sl = sample_mc_window_slice(
            rng, n, lo, pool_mode,
            is_end_idx=is_end_idx,
            oos_start_idx=oos_start_idx,
        )
        ix_win = ix[sl]

        series_nav: dict[str, pd.Series] = {
            "merge_bounce_simple_mc": nav_base,
            "merge_bounce_sticky_defense": nav_sticky,
        }
        panel = benchmark_panel_for_slice(ix_win, bench_source, capital)
        for bk, nav_b in panel.items():
            series_nav[bk] = nav_b

        m_store: dict[str, dict[str, float]] = {}
        for key, nav in series_nav.items():
            m_store[key] = full_metrics(nav.loc[ix_win])
            for k in METRIC_KEYS:
                accum[key][k].append(float(m_store[key][k]))

        if collect_vs_qqq_delta:
            qb = m_store["QQQ_bh"]["sortino"]
            qc = m_store["QQQ_bh"]["cagr"]
            deltas_b["delta_sortino"].append(float(m_store["merge_bounce_simple_mc"]["sortino"] - qb))
            deltas_b["delta_cagr"].append(float(m_store["merge_bounce_simple_mc"]["cagr"] - qc))
            deltas_s["delta_sortino"].append(float(m_store["merge_bounce_sticky_defense"]["sortino"] - qb))
            deltas_s["delta_cagr"].append(float(m_store["merge_bounce_sticky_defense"]["cagr"] - qc))

    summary_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for key in row_order_keys:
        summary_metrics[key] = {}
        for mk in METRIC_KEYS:
            summary_metrics[key][mk] = _dist_vec(np.array(accum[key][mk], dtype=float))

    extra: dict[str, object] = {
        "mc_min_days": lo,
        "pool_mode": pool_mode,
        "is_end_calendar": str(is_end_calendar.date()) if is_end_calendar is not None else None,
        "oos_start_calendar": str(oos_start_calendar.date()) if oos_start_calendar is not None else None,
    }
    if collect_vs_qqq_delta:

        def _delta_summary(vals: list[float]) -> dict[str, float]:
            a = np.array(vals, dtype=float)
            return {
                "mean": float(np.mean(a)),
                "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                "median": float(np.median(a)),
                "pct_positive_sortino_or_cagr": float(np.mean(a > 0)),
            }

        extra["delta_vs_qqq_bh"] = {
            "merge_bounce_simple_mc": {
                "sortino": _delta_summary(deltas_b["delta_sortino"]),
                "cagr": _delta_summary(deltas_b["delta_cagr"]),
            },
            "merge_bounce_sticky_defense": {
                "sortino": _delta_summary(deltas_s["delta_sortino"]),
                "cagr": _delta_summary(deltas_s["delta_cagr"]),
            },
        }

    return summary_metrics, extra


def main() -> None:
    ap = argparse.ArgumentParser(description="merge_bounce MC + 벤치4 지표 표")
    ap.add_argument("--bounce", default="TQQQ")
    ap.add_argument("--defense", default="QQQ")
    ap.add_argument("--safe", default="Gold")
    ap.add_argument("--root", default="1999-03-10")
    ap.add_argument("--trail", type=float, default=mbase.TRAIL_DEFAULT)
    ap.add_argument("--mc-years", type=float, default=3.0)
    ap.add_argument("--mc-iters", type=int, default=3000)
    ap.add_argument("--mc-seed", type=int, default=42)
    ap.add_argument("--capital", type=float, default=mbase.CAP0)
    ap.add_argument(
        "--pool-mode",
        choices=("full", "is_only", "oos_only"),
        default="full",
        help="full=전역; is_only=is_end까지 index만 무작위 창; oos_only=oos_start 이후만",
    )
    ap.add_argument(
        "--split-is-end",
        default="",
        help='IS 종료 포함일 (예 2019-12-31). pool is_only 에서 무작위 창의 끝이 이 날짜 이하여야 함',
    )
    ap.add_argument(
        "--split-oos-start",
        default="",
        help='OOS 첫 포함일 (예 2020-01-01). pool oos_only 에서 무작위 창이 이후만 사용',
    )
    ap.add_argument(
        "--collect-delta-qqq",
        action="store_true",
        help="각 창별 Sortino·CAGR 과 QQQ B&H 의 차이 요약 저장",
    )
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    bt, dt, st = args.bounce.strip(), args.defense.strip(), args.safe.strip()
    out_dir = Path(args.out_dir).resolve() if args.out_dir.strip() else _ROOT / "03_RESULT" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    root_ts = pd.Timestamp(args.root.strip())
    sg = load_extended_daily("QQQ")
    ql = load_extended_daily("QLD")
    tg = load_extended_daily("TQQQ")
    ix0 = sg.index.intersection(ql.index).intersection(tg.index).sort_values()
    ix0 = ix0[ix0 >= root_ts]
    gold_df, gold_note = load_gold_series(ix0[0], ix0[-1])
    ix = ix0.intersection(gold_df.index).sort_values()
    if len(ix) < mbase.MA_WIN:
        raise SystemExit(f"일수 {len(ix)} < MA{mbase.MA_WIN}")

    lo = mbase.mc_min_days(args.mc_years)
    if len(ix) < lo:
        raise SystemExit(f"전체 {len(ix)}일 < 최소 창 {lo}일")

    sig_s = sg["Close"].astype(float).reindex(ix)
    mav = mbase.ma200(sig_s).reindex(ix)

    sgx, qlx, tgx, glx = sg.reindex(ix), ql.reindex(ix), tg.reindex(ix), gold_df.reindex(ix)
    bser = mbase.pick(bt, sgx, qlx, tgx, glx)
    dser = mbase.pick(dt, sgx, qlx, tgx, glx)
    sser = mbase.pick(st, sgx, qlx, tgx, glx)

    nav_base = mbase.run_merge_fsm(sig_s, sser, bser, dser, mav, trail=args.trail, capital=args.capital)
    nav_sticky = mb_sticky.run_merge_fsm(sig_s, sser, bser, dser, mav, trail=args.trail, capital=args.capital)
    bench_source = build_bench_source_for_index(ix)

    row_order = [
        ("merge_bounce_simple_mc", "Merge bounce (base)"),
        ("merge_bounce_sticky_defense", "Merge bounce (sticky DEF)"),
        ("QQQ_bh", BENCH_LABELS["QQQ_bh"]),
        ("QLD_bh", BENCH_LABELS["QLD_bh"]),
        ("mix_qq50_tqq50_bh", BENCH_LABELS["mix_qq50_tqq50_bh"]),
        ("TQQQ_bh", BENCH_LABELS["TQQQ_bh"]),
    ]

    is_cal = pd.Timestamp(args.split_is_end.strip()) if args.split_is_end.strip() else None
    oos_cal = pd.Timestamp(args.split_oos_start.strip()) if args.split_oos_start.strip() else None
    if args.pool_mode == "is_only" and is_cal is None:
        raise SystemExit("--pool-mode is_only 는 --split-is-end 가 필요함")
    if args.pool_mode == "oos_only" and oos_cal is None:
        raise SystemExit("--pool-mode oos_only 는 --split-oos-start 가 필요함")

    summary_metrics, mc_extra = run_benchmark_mc(
        ix,
        nav_base,
        nav_sticky,
        bench_source,
        args.capital,
        mc_years=args.mc_years,
        mc_iters=args.mc_iters,
        mc_seed=args.mc_seed,
        pool_mode=args.pool_mode,
        is_end_calendar=is_cal,
        oos_start_calendar=oos_cal,
        collect_vs_qqq_delta=args.collect_delta_qqq,
    )

    n = len(ix)
    fx0 = pd.Timestamp(ix[0]).strftime("%Y%m%d")
    fx1 = pd.Timestamp(ix[-1]).strftime("%Y%m%d")
    tk_b, tk_d, tk_s = bt.upper(), dt.upper(), st.upper()
    pool_tag = _stem_pool_suffix(args.pool_mode, is_cal, oos_cal)
    stem = (
        f"merge_bounce_mc_bench4_{fx0}_{fx1}_{tk_b}_{tk_d}_{tk_s}_"
        f"{args.mc_years:g}yr_n{args.mc_iters}_s{args.mc_seed}"
        f"{pool_tag}"
    )

    pmd = args.pool_mode
    pool_md = (
        f"- **무작위 창 샘플 풀**: `{pmd}`"
        + (f", IS 종료(포함): `{is_cal.date()}`" if is_cal is not None else "")
        + (f", OOS 시작(포함): `{oos_cal.date()}`" if oos_cal is not None else "")
    )
    md_lines = [
        f"# Merge bounce MC vs 벤치 4종 (동일 무작위 창)",
        "",
        f"- **표본 구간**: {ix[0].date()} ~ {ix[-1].date()} (거래일 {n}일), `--root` 상한 `{args.root}`",
        pool_md,
        f"- **MC**: 최소 {args.mc_years:g}년·{args.mc_iters}회, 시드 {args.mc_seed}, trail={args.trail}",
        f"- **표 값**: 각 시행에서 해당 창 NAV로 계산한 지표의 **산술평균** (레짐 규격: SAFE/BOUNCE 103×MA는 상향 돌파)",
        "",
        "| 구분 | CAGR (평균) | MDD (평균) | Sharpe (평균) | Sortino (평균) | Ulcer (평균) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row_key, label in row_order:
        blk = summary_metrics[row_key]
        md_lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _fmt_pct(blk["cagr"]["mean"]),
                    _fmt_pct(blk["mdd"]["mean"]),
                    _fmt_float(blk["sharpe"]["mean"], 3),
                    _fmt_float(blk["sortino"]["mean"], 3),
                    _fmt_float(blk["ulcer"]["mean"], 4),
                ]
            )
            + " |"
        )

    md_p = out_dir / f"{stem}_table.md"
    md_p.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    csv_rows = []
    for row_key, label in row_order:
        blk = summary_metrics[row_key]
        csv_rows.append(
            {
                "name": row_key,
                "label": label,
                **{f"{k}_mean": blk[k]["mean"] for k in METRIC_KEYS},
                **{f"{k}_std": blk[k]["std"] for k in METRIC_KEYS},
            }
        )
    csv_p = out_dir / f"{stem}_table.csv"
    pd.DataFrame(csv_rows).to_csv(csv_p, index=False)

    blob = {
        "meta": {
            "strategy_pair": "merge_bounce_simple_mc vs merge_bounce_simple_mc_sticky_defense",
            "benchmarks": list(BENCH_LABELS.values()),
            "sig": mbase.SIG_TICKER,
            "bounce": tk_b,
            "defense": tk_d,
            "safe": tk_s,
            "gold_series_note": gold_note,
            "root_calendar": str(args.root),
            "index_start": str(ix[0].date()),
            "index_end": str(ix[-1].date()),
            "index_days": n,
            "trail": args.trail,
            "mc_years": args.mc_years,
            "mc_min_days": lo,
            "mc_iters": args.mc_iters,
            "mc_seed": args.mc_seed,
            "pool_mode": args.pool_mode,
            "split_is_end": str(is_cal.date()) if is_cal is not None else None,
            "split_oos_start": str(oos_cal.date()) if oos_cal is not None else None,
            "metrics_note": (
                "Per-iteration: random [start,end); full_metrics on window NAV. "
                "Report mean/median/std across iterations."
            ),
        },
        "mc_pool_detail": mc_extra,
        "by_series": summary_metrics,
    }
    json_p = out_dir / f"{stem}_summary.json"
    json_p.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== 완료 ===")
    print(md_p.resolve())
    print(csv_p.resolve())
    print(json_p.resolve())


if __name__ == "__main__":
    main()
