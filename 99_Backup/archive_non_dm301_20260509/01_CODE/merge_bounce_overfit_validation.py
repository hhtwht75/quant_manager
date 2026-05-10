#!/usr/bin/env python3
"""
Merge bounce base / sticky 오버피팅 점검용 일괄 실행.

산출 (기본 ``03_RESULT/sensitivity/``):
  · 블록별 IS/OOS 정적 지표 표 (벤치 4종 포함)
  · 동일 시드로 full / IS-only / OOS-only 무작위 창 MC 요약 + ΔQQQ
  · 롤링 절단 미래구간 full_metrics (워크포워드 스타일)
  · MULT_UP × MULT_DN × trail 근방 민감도 (전구간 + 선택적 OOS 고정 슬라이스)
  · OOS 구간 paired bootstrap vs QQQ B&H
  · Placebo: 안전(Gold) 종가 날짜쌍만 붕괴(순열) — 동일 QQQ·TQQQ·QQQ 경로

고정 홀드아웃 (문서·JSON ``splits`` 참고):
  • cut_2017: IS ~ 2016-12-31 · OOS 2017-01-01~
  • cut_2020: IS ~ 2019-12-31 · OOS 2020-01-01~

  python3 01_CODE/merge_bounce_overfit_validation.py --root 1999-03-10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

import merge_bounce_simple_mc as mbase  # noqa: E402
import merge_bounce_simple_mc_sticky_defense as mb_sticky  # noqa: E402
import merge_bounce_mc_benchmark_table as mbb  # noqa: E402

from backtest_switching import load_extended_daily  # noqa: E402
from evaluation_metrics import full_metrics, paired_bootstrap_compare  # noqa: E402
from fsm_four_asset_strategy import benchmark_panel_for_slice, build_bench_source_for_index  # noqa: E402
from rmabs_gold_simulation import load_gold_series  # noqa: E402

METRICS5 = ("cagr", "mdd", "sharpe", "sortino", "ulcer")

OVERFIT_SPLITS: list[dict[str, str]] = [
    {
        "id": "cut_2017",
        "description": "IS 구간은 표본 시작~2016-12-31(포함), OOS는 2017-01-01(포함)~표본 끝.",
        "is_end_calendar": "2016-12-31",
        "oos_start_calendar": "2017-01-01",
    },
    {
        "id": "cut_2020",
        "description": "IS ~2019-12-31 포함, OOS 2020-01-01~(코로나 이후 장기 OOS).",
        "is_end_calendar": "2019-12-31",
        "oos_start_calendar": "2020-01-01",
    },
]

BENCH_ORDER = ("QQQ_bh", "QLD_bh", "mix_qq50_tqq50_bh", "TQQQ_bh")

SENS_MULT_UP = [1.01, 1.02, 1.03, 1.04, 1.05]
SENS_MULT_DN = [0.96, 0.97, 0.98]
SENS_TRAIL = [0.80, 0.83, 0.85, 0.87, 0.90]


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, float):
        return float(x)
    if x is None or isinstance(x, (str, int, bool)):
        return x
    if isinstance(x, pd.Timestamp):
        return str(x.date())
    return str(x)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _row_static_block(
    *,
    split_id: str,
    block: str,
    segment_label: str,
    seg_ix: pd.DatetimeIndex,
    nav_base: pd.Series,
    nav_sticky: pd.Series,
    bench_source: pd.DataFrame,
    capital: float,
) -> list[dict[str, Any]]:
    if len(seg_ix) < 30:
        return []
    rows: list[dict[str, Any]] = []
    pan = benchmark_panel_for_slice(seg_ix, bench_source, capital)
    for strat_id, nav, label in (
        ("merge_bounce_base", nav_base, "Merge bounce (base)"),
        ("merge_bounce_sticky", nav_sticky, "Merge bounce (sticky DEF)"),
    ):
        m = full_metrics(nav.reindex(seg_ix).dropna())
        row: dict[str, Any] = {
            "split_id": split_id,
            "block": block,
            "segment": segment_label,
            "series": strat_id,
            "label": label,
            "n_days": len(seg_ix),
        }
        row.update({k: float(m[k]) for k in METRICS5})
        rows.append(row)

    for bk in BENCH_ORDER:
        if bk not in pan:
            continue
        m = full_metrics(pan[bk])
        rows.append(
            {
                "split_id": split_id,
                "block": block,
                "segment": segment_label,
                "series": bk,
                "label": bk,
                "n_days": len(seg_ix),
                **{k: float(m[k]) for k in METRICS5},
            }
        )
    return rows


def _rolling_future_metrics(
    ix: pd.DatetimeIndex,
    nav_base: pd.Series,
    nav_sticky: pd.Series,
    *,
    min_tail: int,
    step: int,
) -> list[dict[str, Any]]:
    n = len(ix)
    out: list[dict[str, Any]] = []
    for cut in range(mbase.MA_WIN, n - min_tail + 1, step):
        seg_ix = ix[cut:]
        if len(seg_ix) < min_tail:
            break
        nb = full_metrics(nav_base.loc[seg_ix])
        ns = full_metrics(nav_sticky.loc[seg_ix])
        out.append(
            {
                "cut_index": cut,
                "cut_date": str(ix[cut].date()),
                "tail_days": len(seg_ix),
                "base_cagr": float(nb["cagr"]),
                "base_sortino": float(nb["sortino"]),
                "base_mdd": float(nb["mdd"]),
                "sticky_cagr": float(ns["cagr"]),
                "sticky_sortino": float(ns["sortino"]),
                "sticky_mdd": float(ns["mdd"]),
            }
        )
    return out


def _paired_bootstrap_oos(
    oos_ix: pd.DatetimeIndex,
    nav_strat: pd.Series,
    bench_source: pd.DataFrame,
    capital: float,
    *,
    n_iter: int,
    seed: int,
) -> dict[str, Any]:
    if len(oos_ix) < 80:
        return {"error": "OOS too short"}
    bh = benchmark_panel_for_slice(oos_ix, bench_source, capital)["QQQ_bh"]
    a = pd.concat(
        [
            nav_strat.reindex(oos_ix).pct_change(),
            bh.pct_change(),
        ],
        axis=1,
        join="inner",
    ).dropna()
    a.columns = ["s", "b"]
    if len(a) < 80:
        return {"error": "aligned too short"}
    blob = paired_bootstrap_compare(
        a["s"], a["b"], block_len=60, n_iter=n_iter, seed=seed
    )
    blob.pop("raw", None)
    return blob


def _placebo_safe_shuffle(
    sig_s: pd.Series,
    sser: pd.Series,
    bser: pd.Series,
    dser: pd.Series,
    mav: pd.Series,
    *,
    n_rep: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ix = sig_s.index
    reals = []
    mc_cagr_nav = float(full_metrics(mbase.run_merge_fsm(sig_s, sser, bser, dser, mav))["cagr"])
    place = []
    for r in range(n_rep):
        perm = rng.permutation(len(ix))
        s_fake = pd.Series(sser.iloc[perm].values, index=ix)
        nav_p = mbase.run_merge_fsm(sig_s, s_fake, bser, dser, mav)
        place.append(float(full_metrics(nav_p)["cagr"]))
    pa = np.array(place, dtype=float)
    return {
        "n_rep": n_rep,
        "seed": seed,
        "realized_full_sample_cagr": mc_cagr_nav,
        "placebo_cagr_mean": float(np.mean(pa)),
        "placebo_cagr_std": float(np.std(pa, ddof=1)) if len(pa) > 1 else 0.0,
        "pctile_realized_vs_placebo": float(np.mean(pa <= mc_cagr_nav)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="merge bounce 오버피팅 검증 묶음")
    ap.add_argument("--root", default="1999-03-10")
    ap.add_argument("--trail", type=float, default=mbase.TRAIL_DEFAULT)
    ap.add_argument("--capital", type=float, default=mbase.CAP0)
    ap.add_argument("--bounce", default="TQQQ")
    ap.add_argument("--defense", default="QQQ")
    ap.add_argument("--safe", default="Gold")
    ap.add_argument("--mc-years", type=float, default=3.0)
    ap.add_argument("--mc-iters", type=int, default=3000)
    ap.add_argument("--mc-seed", type=int, default=42)
    ap.add_argument("--bootstrap-iter", type=int, default=800)
    ap.add_argument("--bootstrap-seed", type=int, default=101)
    ap.add_argument("--rolling-step", type=int, default=504)
    ap.add_argument("--rolling-min-tail", type=int, default=252)
    ap.add_argument("--placebo-rep", type=int, default=30)
    ap.add_argument("--placebo-seed", type=int, default=4242)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

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
    lo = mbase.mc_min_days(args.mc_years)
    if len(ix) < lo:
        raise SystemExit(f"표본 거래일 {len(ix)} < MC 최소 {lo}")

    sig_s = sg["Close"].astype(float).reindex(ix)
    mav = mbase.ma200(sig_s).reindex(ix)
    sgx, qlx, tgx, glx = sg.reindex(ix), ql.reindex(ix), tg.reindex(ix), gold_df.reindex(ix)
    bt, dt, st = args.bounce.strip(), args.defense.strip(), args.safe.strip()
    bser = mbase.pick(bt, sgx, qlx, tgx, glx)
    dser = mbase.pick(dt, sgx, qlx, tgx, glx)
    sser = mbase.pick(st, sgx, qlx, tgx, glx)

    nav_base = mbase.run_merge_fsm(sig_s, sser, bser, dser, mav, trail=args.trail, capital=args.capital)
    nav_sticky = mb_sticky.run_merge_fsm(sig_s, sser, bser, dser, mav, trail=args.trail, capital=args.capital)
    bench_full = build_bench_source_for_index(ix)

    fx0 = ix[0].strftime("%Y%m%d")
    fx1 = ix[-1].strftime("%Y%m%d")
    stem = f"merge_bounce_overfit_{pd.Timestamp(args.root.strip()).strftime('%Y%m%d')}_{fx0}_{fx1}"

    static_rows: list[dict[str, Any]] = []
    mc_pool_rows: list[dict[str, Any]] = []

    sm_full, mx_full = mbb.run_benchmark_mc(
        ix,
        nav_base,
        nav_sticky,
        bench_full,
        args.capital,
        mc_years=args.mc_years,
        mc_iters=args.mc_iters,
        mc_seed=args.mc_seed,
        pool_mode="full",
        is_end_calendar=None,
        oos_start_calendar=None,
        collect_vs_qqq_delta=True,
    )
    mc_pool_rows.append(
        {
            "split_id": "global",
            "pool_tag": "full_sample",
            "pool_mode": "full",
            "summary_metrics": sm_full,
            "extras": mx_full,
        }
    )

    bootstrap_blob: dict[str, Any] = {}
    placebo_blob: dict[str, Any] = {}

    for sp in OVERFIT_SPLITS:
        sid = sp["id"]
        is_cal = pd.Timestamp(sp["is_end_calendar"])
        oos_cal = pd.Timestamp(sp["oos_start_calendar"])
        ix_is = ix[ix <= is_cal]
        ix_oos = ix[ix >= oos_cal]

        static_rows.extend(
            _row_static_block(
                split_id=sid,
                block=sid,
                segment_label="IS",
                seg_ix=ix_is,
                nav_base=nav_base,
                nav_sticky=nav_sticky,
                bench_source=bench_full,
                capital=args.capital,
            )
        )
        static_rows.extend(
            _row_static_block(
                split_id=sid,
                block=sid,
                segment_label="OOS",
                seg_ix=ix_oos,
                nav_base=nav_base,
                nav_sticky=nav_sticky,
                bench_source=bench_full,
                capital=args.capital,
            )
        )

        sm_is, mx_is = mbb.run_benchmark_mc(
            ix,
            nav_base,
            nav_sticky,
            bench_full,
            args.capital,
            mc_years=args.mc_years,
            mc_iters=args.mc_iters,
            mc_seed=args.mc_seed,
            pool_mode="is_only",
            is_end_calendar=is_cal,
            oos_start_calendar=None,
            collect_vs_qqq_delta=True,
        )
        mc_pool_rows.append(
            {
                "split_id": sid,
                "pool_tag": f"{sid}_IS_pool",
                "pool_mode": "is_only",
                "summary_metrics": sm_is,
                "extras": mx_is,
            }
        )
        sm_oos, mx_oos = mbb.run_benchmark_mc(
            ix,
            nav_base,
            nav_sticky,
            bench_full,
            args.capital,
            mc_years=args.mc_years,
            mc_iters=args.mc_iters,
            mc_seed=args.mc_seed,
            pool_mode="oos_only",
            is_end_calendar=None,
            oos_start_calendar=oos_cal,
            collect_vs_qqq_delta=True,
        )
        mc_pool_rows.append(
            {
                "split_id": sid,
                "pool_tag": f"{sid}_OOS_pool",
                "pool_mode": "oos_only",
                "summary_metrics": sm_oos,
                "extras": mx_oos,
            }
        )

        key_b = f"{sid}_OOS_vs_QQQ_bh"
        bootstrap_blob[key_b] = {
            "merge_bounce_base": _paired_bootstrap_oos(
                ix_oos, nav_base, bench_full, args.capital,
                n_iter=args.bootstrap_iter, seed=args.bootstrap_seed,
            ),
            "merge_bounce_sticky": _paired_bootstrap_oos(
                ix_oos, nav_sticky, bench_full, args.capital,
                n_iter=args.bootstrap_iter, seed=args.bootstrap_seed + 1,
            ),
        }

    placebo_blob["base_shuffle_safe_daily"] = _placebo_safe_shuffle(
        sig_s, sser, bser, dser, mav,
        n_rep=args.placebo_rep, seed=args.placebo_seed,
    )

    rolling_blob = _rolling_future_metrics(
        ix, nav_base, nav_sticky,
        min_tail=args.rolling_min_tail,
        step=args.rolling_step,
    )

    sens_rows: list[dict[str, Any]] = []
    for mu in SENS_MULT_UP:
        for md in SENS_MULT_DN:
            for tr in SENS_TRAIL:
                nb = mbase.run_merge_fsm(
                    sig_s, sser, bser, dser, mav,
                    trail=tr, capital=args.capital, mult_dn=md, mult_up=mu,
                )
                ns = mb_sticky.run_merge_fsm(
                    sig_s, sser, bser, dser, mav,
                    trail=tr, capital=args.capital, mult_dn=md, mult_up=mu,
                )
                fb = full_metrics(nb)
                fs = full_metrics(ns)
                sens_rows.append(
                    {
                        "scope": "full_sample",
                        "mult_up": mu,
                        "mult_dn": md,
                        "trail": tr,
                        "base_cagr": float(fb["cagr"]),
                        "base_sortino": float(fb["sortino"]),
                        "sticky_cagr": float(fs["cagr"]),
                        "sticky_sortino": float(fs["sortino"]),
                    }
                )

    split2020 = OVERFIT_SPLITS[1]
    oos_ix_only = ix[ix >= pd.Timestamp(split2020["oos_start_calendar"])]
    for mu in SENS_MULT_UP:
        for md in SENS_MULT_DN:
            for tr in SENS_TRAIL:
                nb = mbase.run_merge_fsm(
                    sig_s, sser, bser, dser, mav,
                    trail=tr, capital=args.capital, mult_dn=md, mult_up=mu,
                )
                ns = mb_sticky.run_merge_fsm(
                    sig_s, sser, bser, dser, mav,
                    trail=tr, capital=args.capital, mult_dn=md, mult_up=mu,
                )
                fb = full_metrics(nb.loc[oos_ix_only])
                fs = full_metrics(ns.loc[oos_ix_only])
                sens_rows.append(
                    {
                        "scope": "OOS_cut_2020",
                        "mult_up": mu,
                        "mult_dn": md,
                        "trail": tr,
                        "base_cagr": float(fb["cagr"]),
                        "base_sortino": float(fb["sortino"]),
                        "sticky_cagr": float(fs["cagr"]),
                        "sticky_sortino": float(fs["sortino"]),
                    }
                )

    pd.DataFrame(static_rows).to_csv(out_dir / f"{stem}_block_metrics.csv", index=False)

    mc_flat: list[dict[str, Any]] = []
    for mr in mc_pool_rows:
        sm = mr["summary_metrics"]
        ex = mr["extras"]
        for series_key in (
            "merge_bounce_simple_mc",
            "merge_bounce_sticky_defense",
            "QQQ_bh",
        ):
            row = {
                "split_id": mr["split_id"],
                "pool_mode": mr["pool_mode"],
                "series": series_key,
                "mean_sortino": sm[series_key]["sortino"]["mean"],
                "mean_cagr": sm[series_key]["cagr"]["mean"],
                "delta_sortino_vs_qqq_mean": (
                    float(ex["delta_vs_qqq_bh"][series_key]["sortino"]["mean"])
                    if series_key in ("merge_bounce_simple_mc", "merge_bounce_sticky_defense")
                    else None
                ),
            }
            mc_flat.append(row)
    pd.DataFrame(mc_flat).to_csv(out_dir / f"{stem}_mc_pool_summary.csv", index=False)

    pd.DataFrame(rolling_blob).to_csv(out_dir / f"{stem}_rolling_future.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(out_dir / f"{stem}_param_sweep.csv", index=False)

    consolidated: dict[str, Any] = {
        "documentation": (
            "IS/OOS 날짜는 스크립트 상수 OVERFIT_SPLITS 및 출력 splits 필드 참고."
        ),
        "splits": OVERFIT_SPLITS,
        "gold_series_note": gold_note,
        "root": args.root,
        "trail_default_run": args.trail,
        "index": {"start": str(ix[0].date()), "end": str(ix[-1].date()), "days": len(ix)},
        "static_block_metrics_csv": str((out_dir / f"{stem}_block_metrics.csv").resolve()),
        "mc_pool_flat_csv": str((out_dir / f"{stem}_mc_pool_summary.csv").resolve()),
        "rolling_future_csv": str((out_dir / f"{stem}_rolling_future.csv").resolve()),
        "param_sweep_csv": str((out_dir / f"{stem}_param_sweep.csv").resolve()),
        "mc_pools_nested": mc_pool_rows,
        "paired_bootstrap_OOS": bootstrap_blob,
        "placebo": placebo_blob,
    }

    (out_dir / f"{stem}.json").write_text(
        json.dumps(_jsonable(consolidated), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (out_dir / f"{stem}_bootstrap.json").write_text(
        json.dumps(_jsonable(bootstrap_blob), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md_mc = ["# Merge bounce 오버피팅 검증 요약 MC (동일 시드·동일 규격)", ""]
    for mr in mc_pool_rows:
        sm = mr["summary_metrics"]
        md_mc.append(
            f"## split={mr['split_id']} pool={mr['pool_mode']}"
        )
        md_mc.extend(
            [
                "| series | Sortino평균 | CAGR평균 |",
                "|---|---:|---:|",
                f"| base | {_fmt_pct(sm['merge_bounce_simple_mc']['sortino']['mean'])} | {_fmt_pct(sm['merge_bounce_simple_mc']['cagr']['mean'])} |",
                f"| sticky | {_fmt_pct(sm['merge_bounce_sticky_defense']['sortino']['mean'])} | {_fmt_pct(sm['merge_bounce_sticky_defense']['cagr']['mean'])} |",
                f"| QQQ B&H | {_fmt_pct(sm['QQQ_bh']['sortino']['mean'])} | {_fmt_pct(sm['QQQ_bh']['cagr']['mean'])} |",
                "",
            ]
        )

    md_static = ["# 블록 정적 지표 IS / OOS", ""]
    df_s = pd.DataFrame(static_rows)
    for sid in df_s["split_id"].unique():
        md_static.append(f"## {sid}")
        sub = df_s[df_s["split_id"] == sid]
        for seg in ["IS", "OOS"]:
            ss = sub[sub["segment"] == seg]
            if ss.empty:
                continue
            md_static.append(f"### {seg}")
            cols = ["label", "cagr", "mdd", "sharpe", "sortino", "ulcer"]
            md_static.append("| " + " | ".join(cols) + " |")
            md_static.append("| " + " | ".join(["---"] * len(cols)) + " |")
            for _, r in ss.iterrows():
                md_static.append(
                    "| "
                    + " | ".join(
                        [
                            str(r["label"]),
                            _fmt_pct(float(r["cagr"])),
                            _fmt_pct(float(r["mdd"])),
                            f"{float(r['sharpe']):.3f}",
                            f"{float(r['sortino']):.3f}",
                            f"{float(r['ulcer']):.4f}",
                        ]
                    )
                    + " |"
                )
            md_static.append("")

    (out_dir / f"{stem}_mc_table.md").write_text("\n".join(md_mc) + "\n", encoding="utf-8")
    (out_dir / f"{stem}_block_table.md").write_text("\n".join(md_static) + "\n", encoding="utf-8")

    print("=== merge_bounce_overfit_validation 완료 ===")
    print(out_dir / f"{stem}.json")


if __name__ == "__main__":
    main()
