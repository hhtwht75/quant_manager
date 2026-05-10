#!/usr/bin/env python3
"""
Merge Bounce 디폴트 근방 전공정 파라미터 스윕 → 플래토 vs 피크(운 좋은 점) 정성·정량 요약.

스윕 대상 (QQQ 종가 고정 레그 규격):
  · 장기 추세 MA: 롤링 일수 ma_win (규칙4·MULT_UP 교차 등)
  · 단·장 스트레스 MA: ma_fast, ma_slow (MA_slow 대비 MA_fast 상향 교차)
  · 규칙4 하방 배수 mult_dn × MA(ma_win), 상방 돌파 mult_up × MA(ma_win)
  · 반등 트레일 trail (고점 대비 비율)

  python3 01_CODE/merge_bounce_param_plateau_sweep.py --root 1999-03-10
  python3 01_CODE/merge_bounce_param_plateau_sweep.py --jobs 1   # 단일 프로세스 디버그용
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

import merge_bounce_simple_mc as mbase  # noqa: E402
import merge_bounce_simple_mc_sticky_defense as mb_sticky  # noqa: E402

from backtest_switching import load_extended_daily  # noqa: E402
from evaluation_metrics import full_metrics  # noqa: E402
from rmabs_gold_simulation import load_gold_series  # noqa: E402

BASELINE_WIN = int(mbase.MA_WIN)
BASELINE_FAST = int(mbase.MA_FAST)
BASELINE_SLOW = int(mbase.MA_SLOW)
BASELINE_DN = float(mbase.MULT_DN)
BASELINE_UP = float(mbase.MULT_UP)
BASELINE_TRAIL = float(mbase.TRAIL_DEFAULT)

MA_WIN_GRID = [160, 180, 190, 200, 210, 220]
MA_PAIR_GRID: list[tuple[int, int]] = [
    (3, 100),
    (5, 100),
    (5, 110),
    (5, 120),
    (5, 130),
    (7, 110),
    (7, 120),
    (10, 120),
    (10, 140),
]

MULT_DN_GRID = [0.955, 0.96, 0.965, 0.97, 0.975, 0.98]
MULT_UP_GRID = [1.01, 1.02, 1.03, 1.04, 1.05]
TRAIL_GRID = [0.78, 0.80, 0.82, 0.85, 0.88, 0.90]

_POOL_DATA: dict[str, object] = {}


@dataclass(frozen=True)
class ParamRow:
    ma_win: int
    ma_fast: int
    ma_slow: int
    mult_dn: float
    mult_up: float
    trail: float


def iter_param_grid(ma_win_candidates: list[int]) -> Iterator[ParamRow]:
    for w, (mf, ms), dn, du, tr in product(
        ma_win_candidates,
        MA_PAIR_GRID,
        MULT_DN_GRID,
        MULT_UP_GRID,
        TRAIL_GRID,
    ):
        if mf >= ms:
            continue
        if mf >= w or ms >= w:
            continue
        yield ParamRow(ma_win=w, ma_fast=mf, ma_slow=ms, mult_dn=float(dn), mult_up=float(du), trail=float(tr))


def _grid_neighbor_values(unique_sorted: np.ndarray, cur: float, *, int_like: bool) -> list[float]:
    if int_like:
        u_int = np.sort(np.unique(unique_sorted.astype(int)))
        kk = np.where(u_int == int(round(cur)))[0]
        if len(kk) == 0:
            return []
        k = int(kk[0])
        out_int: list[float] = []
        if k > 0:
            out_int.append(float(u_int[k - 1]))
        if k + 1 < len(u_int):
            out_int.append(float(u_int[k + 1]))
        return out_int
    u = np.sort(np.unique(unique_sorted.astype(float)))
    eq = np.isclose(u, float(cur), rtol=1e-9, atol=1e-12)
    idxs = np.where(eq)[0]
    out: list[float] = []
    if len(idxs):
        k = int(idxs[0])
        if k > 0:
            out.append(float(u[k - 1]))
        if k + 1 < len(u):
            out.append(float(u[k + 1]))
        return out
    j = int(np.searchsorted(u, float(cur)))
    if j > 0:
        out.append(float(u[j - 1]))
    if j < len(u):
        out.append(float(u[j]))
    return out


def _sortino_empirical_percentile(series: pd.Series, x: float) -> float:
    """Sortino 클수록 상위 분위."""
    v = series.astype(float).values
    n = len(v)
    if n == 0:
        return float("nan")
    return float(100.0 * (np.sum(v < x) + 0.5 * np.sum(np.isclose(v, x))) / n)


def one_param_neighbors_df(df_strat: pd.DataFrame, bl: pd.Series) -> pd.DataFrame:
    rows: list[dict] = []

    def base_mask(ma_w: int, mf: int, ms: int, dn: float, du: float, tr: float) -> pd.Series:
        m = pd.Series(True, index=df_strat.index)
        m &= df_strat["ma_win"].astype(int) == ma_w
        m &= df_strat["ma_fast"].astype(int) == mf
        m &= df_strat["ma_slow"].astype(int) == ms
        m &= np.isclose(df_strat["mult_dn"].astype(float).values, float(dn), rtol=1e-9)
        m &= np.isclose(df_strat["mult_up"].astype(float).values, float(du), rtol=1e-9)
        m &= np.isclose(df_strat["trail"].astype(float).values, float(tr), rtol=1e-9)
        return m

    bw = int(bl["ma_win"])
    bf = int(bl["ma_fast"])
    bs = int(bl["ma_slow"])
    bdn, bup, btr = float(bl["mult_dn"]), float(bl["mult_up"]), float(bl["trail"])

    axes_meta = (
        ("ma_win", df_strat["ma_win"].values.astype(int), float(bl["ma_win"]), True),
        ("ma_fast", df_strat["ma_fast"].values.astype(int), float(bl["ma_fast"]), True),
        ("ma_slow", df_strat["ma_slow"].values.astype(int), float(bl["ma_slow"]), True),
        ("mult_dn", df_strat["mult_dn"].values.astype(float), float(bl["mult_dn"]), False),
        ("mult_up", df_strat["mult_up"].values.astype(float), float(bl["mult_up"]), False),
        ("trail", df_strat["trail"].values.astype(float), float(bl["trail"]), False),
    )

    for name, colvals, cur, int_like in axes_meta:
        neigh = _grid_neighbor_values(colvals, cur, int_like=int_like)
        pooled: list[float] = []
        for nv in neigh:
            nm = base_mask(bw, bf, bs, bdn, bup, btr)
            if name == "ma_win":
                nm = base_mask(int(round(nv)), bf, bs, bdn, bup, btr)
            elif name == "ma_fast":
                nm = base_mask(bw, int(round(nv)), bs, bdn, bup, btr)
            elif name == "ma_slow":
                nm = base_mask(bw, bf, int(round(nv)), bdn, bup, btr)
            elif name == "mult_dn":
                nm = base_mask(bw, bf, bs, float(nv), bup, btr)
            elif name == "mult_up":
                nm = base_mask(bw, bf, bs, bdn, float(nv), btr)
            elif name == "trail":
                nm = base_mask(bw, bf, bs, bdn, bup, float(nv))
            hit = df_strat.loc[nm, "sortino"]
            if len(hit):
                pooled.extend(hit.astype(float).tolist())
        if not pooled:
            continue
        arr = np.array(pooled, dtype=float)
        rows.append(
            {
                "axis": name,
                "n_neighbor_cells": len(pooled),
                "neighbor_sortino_mean": float(arr.mean()),
                "neighbor_sortino_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "delta_vs_baseline_mean": float(arr.mean()) - float(bl["sortino"]),
            }
        )
    return pd.DataFrame(rows)


def _mp_worker_init(
    sig_s: pd.Series, sser: pd.Series, bser: pd.Series, dser: pd.Series, capital: float
) -> None:
    _POOL_DATA["sig_s"] = sig_s
    _POOL_DATA["sser"] = sser
    _POOL_DATA["bser"] = bser
    _POOL_DATA["dser"] = dser
    _POOL_DATA["capital"] = float(capital)


def _evaluate_one_param(pr: ParamRow) -> list[dict]:
    sig_s = _POOL_DATA["sig_s"]
    sser = _POOL_DATA["sser"]
    bser = _POOL_DATA["bser"]
    dser = _POOL_DATA["dser"]
    capital = float(_POOL_DATA["capital"])
    ix = sig_s.index
    mav = mbase.ma_long(sig_s, pr.ma_win).reindex(ix)
    nav_b = mbase.run_merge_fsm(
        sig_s, sser, bser, dser, mav,
        trail=pr.trail,
        capital=capital,
        mult_dn=pr.mult_dn,
        mult_up=pr.mult_up,
        ma_fast=pr.ma_fast,
        ma_slow=pr.ma_slow,
    )
    nav_s = mb_sticky.run_merge_fsm(
        sig_s, sser, bser, dser, mav,
        trail=pr.trail,
        capital=capital,
        mult_dn=pr.mult_dn,
        mult_up=pr.mult_up,
        ma_fast=pr.ma_fast,
        ma_slow=pr.ma_slow,
    )
    out: list[dict] = []
    for tag, nav in (("base", nav_b), ("sticky", nav_s)):
        m = full_metrics(nav)
        out.append(
            {
                "strategy": tag,
                "ma_win": pr.ma_win,
                "ma_fast": pr.ma_fast,
                "ma_slow": pr.ma_slow,
                "mult_dn": pr.mult_dn,
                "mult_up": pr.mult_up,
                "trail": pr.trail,
                "cagr": m["cagr"],
                "mdd": m["mdd"],
                "sharpe": m["sharpe"],
                "sortino": m["sortino"],
                "ulcer": m["ulcer"],
                "calmar": m["calmar"],
            }
        )
    return out


def _fmt_pct(x: float) -> str:
    return f"{float(x) * 100:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="merge bounce 파라미터 플래토 스윕")
    ap.add_argument("--root", default="1999-03-10")
    ap.add_argument("--capital", type=float, default=mbase.CAP0)
    ap.add_argument("--bounce", default="TQQQ")
    ap.add_argument("--defense", default="QQQ")
    ap.add_argument("--safe", default="Gold")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--jobs", type=int, default=0, help="0=자동(CPU−1), 1=단일")
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
    max_w_needed = max(MA_WIN_GRID)
    if len(ix) < max_w_needed + 20:
        raise SystemExit(f"일수 부족: {len(ix)} < 권장 {max_w_needed + 20}")

    sig_s = sg["Close"].astype(float).reindex(ix)
    sgx, qlx, tgx, glx = sg.reindex(ix), ql.reindex(ix), tg.reindex(ix), gold_df.reindex(ix)
    bser = mbase.pick(args.bounce.strip(), sgx, qlx, tgx, glx)
    dser = mbase.pick(args.defense.strip(), sgx, qlx, tgx, glx)
    sser = mbase.pick(args.safe.strip(), sgx, qlx, tgx, glx)

    ma_candidates = [w for w in MA_WIN_GRID if w <= len(ix) - 5]
    grid = list(iter_param_grid(ma_candidates))
    fx0 = pd.Timestamp(ix[0]).strftime("%Y%m%d")
    fx1 = pd.Timestamp(ix[-1]).strftime("%Y%m%d")
    stem = f"merge_bounce_plateau_{fx0}_{fx1}"

    total = len(grid)
    n_jobs = args.jobs if args.jobs > 0 else max(1, int((os.cpu_count() or 4) - 1))
    print(f"스윕 {total}조합 × 2전략 · {len(ix)}일 · workers={n_jobs}")

    if n_jobs == 1:
        _mp_worker_init(sig_s, sser, bser, dser, args.capital)
        rows: list[dict] = []
        for i, pr in enumerate(grid):
            rows.extend(_evaluate_one_param(pr))
            if (i + 1) % 500 == 0 or i + 1 == total:
                print(f"  … {i + 1}/{total}")
    else:
        chunksize = max(1, total // max(n_jobs * 8, 1))
        with Pool(
            processes=n_jobs,
            initializer=_mp_worker_init,
            initargs=(sig_s, sser, bser, dser, args.capital),
        ) as pool:
            parts = pool.map(_evaluate_one_param, grid, chunksize=chunksize)
        rows = [r for part in parts for r in part]
        print(f"  완료 {total}조합")

    df = pd.DataFrame(rows)
    csv_p = out_dir / f"{stem}_grid.csv.gz"
    df.to_csv(csv_p, index=False, compression="gzip")

    lines = [
        "# Merge Bounce 파라미터 플래토 vs 피크 검증 리포트",
        "",
        f"- **표본:** {ix[0].date()} ~ {ix[-1].date()} (거래일 {len(ix)}), root=`{args.root}`",
        f"- **격자:** {total} 설정 × 2전략, workers={n_jobs}",
        f"- **스윕 축:** `ma_win`(롱 MA) · `ma_fast`/`ma_slow`(스트레스 쌍 이폭) · `mult_dn` · `mult_up` · 반등 `trail`",
        f"- **디폴트 정합:** ma_win `{BASELINE_WIN}`, MA `{BASELINE_FAST}`/`{BASELINE_SLOW}`, mult_dn `{BASELINE_DN}`, mult_up `{BASELINE_UP}`, trail `{BASELINE_TRAIL}`",
        f"- **금:** {gold_note}",
        "",
        "---",
        "",
    ]

    for strat in ("base", "sticky"):
        sub = df[df["strategy"] == strat].copy()
        is_bl = (
            (sub["ma_win"] == BASELINE_WIN)
            & (sub["ma_fast"] == BASELINE_FAST)
            & (sub["ma_slow"] == BASELINE_SLOW)
            & (np.isclose(sub["mult_dn"], BASELINE_DN))
            & (np.isclose(sub["mult_up"], BASELINE_UP))
            & (np.isclose(sub["trail"], BASELINE_TRAIL))
        )
        bl = sub[is_bl].iloc[0]
        sx = sub["sortino"]
        mx = float(sx.max())
        med = float(sx.median())
        pct_hi = _sortino_empirical_percentile(sx, float(bl["sortino"]))
        share_ge_95pct_max = float(np.mean(sx >= 0.95 * mx))
        share_near_90pct_max = float(np.mean(sx >= 0.90 * mx))
        share_within_baseline_band = float(
            np.mean((sx >= 0.92 * float(bl["sortino"])) & (sx <= 1.08 * float(bl["sortino"])))
        )
        gap_to_max = mx - float(bl["sortino"])
        rel_gap = gap_to_max / max(mx, 1e-12)
        ratio_bm = float(bl["sortino"]) / mx
        best_row = sub.loc[sx.idxmax()]
        nei = one_param_neighbors_df(sub, bl)

        lines.extend(
            [
                f"## 전략 `{strat}`",
                "",
                "### 디폴트 전구간 지표",
                "| 지표 | 값 |",
                "|------|-----:|",
                f"| CAGR | {_fmt_pct(bl['cagr'])} |",
                f"| MDD | {_fmt_pct(bl['mdd'])} |",
                f"| Sortino | {bl['sortino']:.4f} |",
                f"| Sharpe | {bl['sharpe']:.4f} |",
                f"| Ulcer | {bl['ulcer']:.4f} |",
                f"| Sortino 경험 분위 (↑우수) | **{pct_hi:.2f}%** |",
                "",
                "### 격자 대비 디폴트 (Sortino)",
                f"- 격자 최고 **{mx:.4f}**, 디폴트 비율 **{float(bl['sortino'])/mx:.2%}**, 갭 **{gap_to_max:.4f}** (~{rel_gap:.1%})",
                f"- 격자 Sortino σ **{sx.std(ddof=1):.4f}**, 중앙값 **{med:.4f}**",
                f"- 최고의 **≥95%** Sortino 비율: **{share_ge_95pct_max * 100:.2f}%**",
                f"- 최고의 **≥90%** Sortino 비율: **{share_near_90pct_max * 100:.2f}%** (상단 완만성 지표)",
                f"- 디폴트 Sortino의 **±8%** 안 조합 비율: **{share_within_baseline_band * 100:.2f}%**",
                "",
                "### 극대 격자 후보",
                f"- `{best_row['ma_win']:.0f}`, `{best_row['ma_fast']:.0f}/{best_row['ma_slow']:.0f}`, dn {best_row['mult_dn']:.4f}, up {best_row['mult_up']:.4f}, tr {best_row['trail']:.3f}",
                f"- Sortino **{best_row['sortino']:.4f}**, CAGR {_fmt_pct(best_row['cagr'])}, MDD {_fmt_pct(best_row['mdd'])}",
                "",
                "### 이웃 1격자 (축별): 평균 Sortino vs 디폴트",
                "",
                "| 축 | n | 평균 Sortino | Δ |",
                "|:--|--:|--:|--:|",
            ]
        )
        for _, rr in nei.iterrows():
            lines.append(
                f"| {rr['axis']} | {rr['n_neighbor_cells']:.0f} | {rr['neighbor_sortino_mean']:.4f} | "
                f"{rr['delta_vs_baseline_mean']:+.4f} |"
            )
        nei.to_csv(out_dir / f"{stem}_neighbors_{strat}.csv", index=False)

        hi_flat = pct_hi >= 94 and ratio_bm >= 0.92
        mid_zone = pct_hi >= 85 and ratio_bm >= 0.82
        if hi_flat:
            note = f"격자 중 Sortino≥90% 극값 비율 **{share_near_90pct_max * 100:.2f}%**."
            v = (
                "**판단:** 디폴트가 Sortino 분위 기준 극상단에 있으며, 격자 최댓값 대비 회수도 높음 → "
                "**협폭 바늘 1점만은 아니고 우상단 플래토·세부 피크의 경계**로 보임. 최고 근처(ma_win 길게, mult 더 보수적으로) 재검증 권장. "
                + note
            )
        elif mid_zone:
            v = "**판단:** 우량존 확실. 디폴트와 격자 최고 사이 간격은 있음 → **약간 다른 파라메터 묶음이 더 유리할 수 있음**(이웃 표 참조)."
        else:
            v = "**판단:** 이 표본 스윕에서는 디폴트가 상단에서 덜 높거나 최댓값 대비 효율이 낮음. 규격·격자 재검토 또는 OOS 검증 필요.**"
        lines.extend(["", v, "", "---", ""])

    lines.extend(
        [
            "",
            "### 산출",
            "",
            f"- `{csv_p.name}` (gzip CSV)",
            f"- `{stem}_neighbors_base.csv`, `{stem}_neighbors_sticky.csv`",
            "",
            "*미반영: 슬리피지·세금.*",
        ]
    )

    (out_dir / f"{stem}_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("=== 완료 ===")
    print(csv_p.resolve())
    print(out_dir / f"{stem}_report.md")


if __name__ == "__main__":
    main()
