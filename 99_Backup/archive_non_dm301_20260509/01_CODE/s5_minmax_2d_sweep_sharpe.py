"""
S5 앵커 고정(β, exp, SL 등) — min_drop ∈ [1%,20%], max_drop ∈ [10%,30%] (각 1%p) 2-D 스윕 후 Sharpe 비교.

유효 조합: max_drop > min_drop (지수 사이징·진입 조건 일관성)

실행:
  python3 01_CODE/s5_minmax_2d_sweep_sharpe.py
  python3 01_CODE/s5_minmax_2d_sweep_sharpe.py --start 2019-01-01 --end 2025-12-31

출력: 03_RESULT/sensitivity/s5_minmax_2d_sweep_sharpe_{tag}.csv, .png
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import load_extended_daily, strategy_s5  # noqa: E402
from evaluation_metrics import oos_metric_bundle  # noqa: E402

OUT_DIR = _ROOT / "03_RESULT" / "sensitivity"
CAP = 100_000
DEFAULT_START = pd.Timestamp("2002-10-01")


def parse_args():
    p = argparse.ArgumentParser(description="S5 min_drop × max_drop 2D 스윕 Sharpe")
    p.add_argument("--start", type=str, default=str(DEFAULT_START.date()))
    p.add_argument("--end", type=str, default="", help="비우면 데이터 끝까지")
    return p.parse_args()


def main():
    args = parse_args()
    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end) if args.end.strip() else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qqq = load_extended_daily("QQQ")
    qld = load_extended_daily("QLD")
    tqqq = load_extended_daily("TQQQ")
    common = qqq.index.intersection(qld.index).intersection(tqqq.index)
    qqq, qld, tqqq = qqq.loc[common], qld.loc[common], tqqq.loc[common]
    if end_ts is not None:
        m = (qqq.index >= start_ts) & (qqq.index <= end_ts)
    else:
        m = qqq.index >= start_ts
    Q, L, T = qqq.loc[m], qld.loc[m], tqqq.loc[m]
    if len(Q) < 20:
        raise SystemExit(f"구간 너무 짧음: {len(Q)}일")

    min_pcts = list(range(1, 21))
    max_pcts = list(range(10, 31))

    rows = []
    for ip in min_pcts:
        mind = ip / 100.0
        for jp in max_pcts:
            maxd = jp / 100.0
            if maxd <= mind:
                continue
            port, _ = strategy_s5(
                Q, L, T, CAP,
                beta=0.5,
                min_drop=mind,
                max_drop=maxd,
                trailing_stop_pct=-0.15,
                use_stop_loss=True,
                position_mode="exp",
                exp_frac_lo=0.25,
                exp_frac_hi=1.00,
                exp_base=2.0,
                stop_factor=0.75,
            )
            met = oos_metric_bundle(port)
            rows.append({
                "min_drop_pct": ip,
                "max_drop_pct": jp,
                "min_drop": mind,
                "max_drop": maxd,
                "sharpe": met["sharpe"],
                "sortino": met["sortino"],
                "cagr": met["cagr"],
                "total_return": met["total_return"],
                "mdd": met["mdd"],
                "ulcer": met["ulcer"],
            })

    df = pd.DataFrame(rows)
    best = df.loc[df["sharpe"].idxmax()]

    tag = f"{Q.index[0].strftime('%Y%m%d')}_{Q.index[-1].strftime('%Y%m%d')}"
    csv_path = OUT_DIR / f"s5_minmax_2d_sweep_sharpe_{tag}.csv"
    df.sort_values(["min_drop_pct", "max_drop_pct"]).to_csv(
        csv_path, index=False, encoding="utf-8-sig",
    )

    # Heatmap: 행=min_drop 1..20, 열=max_drop 10..30
    nmin, nmax = len(min_pcts), len(max_pcts)
    grid = np.full((nmin, nmax), np.nan, dtype=float)
    imin = {p: i for i, p in enumerate(min_pcts)}
    imax = {p: j for j, p in enumerate(max_pcts)}
    for _, r in df.iterrows():
        grid[imin[int(r["min_drop_pct"])], imax[int(r["max_drop_pct"])]] = r["sharpe"]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[
            min(max_pcts) - 0.5,
            max(max_pcts) + 0.5,
            min(min_pcts) - 0.5,
            max(min_pcts) + 0.5,
        ],
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Sharpe")
    ax.set_xlabel("max_drop (%)")
    ax.set_ylabel("min_drop (%)")
    ax.set_title(
        f"S5 exp: Sharpe(min_drop × max_drop)  |  {Q.index[0].date()} ~ {Q.index[-1].date()}\n"
        f"★ best: min={int(best['min_drop_pct'])}% max={int(best['max_drop_pct'])}%  Sharpe={best['sharpe']:.4f}"
    )
    fig.tight_layout()
    png_path = OUT_DIR / f"s5_minmax_2d_sweep_sharpe_{tag}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print("=" * 76)
    print(f"기간: {Q.index[0].date()} ~ {Q.index[-1].date()}  ({len(Q):,}일)")
    print("min_drop ∈ [1,20]%, max_drop ∈ [10,30]%, 1%p (max_drop > min_drop 만 평가)")
    print(f"조합 수: {len(df)}  |  Sharpe 최고: min={int(best['min_drop_pct'])}% max={int(best['max_drop_pct'])}% → {best['sharpe']:.4f}")
    print("\n[Sharpe 상위 10]")
    top = df.nlargest(10, "sharpe")[
        ["min_drop_pct", "max_drop_pct", "sharpe", "cagr", "mdd", "sortino"]
    ]
    print(top.to_string(index=False))
    print("=" * 76)
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
