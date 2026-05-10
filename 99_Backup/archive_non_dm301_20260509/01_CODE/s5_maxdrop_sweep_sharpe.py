"""
S5 앵커 고정(β, min_drop, exp, SL 등) — max_drop만 20%~40% (1%p) 스윕 후 Sharpe 등 비교.

실행 (저장소 루트):
  python3 01_CODE/s5_maxdrop_sweep_sharpe.py
  python3 01_CODE/s5_maxdrop_sweep_sharpe.py --start 2019-01-01 --end 2025-12-31

출력: 03_RESULT/sensitivity/s5_maxdrop_sweep_sharpe_{tag}.csv, .png
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
    p = argparse.ArgumentParser(description="S5 max_drop 스윕 Sharpe 비교")
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

    max_drops = np.linspace(0.20, 0.40, 21)
    rows = []
    for md in max_drops:
        port, _ = strategy_s5(
            Q, L, T, CAP,
            beta=0.5,
            min_drop=0.10,
            max_drop=float(md),
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
            "max_drop_pct": round(float(md) * 100, 0),
            "max_drop": float(md),
            "sharpe": met["sharpe"],
            "sortino": met["sortino"],
            "cagr": met["cagr"],
            "total_return": met["total_return"],
            "mdd": met["mdd"],
            "ulcer": met["ulcer"],
            "n_years": met["n_years"],
        })

    df = pd.DataFrame(rows)
    best = df.loc[df["sharpe"].idxmax()]
    worst = df.loc[df["sharpe"].idxmin()]

    tag = f"{Q.index[0].strftime('%Y%m%d')}_{Q.index[-1].strftime('%Y%m%d')}"
    csv_path = OUT_DIR / f"s5_maxdrop_sweep_sharpe_{tag}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df["max_drop_pct"], df["sharpe"], "o-", color="#2563EB", lw=1.5, ms=4)
    ax.axvline(best["max_drop_pct"], color="#DC2626", ls="--", lw=1, alpha=0.8,
               label=f"max Sharpe: md={int(best['max_drop_pct'])}%")
    ax.set_xlabel("max_drop (%)" )
    ax.set_ylabel("Sharpe")
    ax.set_title(
        f"S5 exp 앵커 max_drop 스윕 vs Sharpe  |  {Q.index[0].date()} ~ {Q.index[-1].date()}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    png_path = OUT_DIR / f"s5_maxdrop_sweep_sharpe_{tag}.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    print("=" * 72)
    print(f"기간: {Q.index[0].date()} ~ {Q.index[-1].date()}  ({len(Q):,}일)")
    print(f"max_drop: 20%~40% (1%p), 그 외 = 벤치마크 S5 앵커와 동일")
    print("=" * 72)
    df2 = df.copy()
    df2["max_drop_pct"] = df2["max_drop_pct"].astype(int)
    print(df2[["max_drop_pct", "sharpe", "sortino", "cagr", "mdd", "total_return"]].to_string(index=False))
    print("=" * 72)
    print(f"Sharpe 최고: max_drop={int(best['max_drop_pct'])}%  Sharpe={best['sharpe']:.4f}")
    print(f"Sharpe 최저: max_drop={int(worst['max_drop_pct'])}%  Sharpe={worst['sharpe']:.4f}")
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
