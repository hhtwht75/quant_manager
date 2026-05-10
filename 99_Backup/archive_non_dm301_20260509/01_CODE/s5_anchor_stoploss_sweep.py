"""
S5 앵커 전략: stop_factor(손절) 스윕 → 누적수익, CAGR, MDD, Sharpe, Sortino, Ulcer.

• 상단: 스윕 구간 안에서 각 지표를 [0,1]로 정규화(높을수록 유리)하여 한 축에 오버랩.
• 하단: 동일 x축(stop_factor)으로 지표별 원시 곡선 6분할.

기간: 2002-10-01 ~ 데이터 끝 (extended QQQ/QLD/TQQQ)
앵커: β=0.5, max_drop=20%, min_drop=10%, trail=-15%, use_stop_loss=True

실행: 저장소 루트에서  python3 01_CODE/s5_anchor_stoploss_sweep.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Apple SD Gothic Neo"
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2002-10-01")
CAP = 100_000
BETA, MAX_DROP, MIN_DROP, TRAIL = 0.5, 0.20, 0.10, -0.15
# 진입 조건상 stop_factor > beta + 0.02
SF_MIN, SF_MAX, SF_N = 0.53, 1.45, 30


def _minmax01_higher_better(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if hi - lo < 1e-15:
        return np.full_like(x, 0.5, dtype=float)
    return (x - lo) / (hi - lo)


def _minmax01_lower_better(x: np.ndarray) -> np.ndarray:
    """낮을수록 좋음 → (max - x) 후 min-max."""
    return _minmax01_higher_better(-np.asarray(x, dtype=float))


def main():
    qqq = load_extended_daily("QQQ")
    qld = load_extended_daily("QLD")
    tqqq = load_extended_daily("TQQQ")
    common = qqq.index.intersection(qld.index).intersection(tqqq.index)
    qqq, qld, tqqq = qqq.loc[common], qld.loc[common], tqqq.loc[common]
    m = qqq.index >= START
    Q, L, T = qqq.loc[m], qld.loc[m], tqqq.loc[m]

    sf_grid = np.linspace(SF_MIN, SF_MAX, SF_N)
    rows = []
    for sf in sf_grid:
        port, _ = strategy_s5(
            Q, L, T, CAP,
            beta=BETA, max_drop=MAX_DROP, min_drop=MIN_DROP,
            stop_factor=float(sf), trailing_stop_pct=TRAIL,
            use_stop_loss=True,
            position_mode="linear",
        )
        r = oos_metric_bundle(port)
        rows.append({"stop_factor": float(sf), **r})

    df = pd.DataFrame(rows)

    port_nosl, _ = strategy_s5(
        Q, L, T, CAP,
        beta=BETA, max_drop=MAX_DROP, min_drop=MIN_DROP,
        stop_factor=0.75, trailing_stop_pct=TRAIL,
        use_stop_loss=False,
        position_mode="linear",
    )
    ref_nosl = oos_metric_bundle(port_nosl)

    # 정규화 (스윕 구간 내 상대 비교)
    n_tot = _minmax01_higher_better(df["total_return"].values)
    n_cag = _minmax01_higher_better(df["cagr"].values)
    n_mdd = _minmax01_higher_better((-df["mdd"]).values)  # 얕은 DD가 우수
    n_shp = _minmax01_higher_better(df["sharpe"].values)
    n_sor = _minmax01_higher_better(df["sortino"].values)
    n_ulc = _minmax01_lower_better(df["ulcer"].values)

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1.0, 1.0], hspace=0.4, wspace=0.32)
    ax0 = fig.add_subplot(gs[0, :])
    x = df["stop_factor"]
    ax0.plot(x, n_tot, lw=2.0, label="누적수익")
    ax0.plot(x, n_cag, lw=2.0, label="CAGR")
    ax0.plot(x, n_mdd, lw=2.0, label="−MDD (얕을수록↑)")
    ax0.plot(x, n_shp, lw=2.0, label="Sharpe")
    ax0.plot(x, n_sor, lw=2.0, label="Sortino")
    ax0.plot(x, n_ulc, lw=2.0, label="Ulcer (낮을수록↑)")
    ax0.set_xlabel(r"stop_factor ($rebound > sf \times drop_{entry}$ 시 손절)")
    ax0.set_ylabel("정규화 0~1 (스윕 구간 내)")
    ax0.set_title(
        "S5 앵커 — stop_factor 스윕 지표 오버랩\n"
        f"β={BETA}, max_drop={MAX_DROP:.0%}, min_drop={MIN_DROP:.0%}, trail={TRAIL:.0%}  |  "
        f"{Q.index[0].date()} ~ {Q.index[-1].date()}",
        fontsize=11,
    )
    ax0.legend(loc="best", fontsize=8, ncol=3)
    ax0.grid(True, alpha=0.3)

    panels = [
        ("total_return", "누적수익"),
        ("cagr", "CAGR"),
        ("mdd", "MDD"),
        ("sharpe", "Sharpe"),
        ("sortino", "Sortino"),
        ("ulcer", "Ulcer"),
    ]
    for i, (col, ttl) in enumerate(panels):
        rr, cc = i // 3, i % 3
        ax = fig.add_subplot(gs[1 + rr, cc])
        ax.plot(df["stop_factor"], df[col], color="#1D4ED8", lw=1.6)
        if col == "mdd":
            ax.axhline(0, color="gray", lw=0.6, alpha=0.6)
        ax.set_title(ttl, fontsize=10, fontweight="bold")
        ax.set_xlabel("stop_factor", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(ref_nosl[col], color="#DC2626", ls="--", lw=1.0, alpha=0.85)
        if i == 0:
            ax.plot([], [], color="#DC2626", ls="--", lw=1.0, label="SL 없음 (참고)")
            ax.legend(fontsize=8, loc="best")

    out_png = OUT_DIR / "s5_anchor_stoploss_sweep.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    ix = int(df["sharpe"].values.argmax())
    best_row = df.iloc[ix]
    out_json = OUT_DIR / "s5_anchor_stoploss_sweep.json"
    payload = {
        "anchor": {
            "beta": BETA, "max_drop": MAX_DROP, "min_drop": MIN_DROP,
            "trail": TRAIL, "use_stop_loss_sweep": True,
        },
        "period": [str(Q.index[0].date()), str(Q.index[-1].date())],
        "sf_grid": [float(x) for x in sf_grid],
        "sweep_rows": df.to_dict(orient="records"),
        "no_stop_loss_reference": ref_nosl,
        "best_sharpe_row": best_row.to_dict(),
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"저장: {out_png}")
    print(f"저장: {out_json}")
    print(f"  Sharpe 최대 stop_factor ≈ {best_row['stop_factor']:.4f}")


if __name__ == "__main__":
    main()
