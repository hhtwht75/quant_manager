"""
monte_carlo_rbs_dsa_qld.py
==========================
QLD 100% 단순보유 vs DSA vs RMABS (RSI_and_Moving_Average_Based_Switching).

• 전역 1회: 각 전략 ground truth 순자산 → 랜덤 구간은 동일 시작·끝 인덱스로 슬라이스
• 최소 2년 거래일(기본 504일), 반복 기본 N=3000, seed 재현 가능
• 출력: 03_RESULT/sensitivity/*.json · *.csv · 분포 플롯

실행 예:
  python3 01_CODE/monte_carlo_rbs_dsa_qld.py
  python3 01_CODE/monte_carlo_rbs_dsa_qld.py --n 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import (  # noqa: E402
    load_extended_daily,
    strategy_buy_and_hold,
    strategy_rsi_ma_based_switching,
    strategy_s5,
)
from evaluation_metrics import full_metrics  # noqa: E402

OUT_DIR = _ROOT / "03_RESULT" / "sensitivity"
CAP = 100_000

S5_DSA_COMMON = dict(
    beta=0.5,
    min_drop=0.10,
    max_drop=0.20,
    trailing_stop_pct=-0.15,
    use_stop_loss=True,
    position_mode="exp",
    exp_frac_lo=0.25,
    exp_frac_hi=1.00,
    exp_base=2.0,
    stop_factor=0.75,
)


def slice_metrics(norm_slice: pd.Series) -> dict:
    m = full_metrics((norm_slice * CAP).rename("p"))
    return {
        "cagr": float(m["cagr"]),
        "sharpe": float(m["sharpe"]),
        "sortino": float(m["sortino"]),
        "mdd": float(m["mdd"]),
        "ulcer": float(m["ulcer"]),
    }


def summarise_delta(s: pd.Series, *, ulcer_lower_better: bool) -> dict:
    s = s.dropna()
    if len(s) == 0:
        return {"median": 0.0, "mean": 0.0, "p05": 0.0, "p95": 0.0, "win_pct": 0.0}
    wp = float((s < 0).mean() * 100) if ulcer_lower_better else float((s > 0).mean() * 100)
    return {
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p05": float(s.quantile(0.05)),
        "p95": float(s.quantile(0.95)),
        "win_pct": wp,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="2002-10-01", help="평가 시작 필터(Y-M-D)")
    ap.add_argument("--n", type=int, default=3000, help="Monte Carlo 창 개수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-years", type=float, default=2.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qqq = load_extended_daily("QQQ")
    qld = load_extended_daily("QLD")
    tqqq = load_extended_daily("TQQQ")
    common = qqq.index.intersection(qld.index).intersection(tqqq.index)
    root_ts = pd.Timestamp(args.root.strip())
    common = common[common >= root_ts]
    Q_raw = qqq.loc[common]
    L_raw = qld.loc[common]
    T_raw = tqqq.loc[common]
    Ndays = len(common)
    min_len = max(2, int(np.ceil(args.min_years * 252)))
    if Ndays < min_len + 10:
        raise SystemExit(f"데이터 거래일 {Ndays} < 최소창 {min_len}+")

    rng = np.random.default_rng(args.seed)
    si_arr = np.empty(args.n, dtype=np.int64)
    ei_arr = np.empty(args.n, dtype=np.int64)
    for k in range(args.n):
        s_i = int(rng.integers(0, Ndays - min_len + 1))
        e_i = int(rng.integers(s_i + min_len - 1, Ndays))
        si_arr[k] = s_i
        ei_arr[k] = e_i

    print("=" * 72)
    print("Monte Carlo  |  QLD B&H vs DSA vs RMABS")
    print(f"  전체 구간: {common[0].date()} ~ {common[-1].date()}  ({Ndays:,}일)")
    print(f"  창 반복={args.n}  seed={args.seed}  최소 거래일={min_len}")
    print("=" * 72)

    print("\n  [STEP 1] 전 구간 각 전략 1회 실행…")
    t0 = time.time()
    bh_qld = strategy_buy_and_hold(L_raw, CAP, "QLD BH")
    dsa, ev_dsa = strategy_s5(Q_raw, L_raw, T_raw, CAP, series_name="DSA", **S5_DSA_COMMON)
    rmabs, ev_rmabs = strategy_rsi_ma_based_switching(
        Q_raw, L_raw, T_raw, CAP, series_name="RMABS"
    )

    m_bh = full_metrics(bh_qld)
    m_d = full_metrics(dsa)
    m_r = full_metrics(rmabs)
    print(f"  완료 ({time.time() - t0:.1f}s)")
    print(
        f"  QLD B&H  CAGR={m_bh['cagr'] * 100:>+7.2f}%  Sharpe={m_bh['sharpe']:.3f}  "
        f"MDD={m_bh['mdd'] * 100:.2f}%  Ulcer={m_bh['ulcer']:.2f}"
    )
    print(
        f"  DSA      CAGR={m_d['cagr'] * 100:>+7.2f}%  Sharpe={m_d['sharpe']:.3f}  "
        f"MDD={m_d['mdd'] * 100:.2f}%  Ulcer={m_d['ulcer']:.2f}  이벤트={len(ev_dsa)}"
    )
    print(
        f"  RMABS    CAGR={m_r['cagr'] * 100:>+7.2f}%  Sharpe={m_r['sharpe']:.3f}  "
        f"MDD={m_r['mdd'] * 100:.2f}%  Ulcer={m_r['ulcer']:.2f}  이벤트={len(ev_rmabs)}"
    )

    print("\n  [STEP 2] 동일 무작위 구간 슬라이스…")
    rows = []
    t1 = time.time()
    for k in range(args.n):
        si, ei = si_arr[k], ei_arr[k]
        s0 = common[si].strftime("%Y-%m-%d")
        e0 = common[ei].strftime("%Y-%m-%d")
        w_bh = bh_qld.iloc[si : ei + 1] / bh_qld.iloc[si]
        w_ds = dsa.iloc[si : ei + 1] / dsa.iloc[si]
        w_rm = rmabs.iloc[si : ei + 1] / rmabs.iloc[si]

        m0 = slice_metrics(w_bh)
        md = slice_metrics(w_ds)
        mr = slice_metrics(w_rm)

        row = {"iter": int(k), "start": s0, "end": e0, "n_days": int(ei - si + 1)}
        for nm, mx in ("QLD", m0), ("DSA", md), ("RMABS", mr):
            row[f"{nm}_cagr"] = mx["cagr"]
            row[f"{nm}_sharpe"] = mx["sharpe"]
            row[f"{nm}_sortino"] = mx["sortino"]
            row[f"{nm}_mdd"] = mx["mdd"]
            row[f"{nm}_ulcer"] = mx["ulcer"]
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  완료 ({time.time() - t1:.1f}s)\n")

    # Δ메트릭
    MET = ("cagr", "sharpe", "sortino", "mdd", "ulcer")

    df["DSA_minus_QLD_cagr"] = df["DSA_cagr"] - df["QLD_cagr"]
    df["DSA_minus_QLD_sharpe"] = df["DSA_sharpe"] - df["QLD_sharpe"]
    df["DSA_minus_QLD_sortino"] = df["DSA_sortino"] - df["QLD_sortino"]
    df["DSA_minus_QLD_mdd"] = df["DSA_mdd"] - df["QLD_mdd"]
    df["DSA_minus_QLD_ulcer"] = df["DSA_ulcer"] - df["QLD_ulcer"]

    df["RMABS_minus_QLD_cagr"] = df["RMABS_cagr"] - df["QLD_cagr"]
    df["RMABS_minus_QLD_sharpe"] = df["RMABS_sharpe"] - df["QLD_sharpe"]
    df["RMABS_minus_QLD_sortino"] = df["RMABS_sortino"] - df["QLD_sortino"]
    df["RMABS_minus_QLD_mdd"] = df["RMABS_mdd"] - df["QLD_mdd"]
    df["RMABS_minus_QLD_ulcer"] = df["RMABS_ulcer"] - df["QLD_ulcer"]

    df["DSA_minus_RMABS_cagr"] = df["DSA_cagr"] - df["RMABS_cagr"]
    df["DSA_minus_RMABS_sharpe"] = df["DSA_sharpe"] - df["RMABS_sharpe"]
    df["DSA_minus_RMABS_sortino"] = df["DSA_sortino"] - df["RMABS_sortino"]
    df["DSA_minus_RMABS_mdd"] = df["DSA_mdd"] - df["RMABS_mdd"]
    df["DSA_minus_RMABS_ulcer"] = df["DSA_ulcer"] - df["RMABS_ulcer"]

    pairs = ("DSA_minus_QLD", "RMABS_minus_QLD", "DSA_minus_RMABS")
    pair_labels = {
        "DSA_minus_QLD": "DSA − QLD B&H",
        "RMABS_minus_QLD": "RMABS − QLD B&H",
        "DSA_minus_RMABS": "DSA − RMABS",
    }

    summary: dict = {
        "meta": {
            "full_range": [str(common[0].date()), str(common[-1].date())],
            "n_trading_days_full": int(Ndays),
            "n_mc_windows": args.n,
            "seed": args.seed,
            "min_trading_days": min_len,
            "dsa_params": dict(S5_DSA_COMMON),
            "rmabs": "규칙0: 초기 MA200 대비 QLD↔QQQ; RSI·MA↑1.03→TQQQ; TQQQ 청산→QLD(-15%/진입가); MA↓0.97→QQQ(플래그)",
        },
        "full_period_absolute": {
            "QLD": {k: float(m_bh[k]) for k in MET},
            "DSA": {k: float(m_d[k]) for k in MET},
            "RMABS": {k: float(m_r[k]) for k in MET},
            "DSA_events_count": len(ev_dsa),
            "RMABS_events_count": len(ev_rmabs),
        },
        "delta_by_pair": {},
    }

    ulcer_keys = {"cagr": False, "sharpe": False, "sortino": False, "mdd": False, "ulcer": True}
    for pref in pairs:
        summary["delta_by_pair"][pair_labels[pref]] = {}
        for met in MET:
            ulcer_lb = ulcer_keys[met]
            stats = summarise_delta(df[f"{pref}_{met}"], ulcer_lower_better=ulcer_lb)
            summary["delta_by_pair"][pair_labels[pref]][met] = stats

    tag = f"{common[0].strftime('%Y%m%d')}_{common[-1].strftime('%Y%m%d')}_n{args.n}_s{args.seed}"
    csv_path = OUT_DIR / f"mc_rmabs_dsa_qld_{tag}.csv"
    json_path = OUT_DIR / f"mc_rmabs_dsa_qld_{tag}.json"
    png_path = OUT_DIR / f"mc_rmabs_dsa_qld_{tag}.png"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 플롯: Δ CAGR / Sharpe 히스토그램
    fig = plt.figure(figsize=(12.5, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    axes_cfg = [
        (0, 0, "DSA − QLD  ΔCAGR", "DSA_minus_QLD_cagr"),
        (0, 1, "RMABS − QLD  ΔCAGR", "RMABS_minus_QLD_cagr"),
        (0, 2, "DSA − RMABS  ΔCAGR", "DSA_minus_RMABS_cagr"),
        (1, 0, "DSA − QLD  ΔSharpe", "DSA_minus_QLD_sharpe"),
        (1, 1, "RMABS − QLD  ΔSharpe", "RMABS_minus_QLD_sharpe"),
        (1, 2, "DSA − RMABS  ΔSharpe", "DSA_minus_RMABS_sharpe"),
    ]
    for r, c, ttl, col in axes_cfg:
        ax = fig.add_subplot(gs[r, c])
        x = df[col].astype(float).values * 100 if "_cagr" in col else df[col].astype(float).values
        ax.hist(x, bins=65, density=True, color="#475569", alpha=0.75, edgecolor="white", linewidth=0.4)
        med = np.median(x)
        ax.axvline(med, color="#DC2626", lw=2, label=f"median {med:.2f}")
        ax.axvline(0.0, color="#9CA3AF", ls="--", lw=1)
        ax.set_title(ttl)
        if "_cagr" in col:
            ax.set_xlabel(r"ΔCAGR (pct points)")
        else:
            ax.set_xlabel(r"ΔSharpe")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Monte Carlo window deltas (normalized slice, min 2y) — QLD B&H vs DSA vs RMABS",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("=== 창별 Δ 요약 · 승률: CAGR/Sharpe/Sortino/MDD=P(Δ>0), Ulcer=P(Δ<0)")
    print("---")
    for pref in pairs:
        plab = pair_labels[pref]
        print(f"\n  [{plab}]")
        for met in MET:
            st = summary["delta_by_pair"][plab][met]
            mlab = "%" if met == "cagr" else ""
            median = st["median"] * 100 if met == "cagr" else st["median"]
            mean = st["mean"] * 100 if met == "cagr" else st["mean"]
            print(
                f"    {met:8s}: median={median:+8.4f}{mlab}  mean={mean:+8.4f}{mlab}  "
                f"승률={st['win_pct']:.1f}%"
            )

    print("\n파일 저장:")
    print(" ", csv_path)
    print(" ", json_path)
    print(" ", png_path)


if __name__ == "__main__":
    main()
