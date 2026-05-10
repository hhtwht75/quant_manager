"""
Monte Carlo (동일 무작위 창 슬라이스)
======================================
B1~B4 벤치 + 커스텀 3종(DSA, RMABS-QQQ, RMABS-QLD).

• 전구간 1회: 각 시리즈 NAV → 같은 (start,end) 인덱스로 정규화 슬라이스
• 기본 반복 N=3000, seed=42, 최소 창 거래일 2년

출력: ``03_RESULT/sensitivity/mc_bench_rmabs_suite_*.csv|json|png|*_tables.txt`` (표 ①②③ 통합 텍스트)

실행:
  python3 01_CODE/monte_carlo_bench_dsa_rmabs.py --n 3000 --seed 42
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

from benchmark_s2_s5_anchor_table import S5_ANCHOR_COMMON, half_half_drift  # noqa: E402
from backtest_switching import (  # noqa: E402
    load_extended_daily,
    strategy_buy_and_hold,
    strategy_rsi_ma_based_switching,
    strategy_rsi_ma_based_switching_qqq_only,
    strategy_s5,
)
from evaluation_metrics import full_metrics  # noqa: E402
from strategy_result_tables import (  # noqa: E402
    MET_TITLE_KR,
    delta_vs_b1_median_table,
    df_to_github_markdown_table,
    df_to_wide_console_block,
    full_period_metrics_table,
    mc_window_distrib_by_metric,
    mc_window_median_table,
)

OUT_DIR = _ROOT / "03_RESULT" / "sensitivity"
CAP = 100_000

S5_DSA_COMMON = {**S5_ANCHOR_COMMON, "stop_factor": 0.75}

SERIES_META = (
    ("B1", "QQQ 100%"),
    ("B2", "QQQ/TQQQ 50:50 drift"),
    ("B3", "QLD 100%"),
    ("B4", "TQQQ 100%"),
    ("DSA", "strategy_s5"),
    ("RMQQ", "RMABS-QQQ"),
    ("RQLD", "RMABS-QLD"),
)


def slice_metrics(norm_slice: pd.Series) -> dict[str, float]:
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
    wp = (
        float((s < 0).mean() * 100)
        if ulcer_lower_better
        else float((s > 0).mean() * 100)
    )
    return {
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p05": float(s.quantile(0.05)),
        "p95": float(s.quantile(0.95)),
        "win_pct": wp,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="2002-10-01")
    ap.add_argument("--n", type=int, default=3000)
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
    Q, L, T = qqq.loc[common], qld.loc[common], tqqq.loc[common]

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

    codes = tuple(c for c, _ in SERIES_META)

    print("=" * 76)
    print("Monte Carlo  |  B1~B4 + DSA + RMABS-QQQ + RMABS-QLD")
    print(f"  구간: {common[0].date()} ~ {common[-1].date()}  ({Ndays:,}일)")
    print(f"  창 반복={args.n}  seed={args.seed}  최소 거래일={min_len}")
    print("=" * 76)

    print("\n  [STEP 1] 전구간 1회…")
    t0 = time.time()
    nav: dict[str, pd.Series] = {}
    ev_ct: dict[str, int | None] = {}

    nav["B1"] = strategy_buy_and_hold(Q, CAP, "B1")
    ev_ct["B1"] = None
    nav["B2"] = half_half_drift(Q, T, CAP).rename("B2")
    ev_ct["B2"] = None
    nav["B3"] = strategy_buy_and_hold(L, CAP, "B3")
    ev_ct["B3"] = None
    nav["B4"] = strategy_buy_and_hold(T, CAP, "B4")
    ev_ct["B4"] = None

    dsa_series, ev_dsa = strategy_s5(Q, L, T, CAP, series_name="DSA", **S5_DSA_COMMON)
    nav["DSA"] = dsa_series
    ev_ct["DSA"] = len(ev_dsa)

    rq, ev_q = strategy_rsi_ma_based_switching_qqq_only(Q, L, T, CAP, series_name="RMABS-QQQ")
    nav["RMQQ"] = rq
    ev_ct["RMQQ"] = len(ev_q)

    rl, ev_l = strategy_rsi_ma_based_switching(Q, L, T, CAP, series_name="RMABS-QLD")
    nav["RQLD"] = rl
    ev_ct["RQLD"] = len(ev_l)

    MET = ("cagr", "sharpe", "sortino", "mdd", "ulcer")

    df_full = full_period_metrics_table(
        codes=codes,
        series_meta=SERIES_META,
        nav=nav,
        event_counts=ev_ct,
    )
    print("\n")
    print(
        df_to_wide_console_block(
            "① 전구간 일봉 백테스트 — 전략별 절대 지표 (모든 행 한 표)",
            df_full,
        )
    )

    legacy_full: dict[str, dict[str, float]] = {}
    for code in codes:
        m = full_metrics(nav[code])
        legacy_full[code] = {k: float(m[k]) for k in MET}

    challengers_tuple = ("B2", "B3", "B4", "DSA", "RMQQ", "RQLD")

    print(f"  [STEP 1] 표 출력 포함 완료 ({time.time() - t0:.1f}s)\n")

    print("\n  [STEP 2] 동일 무작위 창 슬라이스…")
    rows: list[dict] = []
    t1 = time.time()
    for k in range(args.n):
        si, ei = si_arr[k], ei_arr[k]
        s0 = common[si].strftime("%Y-%m-%d")
        e0 = common[ei].strftime("%Y-%m-%d")
        row = {"iter": int(k), "start": s0, "end": e0, "n_days": int(ei - si + 1)}
        for code in codes:
            w = nav[code].iloc[si : ei + 1] / nav[code].iloc[si]
            sm = slice_metrics(w)
            for met in MET:
                row[f"{code}_{met}"] = sm[met]
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  완료 ({time.time() - t1:.1f}s)\n")

    # vs B1 (QQQ 매수후보유)
    challengers = challengers_tuple
    for pref in challengers:
        for met in MET:
            df[f"{pref}_minus_B1_{met}"] = df[f"{pref}_{met}"] - df[f"B1_{met}"]

    ulcer_keys = {"cagr": False, "sharpe": False, "sortino": False, "mdd": False, "ulcer": True}

    summary: dict = {
        "meta": {
            "full_range": [str(common[0].date()), str(common[-1].date())],
            "n_trading_days_full": int(Ndays),
            "n_mc_windows": args.n,
            "seed": args.seed,
            "min_trading_days": int(min_len),
            "series": [{"code": c, "desc": d} for c, d in SERIES_META],
            "dsa_kwargs": dict(S5_DSA_COMMON),
        },
        "full_period_absolute": {c: legacy_full[c] for c in codes},
        "event_counts": {k: int(v) if v is not None else None for k, v in ev_ct.items()},
        "window_metric_summaries_per_series": {},
        "delta_vs_B1_QQQ": {},
    }

    for code in codes:
        summary["window_metric_summaries_per_series"][code] = {}
        for met in MET:
            stats = summarise_delta(
                df[f"{code}_{met}"],
                ulcer_lower_better=(met == "ulcer"),
            )
            summary["window_metric_summaries_per_series"][code][met] = stats

    for pref in challengers:
        label = {"B2": "B2−B1", "B3": "B3−B1", "B4": "B4−B1", "DSA": "DSA−B1", "RMQQ": "RMQQ−B1", "RQLD": "RQLD−B1"}[pref]
        summary["delta_vs_B1_QQQ"][label] = {}
        for met in MET:
            ulcer_lb = ulcer_keys[met]
            summary["delta_vs_B1_QQQ"][label][met] = summarise_delta(
                df[f"{pref}_minus_B1_{met}"],
                ulcer_lower_better=ulcer_lb,
            )

    tag = f"{common[0].strftime('%Y%m%d')}_{common[-1].strftime('%Y%m%d')}_n{args.n}_s{args.seed}_min{min_len}d"
    stem = f"mc_bench_rmabs_suite_{tag}"
    csv_path = OUT_DIR / f"{stem}.csv"
    json_path = OUT_DIR / f"{stem}.json"
    png_path = OUT_DIR / f"{stem}.png"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    df_mc = mc_window_median_table(
        codes=codes,
        series_meta=SERIES_META,
        summaries=summary["window_metric_summaries_per_series"],
    )
    df_delta = delta_vs_b1_median_table(
        challengers=challengers_tuple,
        series_meta=SERIES_META,
        delta_summary=summary["delta_vs_B1_QQQ"],
    )

    df_by_met = mc_window_distrib_by_metric(
        df,
        codes=codes,
        series_meta=SERIES_META,
        metrics=MET,
    )
    distrib_json = {
        MET_TITLE_KR.get(k, k): v.to_dict(orient="records") for k, v in df_by_met.items()
    }
    blk_4 = "\n\n".join(
        df_to_wide_console_block(
            f"④ 무작위 창 — [{MET_TITLE_KR[m]}] 평균·중앙값·표준편차(전략별)",
            df_by_met[m],
        )
        for m in MET
    )

    tbl_readme_ko = (
        f"[표 설명]\n"
        f"• 전구간: {common[0].date()} ~ {common[-1].date()} ({Ndays}거래일)\n"
        f"• 무작위 창: N={args.n}, seed={args.seed}, 최소 길이={min_len}거래일; "
        f"매 반복에서 7개 전략이 동일 (시작일·종료일) 구간으로 슬라이스됨.\n"
        f"• 표②: 각 창 지표값의 중앙값(p50).\n"
        f"• 표③: 창별 (전략−B1) Δ의 중앙값; "
        f"CAGR·Sharpe·Sortino·MDD 승%=P(Δ>0), Ulcer 승%=P(Δ<0).\n"
        f"• 표④: 창별 지표마다 평균·중앙값·표준편차(표본, ddof=1). CAGR·MDD는 %포인트."
    )

    summary["tables"] = {
        "readme_ko": tbl_readme_ko,
        "full_period_absolute_wide": df_full.to_dict(orient="records"),
        "random_window_metric_median": df_mc.to_dict(orient="records"),
        "random_window_metric_mean_median_std_by_metric": distrib_json,
        "delta_vs_B1_random_window": df_delta.to_dict(orient="records"),
    }

    txt_path = OUT_DIR / f"{stem}_tables.txt"
    txt_path.write_text(
        tbl_readme_ko
        + "\n\n"
        + df_to_wide_console_block(
            "① 전구간 — 전략별 절대 지표",
            df_full,
        )
        + "\n\n"
        + df_to_wide_console_block(
            "② 무작위 창 — 지표 중앙값 전략 비교표",
            df_mc,
        )
        + "\n\n"
        + df_to_wide_console_block(
            "③ B1(QQQ) 대비 — 창별 Δ 중앙값·승률표",
            df_delta,
        )
        + "\n\n"
        + blk_4,
        encoding="utf-8",
    )

    md_sections = [
        "# Monte Carlo — 표 요약 (GFM Markdown)\n",
        tbl_readme_ko.strip(),
        "## ① 전구간 절대 지표",
        df_to_github_markdown_table(df_full),
        "## ② 무작위 창 지표 중앙값(p50)",
        df_to_github_markdown_table(df_mc),
        "## ③ B1(QQQ) 대비 — 창별 Δ 중앙값·승률",
        df_to_github_markdown_table(df_delta),
    ]
    for m in MET:
        md_sections.extend(
            [
                f"## ④ 무작위 창 — {MET_TITLE_KR[m]} (평균·중앙값·표준편차)",
                df_to_github_markdown_table(df_by_met[m]),
                "",
            ]
        )
    md_path = OUT_DIR / f"{stem}_tables.md"
    md_path.write_text("\n\n".join(md_sections), encoding="utf-8")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 플롯: B2,B3,B4,DSA,RMQQ,RQLD 대 B1 ΔCAGR / ΔSharpe
    pair_labels_mc = []
    cols_cagr = []
    for pref in challengers:
        pair_labels_mc.append(
            {"B2": "B2−B1", "B3": "B3−B1", "B4": "B4−B1", "DSA": "DSA−B1", "RMQQ": "RMABS-QQQ−B1", "RQLD": "RMABS-QLD−B1"}[
                pref
            ]
        )
        cols_cagr.append(f"{pref}_minus_B1_cagr")

    fig = plt.figure(figsize=(13.5, 8.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.32)
    for i in range(6):
        r, c = divmod(i, 3)
        ax = fig.add_subplot(gs[r, c])
        x = df[cols_cagr[i]].astype(float).values * 100.0
        ax.hist(x, bins=65, density=True, color="#334155", alpha=0.78, edgecolor="white", linewidth=0.35)
        med = float(np.median(x))
        ax.axvline(med, color="#DC2626", lw=2, label=f"median {med:+.2f}")
        ax.axvline(0.0, color="#9CA3AF", ls="--", lw=1)
        ax.set_title(pair_labels_mc[i] + "  ΔCAGR")
        ax.set_xlabel("ΔCAGR (pct points)")
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Monte Carlo (N={args.n}, seed={args.seed}) — 창별 ΔCAGR vs B1 QQQ",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(
        df_to_wide_console_block(
            "② 무작위 창 — 지표 중앙값 전략 비교표",
            df_mc,
        )
    )
    print(
        df_to_wide_console_block(
            "③ B1(QQQ) 대비 — 창별 Δ 중앙값·승률표",
            df_delta,
        )
    )
    print(blk_4)

    print("\n저장 파일:")
    print(" ", csv_path)
    print(" ", json_path)
    print(" ", png_path)
    print(" ", txt_path, "   (고정폭 텍스트)")
    print(" ", md_path, " (마크다운 미리보기용)")


if __name__ == "__main__":
    main()
