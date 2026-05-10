"""
4벤치마크 + Dynamic Switching Algorithm (DSA, 기존 S5 앵커) 랜덤 구간(최소 2년) 반복 백테스트.

• BM1=B1 = QQQ 단순 보유 100%
• BM2=B2 = QQQ/TQQQ 50:50 드리프트(고정주수, 미리밸런싱)
• BM3=B3 = QLD 단순 보유 100%
• BM4=B4 = TQQQ 단순 보유 100%

• `--trail-pcts` 로 DSA 트레일 스탑(고점 대비)을 복수 지정 가능 (예: -0.15,-0.20).
  동일 seed·동일 창 목록으로 각 트레일 설정을 비교.

실행:
  python3 01_CODE/random_window_bench_s5.py
  python3 01_CODE/random_window_bench_s5.py --n 3000 --seed 42 --trail-pcts -0.15,-0.20
"""

from __future__ import annotations

DSA_DISPLAY = "Dynamic Switching Algorithm (DSA)"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import (  # noqa: E402
    load_extended_daily,
    strategy_buy_and_hold,
    strategy_s5,
)
from benchmark_s2_s5_anchor_table import (  # noqa: E402
    S5_ANCHOR_COMMON,
    half_half_drift,
)
from evaluation_metrics import full_metrics  # noqa: E402

OUT_DIR = _ROOT / "03_RESULT" / "sensitivity"
ROOT = pd.Timestamp("2002-10-01")
CAP = 100_000
S5_BASE = {**S5_ANCHOR_COMMON, "stop_factor": 0.75}

BENCH_META = [
    ("B1", "QQQ 100% 매수후보유"),
    ("B2", "QQQ+TQQQ 50:50 드리프트(고정주수, 미리밸런싱)"),
    ("B3", "QLD 100% 매수후보유"),
    ("B4", "TQQQ 100% 매수후보유"),
]

DELTA_METRIC_ORDER = ("cagr", "mdd", "sharpe", "sortino", "ulcer")


def _format_median_cell(met: str, md: float) -> str:
    if met == "cagr":
        return f"{md * 100:+.2f}"
    if met == "mdd":
        return f"{md * 100:+.2f}"
    if met in ("sharpe", "sortino"):
        return f"{md:+.3f}"
    return f"{md:+.2f}"


def _print_median_and_win_tables(delta_by_trail: dict, trail_lab: str) -> None:
    """DSA−벤치: 창별 Δ의 중앙값 표 + 승률 표 (콘솔)."""
    dbt = delta_by_trail[trail_lab]
    col_w = 11
    div = "-" * (8 + col_w * 5)

    print(f"\n  [{trail_lab}] 표1 — Δ=DSA−벤치 의 중앙값 (CAGR·MDD 열은 %p)")
    hdr = (
        f"{'vs':<8}"
        f"{'ΔCAGR':>{col_w}}"
        f"{'ΔMDD':>{col_w}}"
        f"{'ΔSharpe':>{col_w}}"
        f"{'ΔSortino':>{col_w}}"
        f"{'ΔUlcer':>{col_w}}"
    )
    print("  " + hdr)
    print("  " + div)
    for bcode, _ in BENCH_META:
        parts = [f"{bcode:<8}"]
        for met in DELTA_METRIC_ORDER:
            md = dbt[bcode][met]["median_delta"]
            parts.append(f"{_format_median_cell(met, md):>{col_w}}")
        print("  " + "".join(parts))

    print(f"\n  [{trail_lab}] 표2 — 승률 % (CAGR·MDD·Sh·So: P(Δ>0) | Ulcer: P(Δ<0))")
    hdr2 = (
        f"{'vs':<8}"
        f"{'CAGR':>{col_w}}"
        f"{'MDD':>{col_w}}"
        f"{'Sharpe':>{col_w}}"
        f"{'Sortino':>{col_w}}"
        f"{'Ulcer':>{col_w}}"
    )
    print("  " + hdr2)
    print("  " + div)
    for bcode, _ in BENCH_META:
        parts = [f"{bcode:<8}"]
        for met in DELTA_METRIC_ORDER:
            wp = dbt[bcode][met]["win_pct_pct"]
            parts.append(f"{wp:>{col_w-1}.1f}%")
        print("  " + "".join(parts))
    print("  ※ Ulcer 열: 승률 = P(Δ<0). 나머지 열: 승률 = P(Δ>0).")


def _trail_label(tr: float) -> str:
    return f"tr{int(round(abs(tr * 100)))}"


def _parse_trail_pcts(s: str) -> list[float]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise SystemExit("trail-pcts 비어 있음")
    return out


def _slice_metrics(ser: pd.Series) -> dict:
    m = full_metrics(ser)
    return {
        "cagr": float(m["cagr"]),
        "mdd": float(m["mdd"]),
        "sharpe": float(m["sharpe"]),
        "sortino": float(m["sortino"]),
        "ulcer": float(m["ulcer"]),
    }


def _summarize_col(s: pd.Series) -> dict:
    s = s.dropna()
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p05": float(s.quantile(0.05)),
        "p50": float(s.quantile(0.50)),
        "p95": float(s.quantile(0.95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(ROOT.date()))
    ap.add_argument("--n", type=int, default=3000, help="랜덤 구간 반복 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-years", type=float, default=2.0, help="최소 구간 길이(년)")
    ap.add_argument(
        "--trail-pcts",
        type=str,
        default="-0.15",
        help="콤마 구분 DSA trailing_stop_pct (예: -0.15,-0.20)",
    )
    args = ap.parse_args()

    trail_pcts = _parse_trail_pcts(args.trail_pcts)
    multi_trail = len(trail_pcts) > 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    qqq = load_extended_daily("QQQ")
    qld = load_extended_daily("QLD")
    tqq = load_extended_daily("TQQQ")
    c = qqq.index.intersection(qld.index).intersection(tqq.index)
    Q, L, T = qqq.loc[c], qld.loc[c], tqq.loc[c]
    m = Q.index >= pd.Timestamp(ROOT)
    Q, L, T = Q.loc[m], L.loc[m], T.loc[m]
    idx = Q.index
    n_days = len(idx)
    min_len = max(2, int(np.ceil(args.min_years * 252)))

    if n_days < min_len:
        raise SystemExit(f"데이터 길이 {n_days}일 < 최소 구간 길이 {min_len}거래일")

    b1 = strategy_buy_and_hold(Q, CAP, "B1")
    b2 = half_half_drift(Q, T, CAP)
    b3 = strategy_buy_and_hold(L, CAP, "B3")
    b4 = strategy_buy_and_hold(T, CAP, "B4")

    p5_by: dict[float, pd.Series] = {}
    ev_by: dict[float, pd.DataFrame] = {}
    keys_by: dict[float, set[tuple[str, str]]] = {}
    for tr in trail_pcts:
        kw = {**S5_BASE, "trailing_stop_pct": tr}
        p5, ev = strategy_s5(Q, L, T, CAP, **kw)
        p5_by[tr] = p5
        ev_df = ev.copy()
        ev_df["Date"] = pd.to_datetime(ev_df["Date"])
        ev_by[tr] = ev_df
        keys_by[tr] = (
            set(zip(ev_df["Date"].astype(str), ev_df["type"].astype(str)))
            if not ev_df.empty
            else set()
        )

    def col_pref(tr: float) -> str:
        if multi_trail:
            return f"DSA_{_trail_label(tr)}_"
        return "DSA_"

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for _it in range(args.n):
        s_i = int(rng.integers(0, n_days - min_len + 1))
        e_i = int(rng.integers(s_i + min_len - 1, n_days))
        windows.append((idx[s_i], idx[e_i]))

    rows = []
    spurious_by_tr: dict[float, list[int]] = {tr: [] for tr in trail_pcts}

    for d0, d1 in windows:
        s_b1 = b1.loc[d0:d1]
        s_b2 = b2.loc[d0:d1]
        s_b3 = b3.loc[d0:d1]
        s_b4 = b4.loc[d0:d1]
        if len(s_b1) < 10:
            continue

        m_b1 = _slice_metrics(s_b1)
        m_b2 = _slice_metrics(s_b2)
        m_b3 = _slice_metrics(s_b3)
        m_b4 = _slice_metrics(s_b4)

        row = {
            "start": str(d0.date()),
            "end": str(d1.date()),
            "n_days": int(len(s_b1)),
        }
        for tag, mm in ("B1", m_b1), ("B2", m_b2), ("B3", m_b3), ("B4", m_b4):
            for k, v in mm.items():
                row[f"{tag}_{k}"] = v

        Qw, Lw, Tw = Q.loc[d0:d1], L.loc[d0:d1], T.loc[d0:d1]

        for tr in trail_pcts:
            pref = col_pref(tr)
            s_p5 = p5_by[tr].loc[d0:d1]
            m5 = _slice_metrics(s_p5)
            for k, v in m5.items():
                row[f"{pref}{k}"] = v

            kw_w = {**S5_BASE, "trailing_stop_pct": tr}
            _, ev_w = strategy_s5(Qw, Lw, Tw, CAP, **kw_w)
            fk = keys_by[tr]
            if ev_w.empty:
                spur = 0
            else:
                ev_w = ev_w.copy()
                ev_w["Date"] = pd.to_datetime(ev_w["Date"])
                spur = sum(
                    1
                    for _, r in ev_w.iterrows()
                    if (str(r["Date"]), str(r["type"])) not in fk
                )
            spurious_by_tr[tr].append(spur)
            col_sp = f"spurious_{_trail_label(tr)}" if multi_trail else "spurious_subrun_events"
            row[col_sp] = spur

        rows.append(row)

    # Rebuild rows for spurious multi: the loop above wrongly sets row[col_sp] each tr - good unique keys
    # But single trail: only one spurious column - done inside loop once per tr - for single trail only one iteration sets row["spurious_subrun_events"]

    df = pd.DataFrame(rows)

    METRICS = ("cagr", "mdd", "sharpe", "sortino", "ulcer")
    bench_keys = [(code, code) for code, _ in BENCH_META]

    summ_abs: dict = {
        "n_windows": len(df),
        "trail_pcts": trail_pcts,
        "spurious_per_trail": {},
    }
    for tr in trail_pcts:
        arr = np.array(spurious_by_tr[tr], dtype=np.int32)
        lab = _trail_label(tr)
        summ_abs["spurious_per_trail"][lab] = {
            "mean": float(arr.mean()),
            "p50": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": int(arr.max()),
        }

    summ_abs["by_series"] = {}
    for col_prefix, _lab in bench_keys:
        summ_abs["by_series"][col_prefix] = {}
        for met in METRICS:
            c = f"{col_prefix}_{met}"
            if c in df.columns:
                summ_abs["by_series"][col_prefix][met] = _summarize_col(df[c])

    for tr in trail_pcts:
        pref = col_pref(tr)
        lab = _trail_label(tr)
        key = pref.rstrip("_")
        summ_abs["by_series"][key] = {}
        for met in METRICS:
            c = f"{pref}{met}"
            if c in df.columns:
                summ_abs["by_series"][key][met] = _summarize_col(df[c])

    # DSA − 벤치: 랜덤 창별 차이 요약 (중앙값, 승률)
    delta_by_trail: dict = {}
    for tr in trail_pcts:
        pref_tr = col_pref(tr)
        lab = _trail_label(tr)
        delta_by_trail[lab] = {}
        for bcode, _ in BENCH_META:
            delta_by_trail[lab][bcode] = {}
            for met in METRICS:
                d = df[f"{pref_tr}{met}"] - df[f"{bcode}_{met}"]
                med = float(d.median())
                if met in ("cagr", "sharpe", "sortino"):
                    wr = float((d > 0).mean() * 100)
                elif met == "mdd":
                    # MDD는 음수(max_drawdown): |DD|↓ = 값↑ → DSA가 덜 음수면 Δ>0
                    wr = float((d > 0).mean() * 100)
                else:
                    # ulcer: 양수 지표, 낮을수록 좋음 → Δ<0 이면 DSA 우위
                    wr = float((d < 0).mean() * 100)
                delta_by_trail[lab][bcode][met] = {
                    "median_delta": med,
                    "win_pct_pct": wr,
                }

    bench_legend = [{"code": c, "설명": desc} for c, desc in BENCH_META]
    tag_tr = "_".join(_trail_label(t) for t in trail_pcts)
    tag = (
        f"{idx[0].strftime('%Y%m%d')}_{idx[-1].strftime('%Y%m%d')}_n{args.n}_s{args.seed}_trail{tag_tr}"
    )

    ev_paths = {}
    for tr in trail_pcts:
        evp = OUT_DIR / f"random_window_dsa_full_events_{tag}_{_trail_label(tr)}.csv"
        ev_df = ev_by[tr]
        if not ev_df.empty:
            ev_df.to_csv(evp, index=False, encoding="utf-8-sig")
        ev_paths[_trail_label(tr)] = str(evp.name) if not ev_df.empty else None

    meta = {
        "strategy_name": DSA_DISPLAY,
        "implementation": "strategy_s5 (backtest_switching)",
        "delta_definition": "모든 Δ = DSA − 벤치 (동일 랜덤 창)",
        "win_rate_rules": {
            "cagr": "P(Δ > 0)",
            "sharpe": "P(Δ > 0)",
            "sortino": "P(Δ > 0)",
            "mdd": "P(Δ > 0) — MDD는 max_drawdown 음수 저장, |DD| 작을수록 0에 가깝게 덜 음수",
            "ulcer": "P(Δ < 0) — Ulcer는 양수, 낮을수록 좋음",
        },
        "벤치마크": bench_legend,
        "trail_pcts_dsa": trail_pcts,
        "full_period": [str(idx[0].date()), str(idx[-1].date())],
        "n_trading_days_full": int(n_days),
        "min_years": args.min_years,
        "min_trading_days": min_len,
        "summary_absolute_per_window": summ_abs,
        "dsa_vs_bench_median_delta": delta_by_trail,
        "full_dsa_events_by_trail_csv": ev_paths,
    }

    csv_path = OUT_DIR / f"random_window_dsa_vs_bench_{tag}.csv"
    json_path = OUT_DIR / f"random_window_dsa_vs_bench_{tag}.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("전체 구간:", idx[0].date(), "~", idx[-1].date())
    print(f"{DSA_DISPLAY} trailing_stop_pct:", trail_pcts)
    print("랜덤 창:", args.n, "| seed:", args.seed, "| 최소 거래일:", min_len)
    print("\n[벤치마크]")
    for c, d in BENCH_META:
        print(f"  {c}: {d}")

    print("\n=== 가짜 이벤트 수 / 창 (구간 단독 vs 해당 트레일 전체 이벤트 집합) ===")
    for tr in trail_pcts:
        lab = _trail_label(tr)
        sp = summ_abs["spurious_per_trail"][lab]
        print(f"  {_trail_label(tr)}: mean {sp['mean']:.2f}, p50 {sp['p50']:.0f}, max {sp['max']}")

    print("\n=== 창별 지표 절대값 — 평균 (mean) ===")
    for col_prefix, _ in bench_keys:
        ss = summ_abs["by_series"][col_prefix]
        print(
            f"{col_prefix:<10} CAGR {ss['cagr']['mean']:+.2%} MDD {ss['mdd']['mean']:.2%} "
            f"Sh {ss['sharpe']['mean']:.3f} So {ss['sortino']['mean']:.3f} U {ss['ulcer']['mean']:.2f}"
        )
    for tr in trail_pcts:
        key = col_pref(tr).rstrip("_")
        ss = summ_abs["by_series"][key]
        pct = int(round(abs(tr * 100)))
        lbl = f"DSA 고점대비 -{pct}%" if multi_trail else "DSA"
        print(
            f"{lbl:<10} CAGR {ss['cagr']['mean']:+.2%} MDD {ss['mdd']['mean']:.2%} "
            f"Sh {ss['sharpe']['mean']:.3f} So {ss['sortino']['mean']:.3f} U {ss['ulcer']['mean']:.2f}"
        )

    print("\n=== DSA − 벤치: 중앙값 & 승률 표 ===")
    print("  Δ = DSA − 벤치 (동일 창) | MDD는 음수 저장 → |낙폭| 작을수록 ΔMDD>0")
    for tr in trail_pcts:
        lab = _trail_label(tr)
        if len(trail_pcts) > 1:
            print(f"\n  --- trail {lab} ---")
        _print_median_and_win_tables(delta_by_trail, lab)

    if multi_trail:
        print(f"\n=== DSA 트레일 간 mean 비교 (동일 {args.n}창) ===")
        base_tr = trail_pcts[0]
        bkey = col_pref(base_tr).rstrip("_")
        for tr in trail_pcts[1:]:
            k0 = col_pref(base_tr).rstrip("_")
            k1 = col_pref(tr).rstrip("_")
            print(f"\n{_trail_label(tr)} vs {_trail_label(base_tr)} (최종자산 비율 분포는 생략, CAGR mean 차):")
            for met in METRICS:
                m0 = summ_abs["by_series"][k0][met]["mean"]
                m1 = summ_abs["by_series"][k1][met]["mean"]
                if met == "cagr":
                    print(f"  {met}: {m1-m0:+.4f} (절대 차)")
                elif met == "mdd":
                    print(f"  {met}: {m1-m0:+.4f}")
                else:
                    print(f"  {met}: {m1-m0:+.4f}")

    print("\nCSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
