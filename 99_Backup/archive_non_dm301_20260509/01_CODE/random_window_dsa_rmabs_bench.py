"""
4 벤치마크 + DSA(strategy_s5) + RMABS(strategy_rsi_ma_based_switching) 랜덤 구간(최소 2년) 백테스트.

규칙
----
1. 무작위 [start,end], 거래일 ≥ ceil(min_years*252).
2. DSA·RMABS·벤치는 전 구간 1회 ground truth 순자산 → 창에서는 슬라이스 지표계산 (가짜 경계 신호 미반영).
3. 진단용으로 구간만 다시 돌려 이벤트가 전구간 이벤트 집합에 없는 건 카운트(spurious_*).
4. 반복 기본 N=3000.
5. 결과는 콘솔 + CSV/JSON 외 **`03_RESULT/sensitivity/*_report.md`** 한국어 리포트(절대값→상대 요약).

벤치: B1 QQQ100% | B2 QQQ/TQQQ 50:50 drift | B3 QLD100% | B4 TQQQ100%

실행:
  python3 01_CODE/random_window_dsa_rmabs_bench.py
  python3 01_CODE/random_window_dsa_rmabs_bench.py --n 3000 --seed 42 --trail-pcts -0.15
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

from backtest_switching import (  # noqa: E402
    load_extended_daily,
    strategy_buy_and_hold,
    strategy_rsi_ma_based_switching,
    strategy_s5,
)
from benchmark_s2_s5_anchor_table import S5_ANCHOR_COMMON, half_half_drift  # noqa: E402
from evaluation_metrics import full_metrics  # noqa: E402

OUT_DIR = _ROOT / "03_RESULT" / "sensitivity"
ROOT = pd.Timestamp("2002-10-01")
CAP = 100_000
S5_BASE = {**S5_ANCHOR_COMMON, "stop_factor": 0.75}

RMABS_RULE_MD = """- 시그널 자산: QQQ 종가 (RSI14, MA200). 보유 레인: **QQQ / QLD / TQQQ** 각 100% 한 종목.
- 시작: MA200 유효 이전까지 QQQ 보유 후, 최초 MA 유효일에 **종가 > MA200 → QLD**, **종가 ≤ MA200 → QQQ**.
- RSI 30 하향 돌파 후 30 상향 재돌파 완료 + QQQ 종가 > MA200×1.03 이면 **전량 TQQQ**.
- TQQQ 청산 → **전량 QLD**: (1) 종가 < 진입가 (2) 고점 대비 트레일 −15%.
- **QLD** 보유 중(진입 경로 무관) **QQQ 종가 ≤ MA200×0.97** 이면 **전량 QQQ** (RSI 사이클은 유지)."""

BENCH_META = [
    ("B1", "QQQ 100% 매수후보유"),
    ("B2", "QQQ+TQQQ 50:50 드리프트(고정주수, 미리밸런싱)"),
    ("B3", "QLD 100% 매수후보유"),
    ("B4", "TQQQ 100% 매수후보유"),
]

METRICS = ("cagr", "mdd", "sharpe", "sortino", "ulcer")
DELTA_METRIC_ORDER = METRICS


def _trail_label(tr: float) -> str:
    return f"tr{int(round(abs(tr * 100)))}"


def _parse_trail_pcts(s: str) -> list[float]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise SystemExit("trail-pcts 비어 있음")
    return out


def _slice_metrics(ser: pd.Series) -> dict:
    m = full_metrics(ser)
    return {k: float(m[k]) for k in METRICS}


def _summarize_col(s: pd.Series) -> dict:
    s = s.dropna()
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p05": float(s.quantile(0.05)),
        "p50": float(s.quantile(0.50)),
        "p95": float(s.quantile(0.95)),
    }


def _win_rate(d: pd.Series, met: str) -> float:
    if met == "ulcer":
        return float((d < 0).mean() * 100)
    return float((d > 0).mean() * 100)


def _cell_abs(metric: str, value: float) -> str:
    if metric == "cagr":
        return f"{value * 100:+.2f}"
    if metric == "mdd":
        return f"{value * 100:+.2f}"
    if metric in ("sharpe", "sortino"):
        return f"{value:.3f}"
    return f"{value:.2f}"


def _print_absolute_section(summ_abs: dict, trail_pcts: list[float], col_pref_dsa) -> None:
    """summ_abs['by_series'][키][메트릭]['p50'|'mean'|...]."""
    bys = summ_abs["by_series"]
    specs: list[tuple[str, str]] = []
    for code, lab in BENCH_META:
        specs.append((f"{code} ({lab})", code))
    specs.append(("RMABS (QQQ·QLD/TQQ 스위치)", "RMABS"))
    for tr in trail_pcts:
        k = col_pref_dsa(tr).rstrip("_")
        pct_i = int(round(abs(tr * 100)))
        specs.append((f"DSA 고점대비 -{pct_i}% ({k})", k))

    for stat_label, stat_key in (
        ("중앙값 (각 창 지표의 p50)", "p50"),
        ("평균 (각 창 지표의 mean)", "mean"),
    ):
        print(f"\n  ─── 절대값 테이블: {stat_label} ───")
        hdr = (
            f"{'시리즈':<44}"
            f"{'CAGR%':>10}"
            f"{'MDD%':>10}"
            f"{'Sharpe':>10}"
            f"{'Sortino':>10}"
            f"{'Ulcer':>10}"
        )
        print("  " + hdr)
        print("  " + "-" * 103)
        for disp, key in specs:
            if key not in bys:
                continue
            row = [f"{disp:<44}"]
            for met in METRICS:
                v = float(bys[key][met][stat_key])
                row.append(f"{_cell_abs(met, v):>10}")
            print("  " + "".join(row))
    print(
        "  ※ 절대값: 각 랜덤 창 순자산에서 구한 CAGR·MDD·…의 중앙값·평균 요약입니다."
        " CAGR·MDD 숫자는 %(연율/최대낙폭). MDD 음수가 더 깊은 낙폭. Ulcer↓ 유리."
    )


def _print_relative_delta_tables(strat_title: str, delta_for_bench: dict) -> None:
    col_w = 11
    div = "-" * (8 + col_w * 5)
    title = strat_title  # 이름 유지 호환
    print(f"\n  ▶ 【상대값】 {title} — Δ중앙값 (= 전략 − 범치, CAGR·MDD는 %포인트 차)")
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
            md = delta_for_bench[bcode][met]["median_delta"]
            if met == "cagr":
                cell = f"{md * 100:+.2f}"
            elif met == "mdd":
                cell = f"{md * 100:+.2f}"
            elif met in ("sharpe", "sortino"):
                cell = f"{md:+.3f}"
            else:
                cell = f"{md:+.2f}"
            parts.append(f"{cell:>{col_w}}")
        print("  " + "".join(parts))

    print(f"\n  ▶ 【상대값】 {title} — 승률 % (Sh·So·CAGR·MDD: P(Δ>0) | Ulcer: P(Δ<0))")
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
            wp = delta_for_bench[bcode][met]["win_pct_pct"]
            parts.append(f"{wp:>{col_w-1}.1f}%")
        print("  " + "".join(parts))
    print("  ※ Δ = 전략 − 벤치 | MDD 음수 저장 → ΔMDD>0 이면 낙폭 완화 | Ulcer 낮을수록 유리")


def _spec_rows_for_tables(
    trail_pcts: list[float], col_pref_dsa
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for code, lab in BENCH_META:
        rows.append((f"{code} ({lab})", code))
    rows.append(("RMABS (QQQ·QLD/TQQ 스위치)", "RMABS"))
    for tr in trail_pcts:
        k = col_pref_dsa(tr).rstrip("_")
        pct_i = int(round(abs(tr * 100)))
        rows.append((f"DSA 고점대비 -{pct_i}% ({k})", k))
    return rows


def _md_table_abs(bys: dict, specs: list[tuple[str, str]], stat_key: str) -> str:
    hdr = "| 시리즈 | CAGR% | MDD% | Sharpe | Sortino | Ulcer |\n|---|---:|---:|---:|---:|---:|"
    body: list[str] = []
    for disp, key in specs:
        if key not in bys:
            continue
        vals = [_cell_abs(m, float(bys[key][m][stat_key])) for m in METRICS]
        body.append("| " + disp + " | " + " | ".join(vals) + " |")
    return hdr + "\n" + "\n".join(body)


def _delta_interpret_sentence(
    title: str, delta_for_bench: dict, benches: tuple[str, ...] = ("B1", "B2", "B3", "B4")
) -> str:
    lines: list[str] = []
    for bcode in benches:
        cagr_m = delta_for_bench[bcode]["cagr"]["median_delta"]
        shr_m = delta_for_bench[bcode]["sharpe"]["median_delta"]
        mdd_m = delta_for_bench[bcode]["mdd"]["median_delta"]
        ulc_wp = delta_for_bench[bcode]["ulcer"]["win_pct_pct"]
        cagr_wp = delta_for_bench[bcode]["cagr"]["win_pct_pct"]
        lines.append(
            f"- **{title} vs {bcode}**: Δ중앙값 CAGR {cagr_m * 100:+.2f}pp, Sharpe {shr_m:+.3f}, "
            f"MDD {mdd_m * 100:+.2f}pp · 창별 승률(CAGR↑) {cagr_wp:.1f}% / Ulcer 유리 Δ<0 빈도 {ulc_wp:.1f}%."
        )
    return "\n".join(lines)


def write_random_window_md_report(
    path: Path,
    *,
    idx_first: pd.Timestamp,
    idx_last: pd.Timestamp,
    n_mc: int,
    seed: int,
    min_trading_days: int,
    summ_abs: dict,
    rmabs_vs_bench: dict,
    dsa_vs_bench_tr: dict,
    trail_pcts: list[float],
    col_pref_dsa,
    csv_name: str,
    json_name: str,
) -> None:
    bys = summ_abs["by_series"]
    specs = _spec_rows_for_tables(trail_pcts, col_pref_dsa)

    chunks: list[str] = [
        "# 랜덤 구간 백테스트 리포트 (4 벤치 + DSA + RMABS)\n",
        f"- **전구간**: {idx_first.date()} ~ {idx_last.date()}",
        f"- **창 반복**: {n_mc}, **seed**: {seed}, **최소 거래일**: {min_trading_days}",
        f"- **결과 파일**: `{csv_name}`, `{json_name}`",
        "",
        "## A. 방법 요약\n",
        "전 구간 각 시리즈 **ground truth 순자산** 한 번 계산 후, 동일 무작위 `[시작·종료]` 구간만 잘라 "
        "**CAGR · MDD · Sharpe · Sortino · Ulcer**를 구한다.",
        "",
        "**RMABS 정의**\n\n" + RMABS_RULE_MD,
        "",
        "**스퓨리어스 이벤트**: 구간만 따로 재실행한 이벤트 `(날짜, 유형)`이 전구간 집합에 없으면 카운트.",
        " INIT 구간 시작 등은 창별로 발생 시점이 달라 카운트가 크게 나올 수 있다.",
        "",
        "**지표 부호**: MDD·Ulcer는 낮을수록 유리하게 저장된다. 표의 상대값에서 **ΔMDD>0**이면 구간 단위 낙폭 완화, "
        "**Ulcer 승률**은 Δ<0 비율(%)이다.",
        "",
        "## B. 절대값 표 (중앙값 p50)",
        _md_table_abs(bys, specs, "p50"),
        "",
        "## C. 절대값 표 (평균 mean)",
        _md_table_abs(bys, specs, "mean"),
        "",
        "## D. 같은 창에서의 상대 우열 요약\n",
        "### DSA vs 벤치 (Δ중앙값)",
    ]
    for tr in trail_pcts:
        lab = _trail_label(tr)
        dlt = dsa_vs_bench_tr[lab]
        h = f"#### trailing {lab}\n\n"
        tbl = "| vs | ΔCAGR(pp) | ΔMDD(pp) | ΔSharpe | ΔSortino | ΔUlcer |\n|---|---:|---:|---:|---:|---:|\n"
        for bcode, _ in BENCH_META:
            row_cells = []
            for met in METRICS:
                md = float(dlt[bcode][met]["median_delta"])
                if met == "cagr":
                    row_cells.append(f"{md * 100:+.2f}")
                elif met == "mdd":
                    row_cells.append(f"{md * 100:+.2f}")
                elif met in ("sharpe", "sortino"):
                    row_cells.append(f"{md:+.3f}")
                else:
                    row_cells.append(f"{md:+.2f}")
            tbl += "| " + bcode + " | " + " | ".join(row_cells) + " |\n"
        chunks.append(h + tbl + "\n" + _delta_interpret_sentence(f"DSA({lab})", dlt))

    chunks.extend(
        [
            "### RMABS vs 벤치",
            "",
        ]
    )
    rtbl = "| vs | ΔCAGR(pp) | ΔMDD(pp) | ΔSharpe | ΔSortino | ΔUlcer |\n|---|---:|---:|---:|---:|---:|\n"
    for bcode, _ in BENCH_META:
        rs = []
        for met in METRICS:
            md = float(rmabs_vs_bench[bcode][met]["median_delta"])
            if met == "cagr":
                rs.append(f"{md * 100:+.2f}")
            elif met == "mdd":
                rs.append(f"{md * 100:+.2f}")
            elif met in ("sharpe", "sortino"):
                rs.append(f"{md:+.3f}")
            else:
                rs.append(f"{md:+.2f}")
        rtbl += "| " + bcode + " | " + " | ".join(rs) + " |\n"
    chunks.append(rtbl)
    chunks.append("")
    chunks.append(_delta_interpret_sentence("RMABS", rmabs_vs_bench))
    chunks.append("")
    chunks.append(
        "## E. 해석 가이드 (한 줄)\n"
        "- 벤치 **B4(TQQQ)**·**B2(반반)**와 비교했을 때 **CAGR 우위**만으로는 레버 노출 차이 때문에 신호 과대평가가 나기 쉬우니, "
        "**Sharpe/Sortino**와 **MDD·Ulcer**를 함께 보라.\n"
        "- **RMABS**가 **B3(QLD 단순)** 대비 CAGR·Sharpe 우위가 줄면 RSI·레인 규칙이 변동 소비 비용으로 작용했다고 볼 수 있다."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(chunks), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(ROOT.date()))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-years", type=float, default=2.0)
    ap.add_argument("--trail-pcts", type=str, default="-0.15")
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
    m = Q.index >= pd.Timestamp(args.root.strip())
    Q, L, T = Q.loc[m], L.loc[m], T.loc[m]
    idx = Q.index
    n_days = len(idx)
    min_len = max(2, int(np.ceil(args.min_years * 252)))

    if n_days < min_len:
        raise SystemExit(f"데이터 {n_days}일 < 최소 {min_len}일")

    b1 = strategy_buy_and_hold(Q, CAP, "B1")
    b2 = half_half_drift(Q, T, CAP)
    b3 = strategy_buy_and_hold(L, CAP, "B3")
    b4 = strategy_buy_and_hold(T, CAP, "B4")

    rmabs_full, ev_rm_full = strategy_rsi_ma_based_switching(Q, L, T, CAP, series_name="RMABS")
    ev_rm_df = ev_rm_full.copy()
    if not ev_rm_df.empty:
        ev_rm_df["Date"] = pd.to_datetime(ev_rm_df["Date"])
    rm_keys = (
        set(zip(ev_rm_df["Date"].astype(str), ev_rm_df["type"].astype(str)))
        if not ev_rm_df.empty
        else set()
    )

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

    def col_pref_dsa(tr: float) -> str:
        return f"DSA_{_trail_label(tr)}_" if multi_trail else "DSA_"

    windows = []
    for _ in range(args.n):
        s_i = int(rng.integers(0, n_days - min_len + 1))
        e_i = int(rng.integers(s_i + min_len - 1, n_days))
        windows.append((idx[s_i], idx[e_i]))

    spurious_dsa = {tr: [] for tr in trail_pcts}
    spurious_rm: list[int] = []
    rows = []

    for d0, d1 in windows:
        s_b1 = b1.loc[d0:d1]
        if len(s_b1) < 10:
            continue
        s_b2, s_b3, s_b4 = b2.loc[d0:d1], b3.loc[d0:d1], b4.loc[d0:d1]
        mm = {
            "B1": _slice_metrics(s_b1),
            "B2": _slice_metrics(s_b2),
            "B3": _slice_metrics(s_b3),
            "B4": _slice_metrics(s_b4),
        }
        row: dict = {
            "start": str(d0.date()),
            "end": str(d1.date()),
            "n_days": int(len(s_b1)),
        }
        for code in ("B1", "B2", "B3", "B4"):
            for k, v in mm[code].items():
                row[f"{code}_{k}"] = v

        rm_s = rmabs_full.loc[d0:d1]
        mrm = _slice_metrics(rm_s)
        for k, v in mrm.items():
            row[f"RMABS_{k}"] = v

        Qw, Lw, Tw = Q.loc[d0:d1], L.loc[d0:d1], T.loc[d0:d1]
        _, ev_rm_w = strategy_rsi_ma_based_switching(Qw, Lw, Tw, CAP)
        if ev_rm_w.empty:
            spur_rm = 0
        else:
            er = ev_rm_w.copy()
            er["Date"] = pd.to_datetime(er["Date"])
            spur_rm = sum(
                1
                for _, r in er.iterrows()
                if (str(pd.Timestamp(r["Date"])), str(r["type"])) not in rm_keys
            )
        spurious_rm.append(spur_rm)
        row["spurious_rmabs_subrun_events"] = spur_rm

        for tr in trail_pcts:
            pref = col_pref_dsa(tr)
            sd = p5_by[tr].loc[d0:d1]
            for k, v in _slice_metrics(sd).items():
                row[f"{pref}{k}"] = v

            kw_w = {**S5_BASE, "trailing_stop_pct": tr}
            _, ev_w = strategy_s5(Qw, Lw, Tw, CAP, **kw_w)
            fk = keys_by[tr]
            if ev_w.empty:
                spur = 0
            else:
                ew = ev_w.copy()
                ew["Date"] = pd.to_datetime(ew["Date"])
                spur = sum(
                    1
                    for _, r in ew.iterrows()
                    if (str(pd.Timestamp(r["Date"])), str(r["type"])) not in fk
                )
            spurious_dsa[tr].append(spur)
            if multi_trail:
                row[f"spurious_dsa_{_trail_label(tr)}"] = spur
            else:
                row["spurious_dsa_subrun_events"] = spur

        rows.append(row)

    df = pd.DataFrame(rows)

    # 요약 블록
    summ_abs: dict = {
        "n_windows": len(df),
        "trail_pcts_dsa": trail_pcts,
        "spurious_dsa_per_trail": {},
        "spurious_rmabs": _summarize_col(pd.Series(spurious_rm, dtype=float)),
        "note": (
            "지표는 전구간 ground truth 순자산 슬라이스 기준."
            " spurious_* 는 구간단독 재실행 이벤트가 전구간 집합에 없는 건 수."
        ),
    }
    for tr in trail_pcts:
        summ_abs["spurious_dsa_per_trail"][_trail_label(tr)] = _summarize_col(
            pd.Series(spurious_dsa[tr], dtype=float)
        )

    bench_keys = [c for c, _ in BENCH_META]
    summ_abs["by_series"] = {}
    for code in bench_keys + ["RMABS"]:
        prefix = code + "_"
        summ_abs["by_series"][code] = {}
        for met in METRICS:
            summ_abs["by_series"][code][met] = _summarize_col(df[f"{code}_{met}"])
    for tr in trail_pcts:
        pref = col_pref_dsa(tr).rstrip("_")
        summ_abs["by_series"][pref] = {}
        for met in METRICS:
            c = col_pref_dsa(tr) + met
            summ_abs["by_series"][pref][met] = _summarize_col(df[c])

    def build_delta(pref: str) -> dict:
        out: dict = {}
        for bcode, _ in BENCH_META:
            out[bcode] = {}
            for met in METRICS:
                d = df[pref + met] - df[f"{bcode}_{met}"]
                out[bcode][met] = {
                    "median_delta": float(d.median()),
                    "mean_delta": float(d.mean()),
                    "win_pct_pct": _win_rate(d, met),
                }
        return out

    rmabs_vs_bench = build_delta("RMABS_")
    dsa_vs_bench_tr: dict = {}
    for tr in trail_pcts:
        dsa_vs_bench_tr[_trail_label(tr)] = build_delta(col_pref_dsa(tr))

    tag_tr = "_".join(_trail_label(t) for t in trail_pcts)
    tag = f"{idx[0].strftime('%Y%m%d')}_{idx[-1].strftime('%Y%m%d')}_n{args.n}_s{args.seed}_dsa_{tag_tr}_rmabs"
    csv_path = OUT_DIR / f"random_window_dsa_rmabs_vs_bench_{tag}.csv"
    json_path = OUT_DIR / f"random_window_dsa_rmabs_vs_bench_{tag}.json"
    report_path = OUT_DIR / f"random_window_dsa_rmabs_vs_bench_{tag}_report.md"

    meta_json = {
        "벤치마크": [{"code": c, "설명": d} for c, d in BENCH_META],
        "strategies": [
            {"id": "DSA", "implementation": "strategy_s5"},
            {"id": "RMABS", "implementation": "strategy_rsi_ma_based_switching"},
        ],
        "full_period": [str(idx[0].date()), str(idx[-1].date())],
        "n_mc_windows": len(df),
        "seed": args.seed,
        "min_trading_days": min_len,
        "summary_absolute_mean_per_window_by_series": summ_abs,
        "dsa_vs_bench": dsa_vs_bench_tr,
        "rmabs_vs_bench": rmabs_vs_bench,
        "win_rate_rules": {
            "cagr": "P(Δ>0)",
            "mdd": "P(Δ>0)",
            "sharpe": "P(Δ>0)",
            "sortino": "P(Δ>0)",
            "ulcer": "P(Δ<0)",
        },
        "csv": csv_path.name,
        "report_md": report_path.name,
        "rmabs_rule_markdown": RMABS_RULE_MD,
    }

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2, ensure_ascii=False)

    write_random_window_md_report(
        report_path,
        idx_first=idx[0],
        idx_last=idx[-1],
        n_mc=len(df),
        seed=args.seed,
        min_trading_days=min_len,
        summ_abs=summ_abs,
        rmabs_vs_bench=rmabs_vs_bench,
        dsa_vs_bench_tr=dsa_vs_bench_tr,
        trail_pcts=trail_pcts,
        col_pref_dsa=col_pref_dsa,
        csv_name=csv_path.name,
        json_name=json_path.name,
    )

    print("=" * 76)
    print("4 벤치 + DSA + RMABS 랜덤구간 비교")
    print(f"  전구간 {idx[0].date()} ~ {idx[-1].date()}  |  창 {args.n}  seed {args.seed}  최소일 {min_len}")
    print("=" * 76)
    print("가짜 DSA 이벤트 / 창:", {k: v["mean"] for k, v in summ_abs["spurious_dsa_per_trail"].items()})
    print("가짜 RMABS 이벤트 / 창 mean:", summ_abs["spurious_rmabs"]["mean"])

    print("\n" + "=" * 96)
    print(" (1) 절대값 — 동일 조건으로 뽑힌 랜덤 창별 지표 분포 요약")
    print("=" * 96)
    _print_absolute_section(summ_abs, trail_pcts, col_pref_dsa)

    print("\n" + "=" * 96)
    print(" (2) 상대값 — 같은 창에서 「전략 − 범치」(Δ중앙값 + 승률)")
    print("=" * 96)

    for tr in trail_pcts:
        lab = _trail_label(tr)
        if multi_trail:
            print(f"\n--- DSA trailing {lab} ---")
        _print_relative_delta_tables(
            f"DSA − 벤치 ({lab})" if multi_trail else "DSA − 벤치", dsa_vs_bench_tr[lab]
        )

    _print_relative_delta_tables("RMABS − 벤치", rmabs_vs_bench)

    print("\nCSV:", csv_path)
    print("JSON:", json_path)
    print("리포트:", report_path)


if __name__ == "__main__":
    main()
