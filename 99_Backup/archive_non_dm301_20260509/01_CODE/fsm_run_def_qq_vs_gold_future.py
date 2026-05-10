"""
FSM 백테스트 비교 고정 레인:

- sig_asset=QQQ, nor_asset=QLD, agg_asset=TQQQ
- ``def_asset`` 세 가지: **QQQ** · **금(선물 우선 GC=F · 실패 시 GLD)** · **현금**(방어종가 상수 1, 이자 없음)
- 같은 달력에서 **RMABS-QLD** · **RMABS-QQQ** (`backtest_switching`, 기본 파라·warmup 없음) 및 **벤치 4종** 지표 포함.

동일 교집합 일자(``--root`` 이후 ∩ QQQ·QLD·TQQQ·금 선물 종가 존재일)에서 각각 실행.

출력 기본 디렉터리: ``03_RESULT/sensitivity/``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import (  # noqa: E402
    load_extended_daily,
    strategy_rsi_ma_based_switching,
    strategy_rsi_ma_based_switching_qqq_only,
)
from rmabs_gold_simulation import load_gold_series  # noqa: E402

from fsm_four_asset_strategy import (  # noqa: E402
    CAPITAL_START,
    DEFAULT_TRAIL,
    MA_WINDOW,
    _core_metrics,
    benchmark_panel_for_slice,
    build_bench_source_for_index,
    get_or_build_ma200,
    run_fsm_backtest,
    _align_five,
)


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, (pd.Timestamp,)):
        return str(x.date())
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return _jsonable(x.item())
        except Exception:
            pass
    if isinstance(x, float):
        return float(x)
    return x


def flat_cash_ohlc(ix: pd.DatetimeIndex) -> pd.DataFrame:
    """방어자산 ``현금``: 일별 종가 1 고정(QQQ·QLD·TQQQ 대비 무위험이자 미반영 단순 모델)."""
    ix = ix.sort_values()
    one = pd.Series(1.0, index=ix, dtype=float)
    return pd.DataFrame({"Open": one, "High": one, "Low": one, "Close": one})


def _bench_metrics(ix: pd.DatetimeIndex) -> dict[str, dict[str, float]]:
    src = build_bench_source_for_index(ix)
    pans = benchmark_panel_for_slice(ix, src, CAPITAL_START)
    out: dict[str, dict[str, float]] = {}
    mapping = (
        ("QQQ_100_bh", "QQQ_bh"),
        ("QLD_100_bh", "QLD_bh"),
        ("mix_QQQ50_TQQQ50_bh", "mix_qq50_tqq50_bh"),
        ("TQQQ_100_bh", "TQQQ_bh"),
    )
    for lab, kk in mapping:
        s = pans.get(kk)
        if s is not None and len(s) >= 2:
            out[lab] = _core_metrics(s)
    return out


# 요약 MD에 항상 같은 순서·한글 라벨로 벤치 4종을 둔다.
BENCHMARK_DISPLAY_ORDER: tuple[tuple[str, str], ...] = (
    ("QQQ_100_bh", "벤치 · QQQ 100% buy & hold"),
    ("QLD_100_bh", "벤치 · QLD 100% buy & hold"),
    ("mix_QQQ50_TQQQ50_bh", "벤치 · 초기 QQQ 50% + TQQQ 50% 매수 후 보유(무리밸런스)"),
    ("TQQQ_100_bh", "벤치 · TQQQ 100% buy & hold"),
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="FSM def(QQQ·금·현금) + RMABS-QLD·RMABS-QQQ + 벤치·동일 교집합"
    )
    ap.add_argument("--root", default="2002-10-01")
    ap.add_argument("--trail", type=float, default=DEFAULT_TRAIL)
    ap.add_argument(
        "--out-dir",
        default="",
        help="비우면 03_RESULT/sensitivity",
    )
    args = ap.parse_args()

    out_root = Path(args.out_dir) if args.out_dir.strip() else _ROOT / "03_RESULT/sensitivity"
    out_root.mkdir(parents=True, exist_ok=True)

    root = pd.Timestamp(args.root.strip())
    sg = load_extended_daily("QQQ")
    ng = load_extended_daily("QLD")
    ag = load_extended_daily("TQQQ")

    ma_full, csv_cache = get_or_build_ma200(sg, "QQQ", force_rebuild=False)

    ix0 = sg.index.intersection(ng.index).intersection(ag.index).sort_values()
    ix0 = ix0[ix0 >= root]

    gold_df, gold_note = load_gold_series(ix0[0], ix0[-1])
    ix_common = ix0.intersection(gold_df.index).sort_values()
    if len(ix_common) < MA_WINDOW:
        raise SystemExit(f"금 포함 교집합 거래일 {len(ix_common)} < MA200 {MA_WINDOW}")

    g_i = gold_df.reindex(ix_common)
    sg_c = sg.reindex(ix_common)
    dg_qqq = sg_c
    ng_c = ng.reindex(ix_common)
    ag_c = ag.reindex(ix_common)
    mav = ma_full.reindex(ix_common)
    dcash = flat_cash_ohlc(ix_common)

    aligned_def_qqq = _align_five(sg_c, g_i, dg_qqq, ng_c, ag_c)
    aligned_def_gold = _align_five(sg_c, g_i, g_i, ng_c, ag_c)
    aligned_def_cash = _align_five(sg_c, g_i, dcash, ng_c, ag_c)

    nav_q, _ev_q = run_fsm_backtest(aligned_def_qqq, mav, trail_stop=args.trail, initial_capital=CAPITAL_START)
    nav_g, _ev_g = run_fsm_backtest(aligned_def_gold, mav, trail_stop=args.trail, initial_capital=CAPITAL_START)
    nav_c, _ev_c = run_fsm_backtest(aligned_def_cash, mav, trail_stop=args.trail, initial_capital=CAPITAL_START)

    mq, mg, mc = _core_metrics(nav_q), _core_metrics(nav_g), _core_metrics(nav_c)
    bq = _bench_metrics(ix_common)

    Qa, La, Ta = sg.reindex(ix_common), ng.reindex(ix_common), ag.reindex(ix_common)
    nav_rm_qld, ev_rm_qld = strategy_rsi_ma_based_switching(
        Qa, La, Ta, CAPITAL_START, series_name="RMABS-QLD"
    )
    nav_rm_qqq, ev_rm_qqq = strategy_rsi_ma_based_switching_qqq_only(
        Qa, La, Ta, CAPITAL_START, series_name="RMABS-QQQ"
    )
    mr_qld, mr_qqq = _core_metrics(nav_rm_qld), _core_metrics(nav_rm_qqq)

    rmabs_note = (
        "신호=QQQ 종가(RSI·MA200). "
        "RMABS-QLD: 규칙0·청산 후 QQQ 전환 포함—`strategy_rsi_ma_based_switching`. "
        "RMABS-QQQ: 시그널·방어 모두 QQQ—`strategy_rsi_ma_based_switching_qqq_only`. "
        "warmup_hold_cash=False (첫일부터 종가 매수 규격)."
    )

    cash_def_note = "방어 상태 가격종가 상수 1.0 USD(단위)·일중 변동 없음·이자·인플레 없음(FSM 규격상 현금)"

    slug = (
        f"fsm_def3way_rmabs_QQQ_GoldPX_Cash_{pd.Timestamp(ix_common[0]).strftime('%Y%m%d')}_"
        f"{pd.Timestamp(ix_common[-1]).strftime('%Y%m%d')}"
    )
    stem = slug

    def metric_diff(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
        keys = ("cagr", "mdd", "sharpe", "sortino", "ulcer", "total_return")
        return {k: a[k] - b[k] for k in keys if k in a and k in b}

    blob = _jsonable(
        {
            "meta": {
                "sig_asset": "QQQ",
                "def_scenarios_run": ["QQQ", gold_note, cash_def_note],
                "nor_asset": "QLD",
                "agg_asset": "TQQQ",
                "common_trading_days": int(len(ix_common)),
                "first_date": str(ix_common[0].date()),
                "last_date": str(ix_common[-1].date()),
                "--root_gate": str(root.date()),
                "trail_stop": float(args.trail),
                "initial_capital": float(CAPITAL_START),
                "ma200_cache": str(csv_cache.resolve()),
                "gold_price_source_note": gold_note,
                "tier_safe_leg": f"{gold_note} (규칙4·SAFE 레그; def 시나리오와 별개)",
                "def_flat_cash_model_note": cash_def_note,
                "rmabs_comparison_note": rmabs_note,
            },
            "def_equals_QQQ": {
                "strategy_metrics": mq,
                "benchmarks_buy_hold_same_dates": bq,
            },
            "def_equals_gold_futures_et_al": {
                "strategy_metrics": mg,
                "benchmarks_buy_hold_same_dates": bq,
            },
            "def_equals_flat_cash_no_yield": {
                "strategy_metrics": mc,
                "benchmarks_buy_hold_same_dates": bq,
            },
            "rmabs_qld": {
                "strategy_metrics": mr_qld,
                "event_rows": int(len(ev_rm_qld)),
            },
            "rmabs_qqq_variant": {
                "strategy_metrics": mr_qqq,
                "event_rows": int(len(ev_rm_qqq)),
            },
        }
    )

    paths = {
        "combined_json": out_root / f"{stem}_combined.json",
        "strategy_compare_json": out_root / f"{stem}_strategy_only.json",
        "summary_md": out_root / f"{stem}_READ_ME.md",
        "nav_csv_paired": out_root / f"{stem}_nav_all_series.csv",
    }

    strat_only = {
        "meta": blob["meta"],
        "strategy_def_qqq_metrics": mq,
        "strategy_def_gold_metrics": mg,
        "strategy_def_cash_flat_metrics": mc,
        "rmabs_qld_metrics": mr_qld,
        "rmabs_qqq_metrics": mr_qqq,
        "rmabs_event_counts": {"RMABS_QLD": int(len(ev_rm_qld)), "RMABS_QQQ": int(len(ev_rm_qqq))},
        "delta_defGold_minus_defQQQ": metric_diff(mg, mq),
        "delta_defCash_minus_defQQQ": metric_diff(mc, mq),
        "delta_defCash_minus_defGold": metric_diff(mc, mg),
    }

    for p, obj in (
        (paths["combined_json"], blob),
        (paths["strategy_compare_json"], strat_only),
    ):
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)

    paired = pd.DataFrame(
        {
            "Date": ix_common,
            "NAV_FSM_defQQQ": nav_q.values,
            "NAV_FSM_defGoldPX": nav_g.values,
            "NAV_FSM_defCashFlat": nav_c.values,
            "NAV_RMABS_QLD": nav_rm_qld.values,
            "NAV_RMABS_QQQ": nav_rm_qqq.values,
        }
    )
    paired.to_csv(paths["nav_csv_paired"], index=False)

    def _fmt_pct(x: float) -> str:
        return f"{x * 100:.2f}%"

    def row(name: str, m: dict[str, float]) -> str:
        return (
            f"| {name} | {_fmt_pct(m['cagr'])} | {_fmt_pct(m['mdd'])} | {m['sharpe']:.3f} | "
            f"{m['sortino']:.3f} | {m['ulcer']:.4f} | {_fmt_pct(m['total_return'])} |"
        )

    benchmark_table_lines: list[str] = []
    bench_missing_keys: list[str] = []
    for key_bm, disp in BENCHMARK_DISPLAY_ORDER:
        if key_bm in bq:
            benchmark_table_lines.append(row(disp, bq[key_bm]))
        else:
            bench_missing_keys.append(key_bm)
            benchmark_table_lines.append(
                "| "
                + disp
                + " | *(산출 불가)* | — | — | — | — | — |"
            )

    bench_section = (
        "**벤치마크 정의(전략과 동일 달력·동일 평가지표):** "
        "구간 첫날 종가 기준 전액 매수 후 보유(QQQ · QLD · TQQQ). "
        "혼합은 자본을 반으로 나눠 QQQ·TQQQ 각각 매수 후 중간 리밸런스 없음.\n\n"
        "| 이름 | CAGR | MDD | Sharpe | Sortino | Ulcer | 총수익률 |\n"
        "|------|-----:|----:|-------:|--------:|------:|-----------:|\n"
        + "\n".join(benchmark_table_lines)
    )

    md = f"""# FSM + RMABS + 벤치 (동일 교집합)

고정: **sig=QQQ**, **nor=QLD**, **agg=TQQQ**, trail={args.trail}, 초기자본={CAPITAL_START:,.0f}.

- **def=현금 모델:** {cash_def_note}
- 공통 평가 구간: `{ix_common[0].date()}` ~ `{ix_common[-1].date()}` 거래일 **{len(ix_common)}일** (`--root`={root.date()} 이후 ∩ QQQ·QLD·TQQQ·금 종가 존재일).
- 금 가격 소스: {gold_note}
- MA200 캐시: `{csv_cache}`
- **RMABS 요약:** {rmabs_note}

## FSM 전략만 (def 세 가지)

| 이름 | CAGR | MDD | Sharpe | Sortino | Ulcer | 총수익률 |
|------|-----:|----:|-------:|--------:|------:|-----------:|
{row('FSM · def_asset=QQQ', mq)}
{row('FSM · def_asset=금(위 소스)', mg)}
{row('FSM · def_asset=현금(상수종가)', mc)}

## RMABS (동일 달력·전구간 1회)

| 이름 | CAGR | MDD | Sharpe | Sortino | Ulcer | 총수익률 |
|------|-----:|----:|-------:|--------:|------:|-----------:|
{row('RMABS-QLD (`strategy_rsi_ma_based_switching`)', mr_qld)}
{row('RMABS-QQQ (`strategy_rsi_ma_based_switching_qqq_only`)', mr_qqq)}

체결 이벤트 행 수(전구간 로그): RMABS-QLD **{len(ev_rm_qld)}**, RMABS-QQQ **{len(ev_rm_qqq)}**.

## 벤치마크 (항상 동일 4종·동일 일자)

{bench_section}
"""

    if bench_missing_keys:
        md += "\n> 주의: 일부 벤치 행을 계산하지 못했습니다: `" + "`, `".join(bench_missing_keys) + "`\n"

    md += """
### 편차 (전략만, 같은 지표 차원)

"""

    def section_delta(title: str, hi: dict[str, float], lo: dict[str, float]) -> str:
        chunks = [f"#### {title}\n\n"]
        for k in ("cagr", "mdd", "sharpe", "sortino", "ulcer", "total_return"):
            if k not in hi or k not in lo:
                continue
            d = hi[k] - lo[k]
            if k in ("cagr", "mdd"):
                chunks.append(f"- **Δ{k}**: {d*100:+.4f} pct pt (비율 차이 ×100)\n")
            elif k == "total_return":
                chunks.append(
                    f"- **Δ{k}**: {d:+.6f} (총수익 배수‑1 차이; 표 위 총수익열은 ×100 표기)\n"
                )
            else:
                chunks.append(f"- **Δ{k}**: {d:+.6f}\n")
        return "".join(chunks) + "\n"


    md += section_delta("(금 PX) − (QQQ `def`)", mg, mq)
    md += section_delta("(현금 상수종가) − (QQQ `def`)", mc, mq)
    md += section_delta("(현금 상수종가) − (금 PX `def`)", mc, mg)

    md += f"""
## 산출 파일

| 파일 | 설명 |
|------|------|
| `{paths["summary_md"].name}` | FSM 3종 + **RMABS 2종** + 벤치 4종 + 편차 |
| `{paths["combined_json"].name}` | 전부(JSON) |
| `{paths["strategy_compare_json"].name}` | FSM 세 `def` + RMABS 2종 지표 + 편차 |
| `{paths["nav_csv_paired"].name}` | 일자별 NAV (FSM 3 + RMABS 2 ) |
"""
    md_path = paths["summary_md"]
    md_path.write_text(md.strip() + "\n", encoding="utf-8")

    print("=== FSM def 3종 + RMABS-QLD/QQQ + 벤치 4종 (동일 교집합) 완료 ===")
    print(f"평가일수 {len(ix_common)}  금 소스: {gold_note}")
    for label, fp in paths.items():
        print(f"{label}: {fp.resolve()}")


if __name__ == "__main__":
    main()
