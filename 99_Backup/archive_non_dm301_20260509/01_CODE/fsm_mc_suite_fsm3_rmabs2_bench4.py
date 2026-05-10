"""
무작위 윈도 Monte Carlo 한 번에 묶음:

  · **MABS-QQQ / MABS-Gold / MABS-Cash**: 같은 QQQ 시그널·QLD·TQQQ, **교차형 규칙4** 후 **방어종가 100%**(레거시 단일종목 분기).
  · **RMABS-4tier-Gold**: 시그널·방어=QQQ, 일반=QLD, 공격=TQQQ, **안전=금 선물**(GC=F 우선).
  · **RMABS-4tier-Cash**: 동일 FSM인데 **안전 레그 종가 1.0 고정**(무이자 현금·명목 고정액).
  · **DMABS-4tier-Gold**: 레그는 RMABS-4tier-Gold와 동일(안전=금 선물), 스트레스만 **MA5>MA120** 상향 교차 시 안전→방어 100%.
  · **DMABS-4tier-Cash**: 레그는 RMABS-4tier-Cash와 동일(안전=종가 1.0), 스트레스만 **MA5>MA120** 상향 교차 시 안전→방어 100%.
  · **RMABS-4tier-PSQ / DMABS-4tier-PSQ**: 레그 구조는 Gold 티어와 같고 안전만 **PSQ**; DMABS만 MA5>M120 스트레스.

  · **RMABS-QLD / RMABS-QQQ**, 벤치 4종(B&H).

동일 교집합 일자에서 무작위 (start,end) 반복 후, **각 시행별 5지표 → 시행 간 평균** 표 1개로 MD·JSON·CSV 저장.

``--triple-mabs-rmabs-dmagold``: **MABS-QQQ · RMABS-4tier-Gold · DMABS-4tier-Gold** 만 MC·저장.  
``--root`` 이후 QQQ·QLD·TQQQ·금 **교집합** 전 구간을 사용(금은 ``load_gold_series`` 의 GC=F+^XAU 접두 포함 시 1999-03-10경부터 가능).

``--compare-three-only``: **MABS-QQQ · RMABS-4tier-Gold · RMABS-4tier-Cash** 만 계산·출력.

``--gold-mabs-dmabs``: **MABS-QQQ · RMABS-4tier-Gold · DMABS-4tier-Gold · DMABS-4tier-Cash** 만 계산·출력.  
DMABS 계열은 RMABS 동일 레그이고 스트레스만 ``stress_bleed_mode="MA5_CROSS_MA120_FULL"``
(MA5>MA120 상향 교차 시 안전→방어 100%); Gold는 안전=금, Cash는 안전=종가 1.0.

``--five-gold-dmabs-psq``: **MABS-QQQ · RMABS-Gold · DMABS-Gold · RMABS-PSQ · DMABS-PSQ** 만 출력; 공통 일자열은 PSQ 장중 시작 이후 교집합.

실행 예::

    python3 01_CODE/fsm_mc_suite_fsm3_rmabs2_bench4.py \\
        --root 2002-10-01 --mc-years 3 --mc-iters 3000 --mc-seed 42

    python3 01_CODE/fsm_mc_suite_fsm3_rmabs2_bench4.py --compare-three-only \\
        --root 2002-10-01 --mc-years 3 --mc-iters 3000 --mc-seed 42

    python3 01_CODE/fsm_mc_suite_fsm3_rmabs2_bench4.py --gold-mabs-dmabs \\
        --root 2002-10-01 --mc-years 3 --mc-iters 3000 --mc-seed 42

    python3 01_CODE/fsm_mc_suite_fsm3_rmabs2_bench4.py --five-gold-dmabs-psq \\
        --root 2002-10-01 --mc-years 3 --mc-iters 3000 --mc-seed 42

    python3 01_CODE/fsm_mc_suite_fsm3_rmabs2_bench4.py --triple-mabs-rmabs-dmagold \\
        --root 1999-03-10 --mc-years 3 --mc-iters 3000 --mc-seed 42

    DMABS 방어까지 QLD 로 통일한 변형까지 같이 저장::

    python3 01_CODE/fsm_mc_suite_fsm3_rmabs2_bench4.py --triple-mabs-rmabs-dmagold \\
        --include-dmabs-gold-defnor-qld --root 1999-03-10 \\
        --mc-years 3 --mc-iters 3000 --mc-seed 42

    DMABS-TQQQ-QLD-Gold / Cash / QQQ (안전 레그만 다름; def·nor=QLD, 시그 QQ, 공 TQQQ) 세 전만 MC::

    python3 01_CODE/fsm_mc_suite_fsm3_rmabs2_bench4.py \\
        --mc-dmabs-tqq-qld-triple-safe --root 1999-03-10 \\
        --mc-years 3 --mc-iters 3000 --mc-seed 42
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
    strategy_rsi_ma_based_switching,
    strategy_rsi_ma_based_switching_qqq_only,
)
from rmabs_gold_simulation import load_gold_series  # noqa: E402

from fsm_four_asset_strategy import (  # noqa: E402
    CAPITAL_START,
    DEFAULT_TRAIL,
    MA_WINDOW,
    _core_metrics,
    _distribution_block,
    _flatten_metrics,
    benchmark_panel_for_slice,
    build_bench_source_for_index,
    get_or_build_ma200,
    load_daily_extended_or_yahoo,
    mc_min_trading_days,
    run_fsm_backtest,
    _align_five,
    _align_four,
)

OUT_DIR = _ROOT / "03_RESULT" / "sensitivity"

METS = ("cagr", "mdd", "sharpe", "sortino", "ulcer")

SUMMARY_ROWS_FULL: tuple[tuple[str, str], ...] = (
    ("MABS-QQQ", "mabs_qqq"),
    ("MABS-Gold", "mabs_gold"),
    ("MABS-Cash", "mabs_cash"),
    ("RMABS-QLD", "rmabs_qld"),
    ("RMABS-QQQ", "rmabs_qqq"),
    ("RMABS-4tier-Gold", "rmabs_4tier_gold"),
    ("RMABS-4tier-Cash", "rmabs_4tier_cash"),
    ("DMABS-4tier-Gold", "dmabs_4tier_gold"),
    ("DMABS-4tier-Cash", "dmabs_4tier_cash"),
    ("Bench · QQQ 100% B&H", "bh_qqq"),
    ("Bench · QLD 100% B&H", "bh_qld"),
    ("Bench · QQQ50/TQQQ50", "bh_mix50"),
    ("Bench · TQQQ 100% B&H", "bh_tqqq"),
)

SUMMARY_ROWS_THREE: tuple[tuple[str, str], ...] = (
    ("MABS-QQQ", "mabs_qqq"),
    ("RMABS-4tier-Gold", "rmabs_4tier_gold"),
    ("RMABS-4tier-Cash", "rmabs_4tier_cash"),
)

SUMMARY_ROWS_GOLD_MABS_DMABS: tuple[tuple[str, str], ...] = (
    ("MABS-QQQ", "mabs_qqq"),
    ("RMABS-4tier-Gold", "rmabs_4tier_gold"),
    ("DMABS-4tier-Gold", "dmabs_4tier_gold"),
    ("DMABS-4tier-Cash", "dmabs_4tier_cash"),
)

SUMMARY_ROWS_FIVE_PSQ_GOLD_DMABS: tuple[tuple[str, str], ...] = (
    ("MABS-QQQ", "mabs_qqq"),
    ("RMABS-4tier-Gold", "rmabs_4tier_gold"),
    ("DMABS-4tier-Gold", "dmabs_4tier_gold"),
    ("RMABS-4tier-PSQ", "rmabs_4tier_psq"),
    ("DMABS-4tier-PSQ", "dmabs_4tier_psq"),
)

SUMMARY_ROWS_TRIPLE_MABS_RABS_DMA_GOLD: tuple[tuple[str, str], ...] = (
    ("MABS-QQQ", "mabs_qqq"),
    ("RMABS-4tier-Gold", "rmabs_4tier_gold"),
    ("DMABS-4tier-Gold", "dmabs_4tier_gold"),
)

# DMABS만 반등·방어 종가를 QLD로 통일(시그 QQ, 안전=금, 공격=TQQQ); `--triple-mabs-rmabs-dmagold --include-dmabs-gold-defnor-qld` 에서 추가 행.
ROW_DMABS_GOLD_DNQ: tuple[str, str] = (
    "DMABS-4tier-Gold (반등·방어=QLD)",
    "dmabs_4tier_gold_dnq",
)


# `--mc-dmabs-tqq-qld-triple-safe` 전용 라벨·컬럼 접두사
SUMMARY_ROWS_DMABS_TQQ_QLD_TRIPLE_SAFE: tuple[tuple[str, str], ...] = (
    ("DMABS-TQQQ-QLD-Gold", "dmabs_tqqq_qld_safe_gold"),
    ("DMABS-TQQQ-QLD-Cash", "dmabs_tqqq_qld_safe_cash"),
    ("DMABS-TQQQ-QLD-QQQ", "dmabs_tqqq_qld_safe_qq"),
)


def _mc_dmabs_tqq_qld_triple_safe_only(
    *,
    ix: pd.DatetimeIndex,
    mav: pd.Series,
    sg_i: pd.DataFrame,
    ng_i: pd.DataFrame,
    ag_i: pd.DataFrame,
    g_i: pd.DataFrame,
    cash_i: pd.DataFrame,
    args: argparse.Namespace,
    out_root: Path,
    gold_note: str,
    csv_cache: Path,
    root: pd.Timestamp,
) -> None:
    """Close_sig=QQQ, Close_bounce=Close_defense=QLD, Close_agg=TQQQ; 안전만 금/현금/QQQ. DMABS 스트레스 동일."""
    lo = mc_min_trading_days(args.mc_years)
    n_dates = len(ix)
    rng = np.random.default_rng(args.mc_seed)
    aligned_g = _align_five(sg_i, g_i, ng_i, ng_i, ag_i)
    aligned_c = _align_five(sg_i, cash_i, ng_i, ng_i, ag_i)
    aligned_q = _align_five(sg_i, sg_i, ng_i, ng_i, ag_i)
    n_gold, _ = run_fsm_backtest(
        aligned_g,
        mav,
        trail_stop=args.trail,
        initial_capital=CAPITAL_START,
        use_safe_ma_rule=True,
        stress_bleed_mode="MA5_CROSS_MA120_FULL",
    )
    n_cash, _ = run_fsm_backtest(
        aligned_c,
        mav,
        trail_stop=args.trail,
        initial_capital=CAPITAL_START,
        use_safe_ma_rule=True,
        stress_bleed_mode="MA5_CROSS_MA120_FULL",
    )
    n_qq, _ = run_fsm_backtest(
        aligned_q,
        mav,
        trail_stop=args.trail,
        initial_capital=CAPITAL_START,
        use_safe_ma_rule=True,
        stress_bleed_mode="MA5_CROSS_MA120_FULL",
    )

    summary_rows = SUMMARY_ROWS_DMABS_TQQ_QLD_TRIPLE_SAFE
    rows: list[dict[str, object]] = []
    for _ in range(args.mc_iters):
        s_pos = int(rng.integers(0, n_dates - lo + 1))
        l_max = n_dates - s_pos
        l_win = int(rng.integers(lo, l_max + 1))
        win_ix = ix[s_pos : s_pos + l_win]
        sub_g = n_gold.iloc[s_pos : s_pos + l_win]
        sub_c = n_cash.iloc[s_pos : s_pos + l_win]
        sub_q = n_qq.iloc[s_pos : s_pos + l_win]
        row: dict[str, object] = {
            "window_start": str(win_ix[0].date()),
            "window_end": str(win_ix[-1].date()),
            "trading_days": int(l_win),
            "calendar_years_approx": round(
                (pd.Timestamp(win_ix[-1]) - pd.Timestamp(win_ix[0])).days / 365.25, 4
            ),
        }
        row.update(_flatten_metrics("dmabs_tqqq_qld_safe_gold", _core_metrics(sub_g)))
        row.update(_flatten_metrics("dmabs_tqqq_qld_safe_cash", _core_metrics(sub_c)))
        row.update(_flatten_metrics("dmabs_tqqq_qld_safe_qq", _core_metrics(sub_q)))
        rows.append(row)

    df = pd.DataFrame(rows)
    fx0 = pd.Timestamp(ix[0]).strftime("%Y%m%d")
    fx1 = pd.Timestamp(ix[-1]).strftime("%Y%m%d")
    stem = (
        f"mc_triple_dmabsTqqqQld_safeGoldCashQQ_{fx0}_{fx1}_{args.mc_years:g}yr_n{args.mc_iters}_s{args.mc_seed}"
    )
    csv_path = out_root / f"{stem}_windows.csv"
    md_path = out_root / f"{stem}_READ_ME.md"
    json_path = out_root / f"{stem}_summary.json"
    df.to_csv(csv_path, index=False)

    table_lines = [
        "| 전략 | CAGR 평균 | CAGR 중앙 | MDD 평균 | MDD 중앙 | Sharpe 평균 | Sharpe 중앙 | Sortino 평균 | Sortino 중앙 | Ulcer 평균 | Ulcer 중앙 |",
        "|------|----------:|----------:|----------:|----------:|------------:|-----------:|-------------:|------------:|----------:|-----------:|",
    ]
    distro: dict[str, dict[str, dict[str, float]]] = {}
    for lab, pref in summary_rows:
        row_cells = [lab]
        block: dict[str, dict[str, float]] = {}
        for mk in METS:
            col = f"{pref}_{mk}"
            if col not in df.columns:
                row_cells.extend(["—", "—"])
                continue
            s = df[col].astype(float)
            mn = float(s.mean())
            block[mk] = _distribution_block(s)
            med = float(block[mk]["median"])
            if mk in ("cagr", "mdd"):
                row_cells.append(f"{mn * 100:.2f}%")
                row_cells.append(f"{med * 100:.2f}%")
            elif mk == "ulcer":
                row_cells.append(f"{mn:.4f}")
                row_cells.append(f"{med:.4f}")
            else:
                row_cells.append(f"{mn:.3f}")
                row_cells.append(f"{med:.3f}")
        table_lines.append("| " + " | ".join(row_cells) + " |")
        distro[pref] = block

    desc = (
        "MC 무작위 창. **DMABS-TQQQ-QLD-Gold / Cash / QQQ** 만. "
        "공통 레그: 시그 **QQQ** · 방어·노말 종가 **QLD** · 공격 **TQQQ** · "
        "`stress_bleed_mode=MA5_CROSS_MA120_FULL`. "
        "안전만 **금·명목 현금 종가 1·QQQ**로 구분."
    )
    meta = {
        "description": desc,
        "mc_dmabs_tqq_qld_triple_safe": True,
        "--root_requested": str(root.date()),
        "common_index": {"first": str(ix[0].date()), "last": str(ix[-1].date()), "days": len(ix)},
        "gold_series": gold_note,
        "legs": (
            "Close_sig=QQQ · Close_safe∈{Gold,Cash≡1.0,QQQ} · Close_bounce=Close_defense=QLD · Close_agg=TQQQ · "
            "DMABS MA5>M120 full bleed"
        ),
        "tier_labels_row_order": [lab for lab, _ in summary_rows],
        "ma200_csv": str(csv_cache.resolve()),
        "--root": str(root.date()),
        "trail": float(args.trail),
        "mc_years": float(args.mc_years),
        "min_trading_days": int(lo),
        "mc_iters": int(args.mc_iters),
        "mc_seed": int(args.mc_seed),
        "capital": float(CAPITAL_START),
        "csv_windows": str(csv_path.resolve()),
    }
    json_blob = {"meta": meta, "distribution_by_series_metric": distro}
    title = "# Monte Carlo: DMABS-TQQQ-QLD-Gold · Cash · QQQ\n\n"
    bullets = (
        f"- 교집합: `{ix[0].date()}` ~ `{ix[-1].date()}` (**{len(ix)}일**), 금 레그 포함 시: {gold_note}\n"
    )
    md_text = (
        title
        + f"설정: 최소 거래연 **{args.mc_years:g}년**(거래일 ≥ {lo}일), 시행 **{args.mc_iters}회**, seed={args.mc_seed}, trail={args.trail}.\n\n"
        + bullets
        + "\n"
        + f"## 무작위 창 지표 ({args.mc_iters}회): 평균·중앙\n\n"
        + "\n".join(table_lines)
        + "\n\n## 참고 분포 통계(JSON)\n\n"
        f"파일 `{json_path.name}`\n\n## 원시\n\n`{csv_path.name}`\n"
    )
    md_path.write_text(md_text.strip() + "\n", encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(json_blob, fh, indent=2, ensure_ascii=False)
    print("=== MC (DMABS-TQQQ-QLD-Gold / Cash / QQQ) 완료 ===")
    print(str(csv_path.resolve()))
    print(str(md_path.resolve()))
    print(str(json_path.resolve()))
    print("\n" + "\n".join(table_lines))


def _flat_cash(ix: pd.DatetimeIndex) -> pd.DataFrame:
    ix = ix.sort_values()
    one = pd.Series(1.0, index=ix, dtype=float)
    return pd.DataFrame({"Open": one, "High": one, "Low": one, "Close": one})


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MC: MABS×3+RMABS-4tier-Gold/Cash+DMABS-Gold/Cash+RMABS×2+벤치×4 "
        "(또는 --compare-three-only · --gold-mabs-dmabs · --five-gold-dmabs-psq · "
        "--triple-mabs-rmabs-dmagold)"
    )
    ap.add_argument("--root", default="1999-03-10")
    ap.add_argument("--trail", type=float, default=DEFAULT_TRAIL)
    ap.add_argument("--mc-years", type=float, default=3.0)
    ap.add_argument("--mc-iters", type=int, default=3000)
    ap.add_argument("--mc-seed", type=int, default=42)
    ap.add_argument("--out-dir", default="", help="비우면 03_RESULT/sensitivity")
    ap.add_argument(
        "--compare-three-only",
        action="store_true",
        help="MABS-QQQ · RMABS-4tier-Gold · RMABS-4tier-Cash 만 MC·저장",
    )
    ap.add_argument(
        "--gold-mabs-dmabs",
        action="store_true",
        help="MABS-QQQ · RMABS-Gold · DMABS-Gold · DMABS-Cash(MA5>MA120 스트레스)만 MC·저장",
    )
    ap.add_argument(
        "--five-gold-dmabs-psq",
        action="store_true",
        help="MABS·R/DMABS-Gold·R/DMABS-PSQ 5종 MC; 일자열=PSQ·금 교집합(상장 이후만)",
    )
    ap.add_argument(
        "--triple-mabs-rmabs-dmagold",
        action="store_true",
        help="MABS-QQQ·RMABS-Gold·DMABS-Gold 3종 MC; QQ·QLD·TQQ·금 교집합 (--root 이후)",
    )
    ap.add_argument(
        "--include-dmabs-gold-defnor-qld",
        action="store_true",
        help=(
            "`--triple-mabs-rmabs-dmagold` 와 함께만: DMABS 한 종목 추가 계산 "
            "(Close_bounce=Close_defense=QLD·시그=QQQ·안전=금·공격=TQQQ·스트레스 MA5>M120 동일)."
        ),
    )
    ap.add_argument(
        "--mc-dmabs-tqq-qld-triple-safe",
        action="store_true",
        help=(
            "DMABS-TQQQ-QLD-Gold / Cash / QQQ 세 전략만 MC "
            "(Close_sig=QQQ·def=nor=QLD·agg=TQQQ, 안전만 금 vs 현금1 vs QQ)."
        ),
    )
    args = ap.parse_args()
    exclusive = sum(
        [
            bool(args.compare_three_only),
            bool(args.gold_mabs_dmabs),
            bool(args.five_gold_dmabs_psq),
            bool(args.triple_mabs_rmabs_dmagold),
            bool(args.mc_dmabs_tqq_qld_triple_safe),
        ]
    )
    if exclusive > 1:
        raise SystemExit(
            "다음 플래그 중 하나만: "
            "--compare-three-only · --gold-mabs-dmabs · --five-gold-dmabs-psq · "
            "--triple-mabs-rmabs-dmagold · --mc-dmabs-tqq-qld-triple-safe"
        )
    triple_abs_gold_chk = bool(args.triple_mabs_rmabs_dmagold)
    if bool(args.include_dmabs_gold_defnor_qld) and not triple_abs_gold_chk:
        raise SystemExit("--include-dmabs-gold-defnor-qld 는 --triple-mabs-rmabs-dmagold 와 함께만 사용 가능")

    out_root = Path(args.out_dir) if args.out_dir.strip() else OUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    root = pd.Timestamp(args.root.strip())
    sg = load_extended_daily("QQQ")
    ng = load_extended_daily("QLD")
    ag = load_extended_daily("TQQQ")

    ma_full, csv_cache = get_or_build_ma200(sg, "QQQ", force_rebuild=False)

    triple_abs_gold = bool(args.triple_mabs_rmabs_dmagold)

    ix0_all = sg.index.intersection(ng.index).intersection(ag.index).sort_values()
    ix0 = ix0_all[ix0_all >= root]
    gold_df, gold_note = load_gold_series(ix0[0], ix0[-1])
    ix = ix0.intersection(gold_df.index).sort_values()
    psq_meta = ""

    five_psq = bool(args.five_gold_dmabs_psq)
    trio_dma = bool(args.gold_mabs_dmabs)
    three_only = bool(args.compare_three_only)
    psq_df: pd.DataFrame | None = None
    if five_psq:
        psq_df = load_daily_extended_or_yahoo("PSQ")
        psqc = psq_df["Close"].astype(float)
        psq_dates = psqc.index[(psqc > 0) & psqc.notna()]
        ix = ix.intersection(psq_dates).sort_values()
        ts_min = pd.Timestamp(psq_dates.min())
        psq_meta = (
            "PSQ(안전)=Yahoo 수정주가 02_DATA/yahoo/PSQ/PSQ_daily.csv 등; 금·QQQ 교집합과 PSQ 종가 존재일만 사용. "
            f"PSQ 시계열에서 양호 첫 거래일: {ts_min.date()}."
        )

    if len(ix) < MA_WINDOW:
        raise SystemExit(f"교집합 거래일 {len(ix)} < MA200 {MA_WINDOW}")

    lo = mc_min_trading_days(args.mc_years)
    if len(ix) < lo:
        raise SystemExit(f"전체 일수 {len(ix)} < 최소 윈도 {lo}")

    sg_i, ng_i, ag_i = sg.reindex(ix), ng.reindex(ix), ag.reindex(ix)
    g_i = gold_df.reindex(ix)
    cash_i = _flat_cash(ix)
    mav_early = ma_full.reindex(ix)
    if args.mc_dmabs_tqq_qld_triple_safe:
        _mc_dmabs_tqq_qld_triple_safe_only(
            ix=ix,
            mav=mav_early,
            sg_i=sg_i,
            ng_i=ng_i,
            ag_i=ag_i,
            g_i=g_i,
            cash_i=cash_i,
            args=args,
            out_root=out_root,
            gold_note=gold_note,
            csv_cache=csv_cache,
            root=root,
        )
        return

    aligned_4tier_psq: pd.DataFrame | None = None
    aligned_4tier_gold = _align_five(sg_i, g_i, sg_i, ng_i, ag_i)
    aligned_4tier_cash = _align_five(sg_i, cash_i, sg_i, ng_i, ag_i)
    if five_psq:
        if psq_df is None:
            raise SystemExit("internal: PSQ 데이터 없음 (--five-gold-dmabs-psq 분기)")
        psq_i = psq_df.reindex(ix)
        aligned_4tier_psq = _align_five(sg_i, psq_i, sg_i, ng_i, ag_i)

    aligned_mabs_qq = _align_four(sg_i, sg_i, ng_i, ag_i)

    mav = ma_full.reindex(ix)

    if three_only:
        summary_rows = SUMMARY_ROWS_THREE
    elif trio_dma:
        summary_rows = SUMMARY_ROWS_GOLD_MABS_DMABS
    elif five_psq:
        summary_rows = SUMMARY_ROWS_FIVE_PSQ_GOLD_DMABS
    elif triple_abs_gold:
        summary_rows = (
            SUMMARY_ROWS_TRIPLE_MABS_RABS_DMA_GOLD + (ROW_DMABS_GOLD_DNQ,)
            if bool(args.include_dmabs_gold_defnor_qld)
            else SUMMARY_ROWS_TRIPLE_MABS_RABS_DMA_GOLD
        )
    else:
        summary_rows = SUMMARY_ROWS_FULL

    nav_rm_ld = nav_rm_qq = None
    nav_mabs_go = nav_mabs_cx = None
    bench_src = None
    if not three_only and not trio_dma and not five_psq and not triple_abs_gold:
        bench_src = build_bench_source_for_index(ix)
        nav_rm_ld, _ = strategy_rsi_ma_based_switching(
            sg_i, ng_i, ag_i, CAPITAL_START, series_name="RMABS-QLD"
        )
        nav_rm_qq, _ = strategy_rsi_ma_based_switching_qqq_only(
            sg_i, ng_i, ag_i, CAPITAL_START, series_name="RMABS-QQQ"
        )

    if five_psq:
        print(
            "전구간 MABS-QQQ + RMABS/DMABS-4tier-Gold + RMABS/DMABS-4tier-PSQ NAV 계산 중… "
            "(일자열: PSQ·금 교집합 이후만)"
        )
    elif triple_abs_gold:
        print(
            "전구간 MABS-QQQ + RMABS-4tier-Gold + DMABS-4tier-Gold NAV 계산 중… "
            f"(--root {root.date()}, QQ·QLD·TQQ·금 교집합)"
        )
    elif trio_dma:
        print(
            "전구간 MABS-QQQ + RMABS-4tier-Gold + DMABS-4tier-Gold + DMABS-4tier-Cash NAV 계산 중…"
        )
    elif three_only:
        print("전구간 MABS-QQQ + RMABS-4tier-Gold/Cash NAV 계산 중…")
    else:
        print("전구간 MABS×3 + RMABS-4tier-Gold/Cash + DMABS + RMABS×2 NAV 계산 중…")
    nav_mabs_qq, _ = run_fsm_backtest(
        aligned_mabs_qq, mav, trail_stop=args.trail, initial_capital=CAPITAL_START, use_safe_ma_rule=False
    )
    if not three_only and not trio_dma and not five_psq and not triple_abs_gold:
        aligned_mabs_gold = _align_four(sg_i, g_i, ng_i, ag_i)
        aligned_mabs_cash = _align_four(sg_i, cash_i, ng_i, ag_i)
        nav_mabs_go, _ = run_fsm_backtest(
            aligned_mabs_gold, mav, trail_stop=args.trail, initial_capital=CAPITAL_START, use_safe_ma_rule=False
        )
        nav_mabs_cx, _ = run_fsm_backtest(
            aligned_mabs_cash, mav, trail_stop=args.trail, initial_capital=CAPITAL_START, use_safe_ma_rule=False
        )
    nav_rmabs_gold, _ = run_fsm_backtest(
        aligned_4tier_gold, mav, trail_stop=args.trail, initial_capital=CAPITAL_START, use_safe_ma_rule=True
    )
    nav_rmabs_psq = nav_dmabs_psq = nav_rmabs_cash = nav_dmabs_cash = None
    if not five_psq and not triple_abs_gold:
        nav_rmabs_cash, _ = run_fsm_backtest(
            aligned_4tier_cash,
            mav,
            trail_stop=args.trail,
            initial_capital=CAPITAL_START,
            use_safe_ma_rule=True,
        )
        nav_dmabs_cash, _ = run_fsm_backtest(
            aligned_4tier_cash,
            mav,
            trail_stop=args.trail,
            initial_capital=CAPITAL_START,
            use_safe_ma_rule=True,
            stress_bleed_mode="MA5_CROSS_MA120_FULL",
        )
    if five_psq and aligned_4tier_psq is not None:
        nav_rmabs_psq, _ = run_fsm_backtest(
            aligned_4tier_psq,
            mav,
            trail_stop=args.trail,
            initial_capital=CAPITAL_START,
            use_safe_ma_rule=True,
        )
        nav_dmabs_psq, _ = run_fsm_backtest(
            aligned_4tier_psq,
            mav,
            trail_stop=args.trail,
            initial_capital=CAPITAL_START,
            use_safe_ma_rule=True,
            stress_bleed_mode="MA5_CROSS_MA120_FULL",
        )

    nav_dmabs_gold, _ = run_fsm_backtest(
        aligned_4tier_gold,
        mav,
        trail_stop=args.trail,
        initial_capital=CAPITAL_START,
        use_safe_ma_rule=True,
        stress_bleed_mode="MA5_CROSS_MA120_FULL",
    )

    nav_dmabs_gold_dnq = None
    if triple_abs_gold and bool(args.include_dmabs_gold_defnor_qld):
        aligned_dmabs_gold_dnq = _align_five(sg_i, g_i, ng_i, ng_i, ag_i)
        nav_dmabs_gold_dnq, _ = run_fsm_backtest(
            aligned_dmabs_gold_dnq,
            mav,
            trail_stop=args.trail,
            initial_capital=CAPITAL_START,
            use_safe_ma_rule=True,
            stress_bleed_mode="MA5_CROSS_MA120_FULL",
        )

    n_dates = len(ix)
    rng = np.random.default_rng(args.mc_seed)
    rows: list[dict[str, object]] = []

    for _ in range(args.mc_iters):
        s_pos = int(rng.integers(0, n_dates - lo + 1))
        l_max = n_dates - s_pos
        l_win = int(rng.integers(lo, l_max + 1))
        win_ix = ix[s_pos : s_pos + l_win]

        sub_mqq = nav_mabs_qq.iloc[s_pos : s_pos + l_win]
        sub_rgold = nav_rmabs_gold.iloc[s_pos : s_pos + l_win]
        sub_dgold = nav_dmabs_gold.iloc[s_pos : s_pos + l_win]

        row: dict[str, object] = {
            "window_start": str(win_ix[0].date()),
            "window_end": str(win_ix[-1].date()),
            "trading_days": int(l_win),
            "calendar_years_approx": round(
                (pd.Timestamp(win_ix[-1]) - pd.Timestamp(win_ix[0])).days / 365.25, 4
            ),
        }
        row.update(_flatten_metrics("mabs_qqq", _core_metrics(sub_mqq)))
        row.update(_flatten_metrics("rmabs_4tier_gold", _core_metrics(sub_rgold)))
        if five_psq:
            assert nav_rmabs_psq is not None and nav_dmabs_psq is not None
            sub_rpsq = nav_rmabs_psq.iloc[s_pos : s_pos + l_win]
            sub_dpsq = nav_dmabs_psq.iloc[s_pos : s_pos + l_win]
            row.update(_flatten_metrics("dmabs_4tier_gold", _core_metrics(sub_dgold)))
            row.update(_flatten_metrics("rmabs_4tier_psq", _core_metrics(sub_rpsq)))
            row.update(_flatten_metrics("dmabs_4tier_psq", _core_metrics(sub_dpsq)))
        elif triple_abs_gold:
            row.update(_flatten_metrics("dmabs_4tier_gold", _core_metrics(sub_dgold)))
            if nav_dmabs_gold_dnq is not None:
                sub_dnq = nav_dmabs_gold_dnq.iloc[s_pos : s_pos + l_win]
                row.update(_flatten_metrics("dmabs_4tier_gold_dnq", _core_metrics(sub_dnq)))
        else:
            assert nav_rmabs_cash is not None and nav_dmabs_cash is not None
            sub_rcash = nav_rmabs_cash.iloc[s_pos : s_pos + l_win]
            sub_dcash = nav_dmabs_cash.iloc[s_pos : s_pos + l_win]
            row.update(_flatten_metrics("rmabs_4tier_cash", _core_metrics(sub_rcash)))
            row.update(_flatten_metrics("dmabs_4tier_cash", _core_metrics(sub_dcash)))
            row.update(_flatten_metrics("dmabs_4tier_gold", _core_metrics(sub_dgold)))

        if not three_only and not trio_dma and not five_psq and not triple_abs_gold:
            sub_mgr = nav_mabs_go.iloc[s_pos : s_pos + l_win]
            sub_mcx = nav_mabs_cx.iloc[s_pos : s_pos + l_win]
            row.update(_flatten_metrics("mabs_gold", _core_metrics(sub_mgr)))
            row.update(_flatten_metrics("mabs_cash", _core_metrics(sub_mcx)))
            rq = nav_rm_ld.iloc[s_pos : s_pos + l_win]
            rj = nav_rm_qq.iloc[s_pos : s_pos + l_win]
            row.update(_flatten_metrics("rmabs_qld", _core_metrics(rq)))
            row.update(_flatten_metrics("rmabs_qqq", _core_metrics(rj)))
            pans = benchmark_panel_for_slice(win_ix, bench_src, CAPITAL_START)
            for k_bm, pref in (
                ("QQQ_bh", "bh_qqq"),
                ("QLD_bh", "bh_qld"),
                ("mix_qq50_tqq50_bh", "bh_mix50"),
                ("TQQQ_bh", "bh_tqqq"),
            ):
                s = pans.get(k_bm)
                if s is not None and len(s) >= 2:
                    row.update(_flatten_metrics(pref, _core_metrics(s)))
        rows.append(row)

    df = pd.DataFrame(rows)
    fx0 = pd.Timestamp(ix[0]).strftime("%Y%m%d")
    fx1 = pd.Timestamp(ix[-1]).strftime("%Y%m%d")
    stem = (
        f"mc_three_mabsqqq_rmabs4tierGold_rmabs4tierCash_{fx0}_{fx1}_"
        f"{args.mc_years:g}yr_n{args.mc_iters}_s{args.mc_seed}"
        if three_only
        else (
            (
                f"mc_four_mabs_rmabs_dmabs_dmabsDnq_fullix_{fx0}_{fx1}_{args.mc_years:g}yr_n{args.mc_iters}_s{args.mc_seed}"
                if bool(args.include_dmabs_gold_defnor_qld)
                else f"mc_three_mabs_rmabs_dmabsGold_fullix_{fx0}_{fx1}_{args.mc_years:g}yr_n{args.mc_iters}_s{args.mc_seed}"
            )
            if triple_abs_gold
            else (
                f"mc_five_gold_dmabs_psq_{fx0}_{fx1}_{args.mc_years:g}yr_n{args.mc_iters}_s{args.mc_seed}"
                if five_psq
                else (
                    f"mc_four_mabs_rmabs_dmabsGold_dmabsCash_{fx0}_{fx1}_{args.mc_years:g}yr_n{args.mc_iters}_s{args.mc_seed}"
                    if trio_dma
                    else (
                        f"mc_suite_mabs3_rmabs4tierGold_rmabs4tierCash_dmabsGold_dmabsCash_rmabs2_bh4_navslice_safeGoldFut_{fx0}_{fx1}_"
                        f"{args.mc_years:g}yr_n{args.mc_iters}_s{args.mc_seed}"
                    )
                )
            )
        )
    )

    csv_path = out_root / f"{stem}_windows.csv"
    md_path = out_root / f"{stem}_READ_ME.md"
    json_path = out_root / f"{stem}_summary.json"
    df.to_csv(csv_path, index=False)

    if triple_abs_gold:
        table_lines = [
            "| 전략 | CAGR 평균 | CAGR 중앙 | MDD 평균 | MDD 중앙 | Sharpe 평균 | Sharpe 중앙 | Sortino 평균 | Sortino 중앙 | Ulcer 평균 | Ulcer 중앙 |",
            "|------|----------:|----------:|----------:|----------:|------------:|-----------:|-------------:|------------:|----------:|-----------:|",
        ]
    else:
        table_lines = [
            "| 구분 | CAGR 평균 | MDD 평균 | Sharpe 평균 | Sortino 평균 | Ulcer 평균 |",
            "|------|----------:|----------:|------------:|-------------:|-----------:|",
        ]
    distro: dict[str, dict[str, dict[str, float]]] = {}
    for lab, pref in summary_rows:
        row_cells = [lab]
        block: dict[str, dict[str, float]] = {}
        for mk in METS:
            col = f"{pref}_{mk}"
            if col not in df.columns:
                if triple_abs_gold:
                    row_cells.extend(["—", "—"])
                else:
                    row_cells.append("—")
                continue
            s = df[col].astype(float)
            mn = float(s.mean())
            block[mk] = _distribution_block(s)
            if triple_abs_gold:
                med = float(block[mk]["median"])
                if mk in ("cagr", "mdd"):
                    row_cells.append(f"{mn * 100:.2f}%")
                    row_cells.append(f"{med * 100:.2f}%")
                elif mk == "ulcer":
                    row_cells.append(f"{mn:.4f}")
                    row_cells.append(f"{med:.4f}")
                else:
                    row_cells.append(f"{mn:.3f}")
                    row_cells.append(f"{med:.3f}")
            else:
                if mk in ("cagr", "mdd"):
                    row_cells.append(f"{mn * 100:.2f}%")
                elif mk == "ulcer":
                    row_cells.append(f"{mn:.4f}")
                else:
                    row_cells.append(f"{mn:.3f}")
        table_lines.append("| " + " | ".join(row_cells) + " |")
        distro[pref] = block

    desc_triple_ix = (
        "MC 동일 무작위 창. **MABS-QQQ · RMABS-4tier-Gold · DMABS-4tier-Gold** 세 전략만. "
        "교집합 일자열: ``--root`` 이후의 QQQ·QLD·TQQQ 확장 거래일과 금 종가 존재일."
    )
    if triple_abs_gold and bool(args.include_dmabs_gold_defnor_qld):
        desc_triple_ix = (
            "MC 동일 무작위 창. **MABS-QQQ · RMABS-4tier-Gold · DMABS-4tier-Gold · "
            "DMABS-4tier-Gold (방어·노말=QLD)** 네 전략. "
            "앞 세 종은 종가 줄이 QQ(시그/def)·QLD(nor)·TQQQ; 네 번째만 DMABS에서 방어·노말 종가 모두 "
            "**QLD**에 통일된 변형이다. 교집합은 ``--root`` 이후 QQQ·QLD·TQQQ·금 존재일."
        )
    desc_three = (
        "MC 동일 무작위 창. MABS-QQQ·RMABS-4tier-Gold·RMABS-4tier-Cash 전구간 1회 NAV 슬라이스. "
        "RMABS-4tier-Gold 안전=금(GC=F 등); RMABS-4tier-Cash 안전=종가 1.0 고정 무이자 현금."
    )
    desc_trio_dmabs = (
        "MC 동일 무작위 창. MABS-QQQ·RMABS-4tier-Gold·DMABS-4tier-Gold·DMABS-4tier-Cash 전구간 1회 NAV 슬라이스. "
        "DMABS-4tier-Gold: RMABS-4tier-Gold와 동일 레그(안전=금), 스트레스만 MA5>M120; "
        "DMABS-4tier-Cash: RMABS-4tier-Cash와 동일 레그(안전=1.0), 스트레스만 MA5>M120."
    )
    desc_five_psq = (
        "MC 동일 무작위 창. MABS-QQQ·RMABS-4tier-Gold·DMABS-4tier-Gold·RMABS-4tier-PSQ·DMABS-4tier-PSQ 전구간 1회 NAV. "
        "공통 일자열은 QQQ·QLD·TQQQ·금·PSQ 모두 존재하는 날(=PSQ 상장 이후·유효 종가)로 제한. "
        "PSQ 티어는 안전 레그만 PSQ로 바꾼 RMABS/DMABS; DMABS-PSQ는 MA5>M120 스트레스."
    )
    desc_full = (
        "MC 동일 무작위 창. MABS 세 종 전구간 NAV·규칙4=DEF직진; "
        "RMABS-4tier-Gold/Cash 안전 레그 차이만·그 외 FSM 동일·RSI safe 블리드; "
        "DMABS-4tier-Gold/Cash는 동일 MA5>M120 스트레스·안전 레그만 금 vs 현금 1.0; "
        "RMABS-QLD/QQQ·벤치는 창 첫일 B&H."
    )
    meta = {
        "description": (
            desc_three
            if three_only
            else (
                desc_triple_ix
                if triple_abs_gold
                else (
                    desc_five_psq
                    if five_psq
                    else (desc_trio_dmabs if trio_dma else desc_full)
                )
            )
        ),
        "compare_three_only": three_only,
        "gold_mabs_dmabs": trio_dma,
        "five_gold_dmabs_psq": five_psq,
        "triple_mabs_rmabs_dmagold": triple_abs_gold,
        "include_dmabs_gold_defnor_qld": bool(args.include_dmabs_gold_defnor_qld),
        "dmabs_gold_defnor_qld_legs": (
            "Close_sig=QQQ Close_safe=금 Close_bounce=QLD Close_defense=QLD Close_agg=TQQQ, stress_bleed MA5>M120 FULL"
            if triple_abs_gold and bool(args.include_dmabs_gold_defnor_qld)
            else ""
        ),
        "triple_mode_ignores_root": False,
        "--root_requested": str(root.date()),
        "common_index": {"first": str(ix[0].date()), "last": str(ix[-1].date()), "days": len(ix)},
        "psq_note": psq_meta,
        "gold_series": gold_note,
        "rmabs_tier_safe_gold": gold_note,
        "rmabs_tier_safe_cash": "Close_safe=1.0 고정 (_flat_cash)",
        "dmabs_tier_cash_note": (
            "RMABS-4tier-Cash와 동일 5종 정렬(QQQ 시그/def, QLD nor, TQQQ agg, safe=1), "
            "stress_bleed_mode=MA5_CROSS_MA120_FULL"
        ),
        "dmabs_tier_gold_note": (
            "RMABS-4tier-Gold와 동일 5종 정렬(안전=금·" + gold_note + "), "
            "stress_bleed_mode=MA5_CROSS_MA120_FULL"
        ),
        "tier_labels_row_order": [lab for lab, _ in summary_rows],
        "ma200_csv": str(csv_cache.resolve()),
        "--root": str(root.date()),
        "trail": float(args.trail),
        "mc_years": float(args.mc_years),
        "min_trading_days": int(lo),
        "mc_iters": int(args.mc_iters),
        "mc_seed": int(args.mc_seed),
        "capital": float(CAPITAL_START),
        "csv_windows": str(csv_path.resolve()),
    }

    json_blob = {"meta": meta, "distribution_by_series_metric": distro}
    title = (
        "# Monte Carlo: MABS-QQQ · RMABS-4tier-Gold · RMABS-4tier-Cash\n\n"
        if three_only
        else (
            "# Monte Carlo: MABS-QQQ · RMABS-4tier-Gold · DMABS-4tier-Gold · DMABS-Gold(def·nor=QLD)\n\n"
            if triple_abs_gold and bool(args.include_dmabs_gold_defnor_qld)
            else (
                "# Monte Carlo: MABS-QQQ · RMABS-4tier-Gold · DMABS-4tier-Gold (--root 교집합)\n\n"
                if triple_abs_gold
                else (
                "# Monte Carlo: MABS · RMABS/DMABS-Gold · RMABS/DMABS-PSQ (PSQ 상장 이후 교집합)\n\n"
                if five_psq
                else (
                    "# Monte Carlo: MABS-QQQ · RMABS-4tier-Gold · DMABS-4tier-Gold · DMABS-4tier-Cash\n\n"
                    if trio_dma
                    else "# Monte Carlo: MABS×3 · RMABS-4tier-Gold/Cash · DMABS-Gold/Cash · RMABS×2 · 벤치×4\n\n"
                )
            )
            )
        )
    )
    if three_only:
        extra_bullet = (
            f"- **RMABS-4tier-Gold** 안전=금 · **RMABS-4tier-Cash** 안전=종가 1.0(무이자)\n"
            f"- 교집합: `{ix[0].date()}` ~ `{ix[-1].date()}` (**{len(ix)}일**) · 금 레그: {gold_note}\n"
        )
    elif triple_abs_gold and bool(args.include_dmabs_gold_defnor_qld):
        extra_bullet = (
            "- **포함 전략**: MABS-QQQ, RMABS-4tier-Gold, DMABS-4tier-Gold, "
            "**DMABS-4tier-Gold (방어·노말=QLD)** (방어도 QLD 로 통일한 DMABS).\n"
            "- **교집합**: ``--root`` 이후 QQQ·QLD·TQQQ·금 **모두 존재하는 거래일** 전 구간.\n"
            f"- 현재 실행: `{ix[0].date()}` ~ `{ix[-1].date()}` (**{len(ix)}일**) · 금 {gold_note}\n"
        )
    elif triple_abs_gold:
        extra_bullet = (
            "- **포함 전략**: MABS-QQQ, RMABS-4tier-Gold, DMABS-4tier-Gold만.\n"
            "- **교집합**: ``--root`` 이후 QQQ·QLD·TQQQ·금 **모두 존재하는 거래일** 전 구간.\n"
            f"- 현재 실행: `{ix[0].date()}` ~ `{ix[-1].date()}` (**{len(ix)}일**) · 금 {gold_note}\n"
        )
    elif five_psq:
        extra_bullet = (
            "- **레그 요약**: 시그 QQ / 방 QQ / 노 QLD / 공 TQQQ\n"
            "- **RMABS-4tier-Gold** / **DMABS-4tier-Gold**: 안전=금("
            f"{gold_note}) — DMABS는 RSI 대신 **MA5>MA120** 스트레스 블리드\n"
            "- **RMABS-4tier-PSQ** / **DMABS-4tier-PSQ**: 레그 동일 · 안전=**PSQ**(Yahoo 수정주가 파일)\n"
            f"- {psq_meta}\n"
            f"- 공통 교집합: `{ix[0].date()}` ~ `{ix[-1].date()}` (**{len(ix)}일**) · 금: {gold_note}\n"
        )
    elif trio_dma:
        extra_bullet = (
            "- **레그 요약**: 시그 QQ / 방 QQ / 노 QLD / 공 TQQQ\n"
            "- **RMABS-4tier-Gold** / **DMABS-4tier-Gold**: 안전=금("
            f"{gold_note}) — DMABS는 RSI 대신 **MA5>MA120** 스트레스 블리드\n"
            "- **DMABS-4tier-Cash**: 안전=**종가 1.0**(무이자), 스트레스 동일(MA5>M120 → 방어)\n"
            f"- 교집합: `{ix[0].date()}` ~ `{ix[-1].date()}` (**{len(ix)}일**)\n"
        )
    else:
        extra_bullet = (
            f"- 교집합: `{ix[0].date()}` ~ `{ix[-1].date()}` (**{len(ix)}일**) · "
            f"RMABS-4tier-Gold·DMABS-Gold 안전=금 · RMABS-4tier-Cash·DMABS-Cash 안전=1.0 · {gold_note}\n"
            "- 각 전략별 **전구간 1회 NAV** 후 창별 슬라이스 (**벤치만** 각 창 첫 거래일 B&H).\n"
        )

    md_text = (
        title
        + f"설정: 최소 거래연 **{args.mc_years:g}년**(거래일 ≥ {lo}일), 시행 **{args.mc_iters}회**, seed={args.mc_seed}, trail={args.trail}.\n\n"
        + extra_bullet
        + "\n"
        + f"## 무작위 창 지표 ({args.mc_iters}회·동일 표본별 슬라이스): {'평균·중앙' if triple_abs_gold else '평균'}\n\n"
        + "\n".join(table_lines)
        + "\n\n## 참고 분포 통계(JSON)\n\n"
        f"파일 `{json_path.name}` 에 시리즈별·지표별 mean/std/median/p05~p95가 있습니다.\n\n"
        "## 원시 행 데이터\n\n"
        f"`{csv_path.name}` (창 시작·종료 포함, 시행별 전 컬럼)\n"
    )

    md_path.write_text(md_text.strip() + "\n", encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(json_blob, fh, indent=2, ensure_ascii=False)

    print(
        "=== MC three-way (MABS-QQQ · RMABS-4tier-Gold · RMABS-4tier-Cash) 완료 ==="
        if three_only
        else (
            (
                "=== MC (MABS · RMABS-Gold · DMABS-Gold · DMABS-Gold def·nor=QLD) 완료 ==="
                if bool(args.include_dmabs_gold_defnor_qld)
                else "=== MC (MABS-QQQ · RMABS-Gold · DMABS-Gold, --root 교집합) 완료 ==="
            )
            if triple_abs_gold
            else (
                "=== MC (MABS + Gold×2 + PSQ×2, PSQ 상장 이후 교집합) 완료 ==="
                if five_psq
                else (
                    "=== MC (MABS-QQQ · RMABS-Gold · DMABS-Gold · DMABS-Cash) 완료 ==="
                    if trio_dma
                    else "=== MC suite (MABS×3+RMABS-4tier-Gold/Cash+DMABS-Gold/Cash+RMABS×2+벤치×4) 완료 ==="
                )
            )
        )
    )
    print(f"{meta['csv_windows']}")
    print(f"{md_path.resolve()}")
    print(f"{json_path.resolve()}")

    print("\n" + "\n".join(table_lines))


if __name__ == "__main__":
    main()
