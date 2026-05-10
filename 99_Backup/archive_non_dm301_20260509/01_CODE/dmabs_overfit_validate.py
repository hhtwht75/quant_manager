"""DMABS-TQQQ-QLD-(Gold|Cash|QQQ) 오버피팅 점검 패키지 실행.

무엇을 함
---------
1. IS / OOS(고정 분할) 각각 **다시 시작(cold)** 전구간 FSM NAV + 평가지표
2. 동일 RNG로 IS-only / OOS-only **최소 3년 랜덤 창 MC**(전략 3종 동시 슬라이스)
3. OOS 일간수익에 **정적 블록 부트스트랩** Sharpe 참조분포 (`evaluation_metrics`)
4. QLD **buy-hold 같은 창 길이** 페어부트스트랩 초과 검정(OOS 일간공선)
5. **파라미터 좁은 격자**(trail·mult_down·mult_bounce_up·stress MA 창): Gold 레그 한 종만 표
6. **거래 발생일** 라운드트립 bps 차감 NAV 재평가(요약표)
7. **다중 검정 참고**(3종 순위 · Bonferroni α 요약 문자열)

예::

  python3 01_CODE/dmabs_overfit_validate.py \\
    --root 1999-03-10 --global-end 2026-04-30 --oos-start 2019-11-02 \\
    --mc-years 3 --mc-iters 2000 --mc-seed 42

저장::
  ``03_RESULT/sensitivity/dmabs_overfit_audit_<ROOT>_<END>_oos<OOS>_*.json`` 및 같은 stem ``_TABLE.md``

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

from rmabs_gold_simulation import load_gold_series  # noqa: E402
from backtest_switching import load_extended_daily  # noqa: E402
from evaluation_metrics import (  # noqa: E402
    full_metrics,
    oos_metric_bundle,
    paired_bootstrap_compare,
    stationary_bootstrap_sharpe_distribution,
)
from fsm_four_asset_strategy import (  # noqa: E402
    CAPITAL_START,
    DEFAULT_TRAIL,
    MA_WINDOW,
    _align_five,
    get_or_build_ma200,
    mc_min_trading_days,
    run_fsm_backtest,
)
from fsm_mc_suite_fsm3_rmabs2_bench4 import METS  # noqa: E402
from fsm_four_asset_strategy import _core_metrics, _distribution_block, _flatten_metrics  # noqa: E402

OUT_BASE = _ROOT / "03_RESULT" / "sensitivity"


def _flat_cash(idx: pd.DatetimeIndex) -> pd.DataFrame:
    one = pd.Series(1.0, index=idx.sort_values(), dtype=float)
    return pd.DataFrame({"Open": one, "High": one, "Low": one, "Close": one})


def build_ix_and_frames(
    root: pd.Timestamp,
    glob_end: pd.Timestamp,
):
    sg = load_extended_daily("QQQ")
    ng = load_extended_daily("QLD")
    ag = load_extended_daily("TQQQ")
    ix0_all = sg.index.intersection(ng.index).intersection(ag.index).sort_values()
    ix0 = ix0_all[(ix0_all >= root) & (ix0_all <= glob_end)]
    gold_df, gold_note = load_gold_series(ix0[0], ix0[-1])
    ix = ix0.intersection(gold_df.index).sort_values()
    if len(ix) < MA_WINDOW:
        raise SystemExit(f"교집합 {len(ix)}일 < MA200 {MA_WINDOW}")
    ma_full, csv_cache = get_or_build_ma200(sg, "QQQ", force_rebuild=False)
    sg_i, ng_i, ag_i = sg.reindex(ix), ng.reindex(ix), ag.reindex(ix)
    g_i = gold_df.reindex(ix)
    cash_i = _flat_cash(ix)
    aligned = {
        "Gold": _align_five(sg_i, g_i, ng_i, ng_i, ag_i),
        "Cash": _align_five(sg_i, cash_i, ng_i, ng_i, ag_i),
        "QQQ": _align_five(sg_i, sg_i, ng_i, ng_i, ag_i),
    }
    mav = ma_full.reindex(ix)
    return ix, aligned, mav, gold_note, csv_cache


def run_nav_stress(
    aligned: pd.DataFrame,
    mav: pd.Series,
    *,
    trail: float,
    mult_down: float,
    mult_bounce_up: float,
    ma_fast: int,
    ma_slow: int,
) -> tuple[pd.Series, pd.DataFrame]:
    return run_fsm_backtest(
        aligned,
        mav,
        trail_stop=trail,
        initial_capital=CAPITAL_START,
        use_safe_ma_rule=True,
        stress_bleed_mode="MA5_CROSS_MA120_FULL",
        mult_down=mult_down,
        mult_bounce_up_to_agg=mult_bounce_up,
        stress_ma_fast=ma_fast,
        stress_ma_slow=ma_slow,
    )


def nav_with_event_bps_drag(
    nav: pd.Series,
    events: pd.DataFrame,
    round_trip_bps: float,
) -> pd.Series:
    """전이가 있는 날 라운드트립 비용을 일간 수익률에서 차감하는 근사."""
    if round_trip_bps <= 0:
        return nav
    rets = nav.pct_change().fillna(0.0).copy()
    ev = events.copy()
    ev["Date"] = pd.to_datetime(ev["Date"]).dt.normalize()
    ev = ev.set_index("Date")
    drag = float(round_trip_bps) / 10000.0
    for d in nav.index:
        if d not in ev.index:
            continue
        tr = str(ev.loc[d, "transitions"])
        if tr and tr.strip() and tr != "nan":
            rets.loc[d] = rets.loc[d] - drag
    adj = (1.0 + rets).cumprod() * float(nav.iloc[0])
    adj.name = nav.name
    return adj


def mc_slice_multi(
    nav_map: dict[str, pd.Series],
    ix: pd.DatetimeIndex,
    *,
    lo: int,
    mc_iters: int,
    seed: int,
) -> pd.DataFrame:
    n_dates = len(ix)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for _ in range(mc_iters):
        s_pos = int(rng.integers(0, n_dates - lo + 1))
        l_max = n_dates - s_pos
        l_win = int(rng.integers(lo, l_max + 1))
        win_ix = ix[s_pos : s_pos + l_win]
        row: dict[str, object] = {
            "window_start": str(win_ix[0].date()),
            "window_end": str(win_ix[-1].date()),
            "trading_days": int(l_win),
        }
        for label, nav in nav_map.items():
            sub = nav.iloc[s_pos : s_pos + l_win]
            pref = f"dmabs_{label.lower()}_"
            m = _core_metrics(sub)
            for k, v in m.items():
                row[pref + k] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_mc_df(df: pd.DataFrame, keys: tuple[str, ...]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for lab in keys:
        pref = f"dmabs_{lab.lower()}_"
        blk: dict[str, dict[str, float]] = {}
        for mk in METS:
            col = pref + mk
            if col not in df.columns:
                continue
            s = df[col].astype(float)
            blk[mk] = _distribution_block(s)
        out[lab] = blk
    return out


def rolling_chunk_metrics(nav: pd.Series, *, chunk_trading_days: int) -> list[dict[str, object]]:
    """약 N거래일 비중첩 창 순회 (워크포워드 스타일 평가 패널, 파라미터 고정 전제)."""
    ix = nav.index.sort_values().values
    out: list[dict[str, object]] = []
    i = 0
    cid = 0
    while i + chunk_trading_days <= len(ix):
        chunk_ix = ix[i : i + chunk_trading_days]
        sl = nav.loc[chunk_ix]
        mb = oos_metric_bundle(sl)
        ts0 = pd.Timestamp(chunk_ix[0])
        ts1 = pd.Timestamp(chunk_ix[-1])
        out.append(
            {
                "chunk_id": cid,
                "first": str(ts0.date()),
                "last": str(ts1.date()),
                **{k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in mb.items()},
            }
        )
        cid += 1
        i += chunk_trading_days
    return out


def fmt_md_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    hdr = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([hdr, sep] + body)


def main() -> None:
    ap = argparse.ArgumentParser(description="DMABS 과최적화 점검 감사( IS/OOS·MC·부트스트랩·민감도·비용 )")
    ap.add_argument("--root", default="1999-03-10")
    ap.add_argument("--global-end", default="2026-04-30", help="금·QQ 교집합 상한(포함)")
    ap.add_argument("--oos-start", default="2019-11-02", help="OOS 시작일(Inclusive)·IS는 이보다 앞)")
    ap.add_argument("--trail", type=float, default=DEFAULT_TRAIL)
    ap.add_argument("--mc-years", type=float, default=3.0)
    ap.add_argument("--mc-iters", type=int, default=2000)
    ap.add_argument("--mc-seed", type=int, default=42)
    ap.add_argument("--boot-iters", type=int, default=800)
    ap.add_argument("--boot-block-len", type=int, default=60)
    ap.add_argument("--boot-seed", type=int, default=91)
    ap.add_argument("--paired-boot-iters", type=int, default=500)
    ap.add_argument("--chunk-days", type=int, default=630, help="롤 패널 1청크 거래일(약 2.5년)")
    ap.add_argument(
        "--cost-bps-grid",
        type=str,
        default="0,10,25",
        help="라운드트립 bps 목록 쉼표(전이 발생일 차감)",
    )
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    root = pd.Timestamp(args.root.strip())
    glob_end = pd.Timestamp(args.global_end.strip())
    oos_cut = pd.Timestamp(args.oos_start.strip())
    trail = float(args.trail)

    ix, aligned, mav, gold_note, ma_csv_cache = build_ix_and_frames(root, glob_end)
    lo = mc_min_trading_days(args.mc_years)
    if len(ix) < lo:
        raise SystemExit(f"전체 교집합 {len(ix)} < 최소 MC창 {lo}")

    ix_is = ix[ix < oos_cut]
    ix_oos = ix[ix >= oos_cut]
    if len(ix_is) < MA_WINDOW or len(ix_oos) < MA_WINDOW:
        raise SystemExit("IS 또는 OOS 구간이 MA200 형성에 비해 너무 짧음")

    # 기준 파라미터 전체 NAV
    nav_evt: dict[str, tuple[pd.Series, pd.DataFrame]] = {}
    for lab, adf in aligned.items():
        n, ev = run_nav_stress(
            adf,
            mav,
            trail=trail,
            mult_down=0.97,
            mult_bounce_up=1.03,
            ma_fast=5,
            ma_slow=120,
        )
        nav_evt[lab] = (n.astype(float), ev)

    variants = ("Gold", "Cash", "QQQ")

    def metrics_block(ix_sub: pd.DatetimeIndex) -> dict[str, dict[str, float]]:
        outv: dict[str, dict[str, float]] = {}
        for lab in variants:
            nav, ev = nav_evt[lab]
            sl = nav.reindex(ix_sub).dropna()
            if len(sl) < 2:
                mb = {}
            else:
                mb = full_metrics(sl)
                mb = {k: float(mb[k]) for k in mb}
            outv[f"DMABS-TQQQ-QLD-{lab}"] = mb
        return outv

    is_metrics = metrics_block(ix_is)
    oos_metrics = metrics_block(ix_oos)

    # cold 재시작 IS/OOS 각각 따로 실행 (패널용)
    def nav_on_subset(ix_sub: pd.DatetimeIndex) -> dict[str, pd.Series]:
        mav_s = mav.reindex(ix_sub)
        res: dict[str, pd.Series] = {}
        for lab, full_ad in aligned.items():
            ad_sub = full_ad.reindex(ix_sub)
            nv, _ = run_nav_stress(
                ad_sub,
                mav_s,
                trail=trail,
                mult_down=0.97,
                mult_bounce_up=1.03,
                ma_fast=5,
                ma_slow=120,
            )
            res[lab] = nv.astype(float)
        return res

    nav_is_cold = nav_on_subset(ix_is)
    nav_oos_cold = nav_on_subset(ix_oos)

    mc_full = mc_slice_multi(
        {lab: nav_evt[lab][0] for lab in variants}, ix, lo=lo, mc_iters=args.mc_iters, seed=args.mc_seed
    )
    mc_is = mc_slice_multi(nav_is_cold, ix_is, lo=lo, mc_iters=min(args.mc_iters, 800), seed=args.mc_seed)
    lo_oos = min(lo, max(252, len(ix_oos) // 4))
    if len(ix_oos) < lo_oos + 10:
        mc_oos = pd.DataFrame()
        mc_oos_note = "OOS 교집합이 최소 창 길이에 비해 짧아 MC 생략"
    else:
        mc_oos = mc_slice_multi(
            nav_oos_cold, ix_oos, lo=lo_oos, mc_iters=min(args.mc_iters, 800), seed=args.mc_seed + 13
        )
        mc_oos_note = f"OOS 최소 창 lo_oos={lo_oos}"

    # OOS 블록 부트 (Gold 신호종 vs buy-hold QLD)
    ql = load_extended_daily("QLD").reindex(ix_oos)["Close"].astype(float)
    nav_g_oos = nav_evt["Gold"][0].reindex(ix_oos).dropna()
    qld_bh = CAPITAL_START * (ql.loc[nav_g_oos.index] / ql.loc[nav_g_oos.index].iloc[0])
    rg = nav_g_oos.pct_change().dropna().values
    r_b = qld_bh.pct_change().dropna().values
    n_sb = min(len(rg), len(r_b))
    rg_aligned = rg[-n_sb:]
    r_b_aligned = r_b[-n_sb:]
    boot_uni = stationary_bootstrap_sharpe_distribution(
        rg_aligned, block_len=args.boot_block_len, n_iter=args.boot_iters, seed=args.boot_seed
    )
    boot_pair = paired_bootstrap_compare(
        pd.Series(rg_aligned).reset_index(drop=True),
        pd.Series(r_b_aligned).reset_index(drop=True),
        block_len=args.boot_block_len,
        n_iter=args.paired_boot_iters,
        seed=args.boot_seed + 1,
    )

    # 롤 패널: 전체 NAV Gold
    roll_gold = rolling_chunk_metrics(nav_evt["Gold"][0], chunk_trading_days=int(args.chunk_days))

    # 민감도 표 (Gold 단일)
    sens_grid_rows: list[dict[str, object]] = []
    for trails in [0.82, DEFAULT_TRAIL, 0.88]:
        for md in [0.96, 0.97, 0.98]:
            for mf in [(5, 120), (5, 100), (7, 120)]:
                nv, _ = run_nav_stress(
                    aligned["Gold"],
                    mav,
                    trail=trails,
                    mult_down=md,
                    mult_bounce_up=1.03,
                    ma_fast=mf[0],
                    ma_slow=mf[1],
                )
                m = full_metrics(nv.astype(float))
                sens_grid_rows.append(
                    {
                        "trail": trails,
                        "mult_down": md,
                        "ma_fast": mf[0],
                        "ma_slow": mf[1],
                        "cagr": float(m["cagr"]),
                        "sharpe": float(m["sharpe"]),
                        "mdd": float(m["mdd"]),
                    }
                )

    # 비용 스트레스 (전이 발생일)
    bps_levels = [int(x.strip()) for x in args.cost_bps_grid.split(",") if x.strip()]
    cost_block: dict[str, dict[str, float]] = {}
    for bps in bps_levels:
        rowd: dict[str, float] = {}
        for lab in variants:
            nav0, ev0 = nav_evt[lab]
            nadj = nav_with_event_bps_drag(nav0, ev0, bps)
            m = full_metrics(nadj.astype(float))
            rowd[f"{lab}_cagr"] = float(m["cagr"])
            rowd[f"{lab}_sharpe"] = float(m["sharpe"])
            rowd[f"{lab}_mdd"] = float(m["mdd"])
        cost_block[str(bps)] = rowd

    # 순위 및 Bonferroni 문자열 (OOS CAGR 기준)
    o_cagr_sorted = sorted(
        [(lab, float(oos_metrics[f"DMABS-TQQQ-QLD-{lab}"].get("cagr", 0.0))) for lab in variants],
        key=lambda x: x[1],
        reverse=True,
    )
    rank_lines = []
    for rnk, (lab, cg) in enumerate(o_cagr_sorted, start=1):
        rank_lines.append(f"{rnk}. DMABS-TQQQ-QLD-{lab}: CAGR(OOS cold)={cg * 100:.2f}%")
    bonf = (
        "3종 안전 레그 후보비교에 대략적인 Bonferroni 참고: "
        "종합 α=5% 라면 각 단일 검정 α≈0.05/3=0.0167 수준까지 엄격히 볼 필요."
    )

    out_root = Path(args.out_dir) if args.out_dir.strip() else OUT_BASE
    out_root.mkdir(parents=True, exist_ok=True)
    stem = (
        f"dmabs_overfit_audit_{pd.Timestamp(ix[0]).strftime('%Y%m%d')}_"
        f"{pd.Timestamp(ix[-1]).strftime('%Y%m%d')}_oos{oos_cut.strftime('%Y%m%d')}_"
        f"{args.mc_years:g}y_mc{args.mc_iters}_s{args.mc_seed}"
    )
    outp = out_root / f"{stem}.json"

    mc_oos_summ = summarize_mc_df(mc_oos, variants) if not mc_oos.empty else {}

    blob = {
        "meta": {
            "gold_series": gold_note,
            "ma200_csv": str(ma_csv_cache.resolve()),
            "ix_days": len(ix),
            "is_days": len(ix_is),
            "oos_days": len(ix_oos),
            "oos_start_inclusive": str(oos_cut.date()),
            "bonferroni_note": bonf,
        },
        "is_full_metrics_trimmed_calendar": is_metrics,
        "oos_full_metrics_trimmed_calendar": oos_metrics,
        "oos_rank_by_cagr_cold_trim": [{"name": name, "cagr_apx": cg} for name, cg in o_cagr_sorted],
        "bootstrap_oos_stationary_block_sharpe_gold_daily": boot_uni,
        "bootstrap_oos_paired_vs_qld_buyhold_daily": boot_pair,
        "rolling_chunk_gold_full_nav": roll_gold,
        "sensitivity_grid_gold": sens_grid_rows,
        "cost_bps_roundtrip_on_transitions": cost_block,
        "mc_full_summary_distribution": summarize_mc_df(mc_full, variants),
        "mc_is_only_summary_distribution": summarize_mc_df(mc_is, variants),
        "mc_oos_cold_summary_distribution": mc_oos_summ,
        "mc_oos_note": mc_oos_note,
    }
    outp.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")

    # MD 요약표
    rows_io = []
    for lab in variants:
        k = f"DMABS-TQQQ-QLD-{lab}"
        mi, mo = is_metrics[k], oos_metrics[k]
        rows_io.append(
            (
                lab,
                f"{mi.get('cagr', 0) * 100:.2f}%",
                f"{mi.get('sharpe', 0):.2f}",
                f"{mo.get('cagr', 0) * 100:.2f}%",
                f"{mo.get('sharpe', 0):.2f}",
            )
        )

    tbl_io = fmt_md_table(
        ("변형", "IS CAGR", "IS Sharpe", "OOS CAGR", "OOS Sharpe"),
        [(a, b, c, d, e) for a, b, c, d, e in rows_io],
    )
    frac_ge = boot_uni.get("fraction_boot_ge_obs") or 0.0
    pct_obs = boot_uni.get("obs_percentile_approx") or 0.0

    tbl_cost = "| bps | Gold Sh | Cash Sh | QQQ Sh | Gold CAGR | Cash CAGR | QQQ CAGR |\n|---:|---:|---:|---:|---:|---:|---:|"
    for bps_str, rr in sorted(cost_block.items(), key=lambda x: int(x[0])):
        tbl_cost += (
            f"\n| {bps_str} | {rr['Gold_sharpe']:.3f} | {rr['Cash_sharpe']:.3f} | {rr['QQQ_sharpe']:.3f} "
            f"| {rr['Gold_cagr'] * 100:.2f}% | {rr['Cash_cagr'] * 100:.2f}% | {rr['QQQ_cagr'] * 100:.2f}% |"
        )

    tbl_sens_hdr = "| trail | md | mf/ms | CAGR | Sharpe | MDD |\n|---:|---:|:---:|---:|---:|---:|"
    tbl_sens_body = ""
    for r in sens_grid_rows:
        tbl_sens_body += (
            f"\n| {r['trail']} | {r['mult_down']} | {r['ma_fast']}/{r['ma_slow']} | "
            f"{float(r['cagr']) * 100:.2f}% | {float(r['sharpe']):.3f} | {float(r['mdd']) * 100:.2f}% |"
        )

    md_txt = (
        f"# DMABS 과최적화 점검 요약 ({ix[0].date()} ~ {ix[-1].date()})\n\n"
        f"- OOS 시작(포함): **{oos_cut.date()}** 이전 일수 IS={len(ix_is)}, 이후 OOS={len(ix_oos)}.\n"
        f"- {bonf}\n"
        "## IS / OOS (같은 캘린더에서 NAV 슬라이스; cold 서브 재런 NAV는 MC 분기 참고용)\n\n"
        + tbl_io
        + "\n\n## OOS 블록 부트 (Gold 일간수익, 신호종)\n\n"
        + f"- 관측 Sharpe: **{boot_uni.get('obs_sharpe')}**\n"
        + f"- 부트 중복 샘플에서 Sharpe ≥ 관측 비율(낮으면 부드러운 과장 약함): **{frac_ge:.3f}**\n"
        + f"- 관측 샤프의 부트 분포 대략 백분위: **~{pct_obs * 100:.1f}%** (낮음=더 드물게 재현되는 공격 성과)\n"
        "## OOS Gold vs 동기간 QLD buy-hold (페이어드 블록 부트 차이)\n\n"
        + f"- ΔSharpe 중앙값 p-value 근사: **{boot_pair['delta_sharpe']['p_value']:.4f}** "
        f"(prob Δ>0 → {boot_pair['delta_sharpe']['prob_better']:.3f})\n"
        "## 비용(전이일 라운드트립 bps)\n\n"
        + tbl_cost
        + "\n\n## 민감도 소격자 (Gold, narrow grid)\n\n"
        + tbl_sens_hdr
        + tbl_sens_body
        + "\n\n## 순위 요약(OOS CAGR, cold 블록)\n\n"
        + "\n".join(f"- {ln}" for ln in rank_lines)
        + "\n\n## 산출 JSON\n\n"
        f"`{outp.name}`\n"
    )
    outp_md = outp.with_suffix("")
    outp_md = out_root / (stem + "_TABLE.md")
    outp_md.write_text(md_txt.strip() + "\n", encoding="utf-8")

    print(outp.resolve())
    print(outp_md.resolve())


if __name__ == "__main__":
    main()
