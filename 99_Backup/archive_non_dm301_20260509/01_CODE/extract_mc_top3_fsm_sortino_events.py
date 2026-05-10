"""
MC windows CSV 에서 Sortino 로 FSM-defQQQ > FSM-defGold 초과폭 상위 3창 추출 후,
동일 교집합 전구간 FSM 재실행 로그에서 해당 일자구간 이벤트만 표시.

예::

    python3 01_CODE/extract_mc_top3_fsm_sortino_events.py \\
      --windows-csv 03_RESULT/sensitivity/mc_suite_fsm3_rmabs2_bh4_navslice_..._windows.csv \\
      --top 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from rmabs_gold_simulation import load_gold_series  # noqa: E402

from backtest_switching import load_extended_daily  # noqa: E402

from fsm_four_asset_strategy import (  # noqa: E402
    CAPITAL_START,
    DEFAULT_TRAIL,
    MA_WINDOW,
    _align_four,
    get_or_build_ma200,
    run_fsm_backtest,
    _align_five,
)


def _df_to_markdown(d: pd.DataFrame) -> str:
    d = d.fillna("")
    cols = [str(c) for c in d.columns]
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    head = "| " + " | ".join(cols) + " |"
    lines = [head, sep]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in d.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--windows-csv",
        type=str,
        default="",
        help="비우면 최신 이름 패턴 navslice_windows.csv 자동 탐색 실패 시 필수",
    )
    ap.add_argument("--root", default="2002-10-01")
    ap.add_argument("--trail", type=float, default=DEFAULT_TRAIL)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    wc = Path(args.windows_csv.strip()) if args.windows_csv.strip() else None
    if wc is None or not wc.exists():
        sens = _ROOT / "03_RESULT" / "sensitivity"
        cands = sorted(
            sens.glob("mc_suite_mabs3_rmabs4tier*_windows.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not cands:
            cands = sorted(
                sens.glob("mc_suite_fsm3_rmabs2_bh4_navslice*_windows.csv"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        if not cands:
            raise SystemExit("windows CSV 없음 · --windows-csv 로 지정")
        wc = cands[0]

    df = pd.read_csv(wc)
    col_candidates_q = (
        "mabs_qqq_sortino",
        "rmabs_4tier_gold_sortino",
        "rmabs_4tier_sortino",
        "fsm_defqqq_sortino",
    )
    col_candidates_g = ("mabs_gold_sortino", "fsm_defgold_sortino")
    col_q = next((c for c in col_candidates_q if c in df.columns), "")
    col_g = next((c for c in col_candidates_g if c in df.columns), "")
    if not col_q or not col_g:
        raise SystemExit(
            f"CSV에 QQ·Gold Sortino 호환 열 없음 시도했음: {col_candidates_q} / {col_candidates_g}"
        )

    df = df.assign(delta_sortino=df[col_q].astype(float) - df[col_g].astype(float))
    pos = df[df["delta_sortino"] > 0].copy()
    if len(pos) < args.top:
        top = df.nlargest(args.top, "delta_sortino")
        note_pos = "(Sortino 초과 폭 양수인 행 부족 — 전체에서 상위 채움)"
    else:
        top = pos.nlargest(args.top, "delta_sortino")
        note_pos = "(Sortino 초과폭 Δ>0 인 행 중 상위)"

    root_ts = pd.Timestamp(args.root.strip())
    sg = load_extended_daily("QQQ")
    ng = load_extended_daily("QLD")
    ag = load_extended_daily("TQQQ")
    ma_full, _ = get_or_build_ma200(sg, "QQQ", force_rebuild=False)
    ix0 = sg.index.intersection(ng.index).intersection(ag.index).sort_values()
    ix0 = ix0[ix0 >= root_ts]
    gold_df, gold_note = load_gold_series(ix0[0], ix0[-1])
    ix = ix0.intersection(gold_df.index).sort_values()
    if len(ix) < MA_WINDOW:
        raise SystemExit("교집합 부족")

    sg_i, ng_i, ag_i = sg.reindex(ix), ng.reindex(ix), ag.reindex(ix)
    g_i = gold_df.reindex(ix)
    use_mabs = col_q.startswith("mabs_")
    if use_mabs:
        aligned_qq = _align_four(sg_i, sg_i, ng_i, ag_i)
        aligned_au = _align_four(sg_i, g_i, ng_i, ag_i)
    else:
        aligned_qq = _align_five(sg_i, g_i, sg_i, ng_i, ag_i)
        aligned_au = _align_five(sg_i, g_i, g_i, ng_i, ag_i)
    mav = ma_full.reindex(ix)
    _, ev_qq = run_fsm_backtest(
        aligned_qq,
        mav,
        trail_stop=args.trail,
        initial_capital=CAPITAL_START,
        use_safe_ma_rule=(not use_mabs),
    )
    _, ev_au = run_fsm_backtest(
        aligned_au,
        mav,
        trail_stop=args.trail,
        initial_capital=CAPITAL_START,
        use_safe_ma_rule=(not use_mabs),
    )
    ev_qq["Date"] = pd.to_datetime(ev_qq["Date"]).dt.normalize()
    ev_au["Date"] = pd.to_datetime(ev_au["Date"]).dt.normalize()

    out_lines: list[str] = []
    title = "# MC 창별 FSM-defQQQ Sortino > Gold 상위 분석"
    out_lines.append(title)
    out_lines.append("")
    out_lines.append(f"- 사용한 시행표: `{wc}`")
    out_lines.append(f"- 안전 레그: 금(선물·GLD 일원화) `{gold_note}`")
    out_lines.append(f"- trail_stop: {args.trail}")
    out_lines.append(f"- 상위 선택: {note_pos}")
    out_lines.append("")

    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        ws = pd.Timestamp(str(row["window_start"])).normalize()
        we = pd.Timestamp(str(row["window_end"])).normalize()
        ds = row["delta_sortino"]
        sq = row[col_q]
        sg_ = row[col_g]

        sub_q = ev_qq[(ev_qq["Date"] >= ws) & (ev_qq["Date"] <= we)].copy()
        sub_a = ev_au[(ev_au["Date"] >= ws) & (ev_au["Date"] <= we)].copy()

        m = sub_q.merge(
            sub_a,
            on="Date",
            how="outer",
            suffixes=("_QQ_defQQQ", "_defGoldPX"),
            sort=True,
        )

        evt_only = m[
            m["transitions_QQ_defQQQ"].fillna("").astype(str).str.len().gt(0)
            | m["transitions_defGoldPX"].fillna("").astype(str).str.len().gt(0)
        ].copy()

        out_lines.append(f"## 시행 순위 #{rank}: `{ws.date()}` ~ `{we.date()}`")
        out_lines.append("")
        out_lines.append(
            f"| 항목 | FSM-defQQQ | FSM-defGold |\n|---|---:|---:|"
        )
        out_lines.append(
            f"| 이 창에서 CSV 기록 Sortino | {sq:.6f} | {sg_:.6f} |\n| Δ_sortino(QQ−Gold) | **{float(ds):.6f}** | — |\n"
        )
        out_lines.append(
            f"- 구간 거래일(시행표): **{int(row['trading_days'])}**, 전구간 이벤트로 펼친 일수: QQ **{len(sub_q)}**, Gold **{len(sub_a)}**"
        )
        out_lines.append(
            f"- **상태/체이닝 문자열이 빈 날 제외한 ‘이벤트 발생일’ 행수: {len(evt_only)}**"
        )
        out_lines.append("")

        disp = evt_only[
            [
                "Date",
                "regime_after_QQ_defQQQ",
                "transitions_QQ_defQQQ",
                "regime_after_defGoldPX",
                "transitions_defGoldPX",
                "nav_eod_QQ_defQQQ",
                "nav_eod_defGoldPX",
                "sig_QQ_defQQQ",
                "sig_defGoldPX",
            ]
        ].copy()
        disp["Date"] = disp["Date"].dt.strftime("%Y-%m-%d")
        for c in disp.columns:
            if c.startswith("sig_"):
                disp[c] = pd.to_numeric(disp[c], errors="coerce").round(4)
            if c.startswith("nav_eod"):
                disp[c] = pd.to_numeric(disp[c], errors="coerce").round(2)

        tbl = _df_to_markdown(disp)
        out_lines.append("### 이벤트 발생일만 (둘 중 하나라도 `transitions` 비어있지 않음)")
        out_lines.append("")
        out_lines.append(tbl)
        out_lines.append("")

    outp = wc.parent / f"{wc.stem.replace('_windows', '')}_TOP{args.top}_sortino_fsm_events_TABLE.md"
    outp.write_text("\n".join(out_lines), encoding="utf-8")
    print(outp.resolve())


if __name__ == "__main__":
    main()
