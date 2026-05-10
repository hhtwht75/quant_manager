"""
S4-2 / S5 동일 캘린더에 대해 **일별** 병합 타임라인 CSV 생성.

열: Date, QLD_Close, QLD_ATH, QLD_dd_pct, S42_tqqq_pct_eod, S5_tqqq_pct_eod, S42_event, S5_event

• TQQQ 비중 = 장 종가 시점 명목가치 비율 (전략 루프와 동일하게 일별 기록).
• QLD_ATH = 백테 시작일~해당일 QLD 종가 누적 최고가 (전략 ath 정의와 동일).

실행 예:
  python3 01_CODE/s42_s5_daily_timeline.py --start 2003-01-01 --end 2003-05-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import load_extended_daily, strategy_s5  # noqa: E402
from backtest_tiered import strategy_tiered  # noqa: E402

OUT_DIR = _ROOT / "03_RESULT" / "sensitivity"
CAP = 100_000
S5_ANCHOR = dict(
    beta=0.5,
    min_drop=0.10,
    max_drop=0.20,
    trailing_stop_pct=-0.15,
    use_stop_loss=True,
    position_mode="exp",
    exp_frac_lo=0.25,
    exp_frac_hi=1.00,
    exp_base=2.0,
)


def _fmt_row_event(r: pd.Series) -> str:
    t = str(r["type"])
    bits = [t]
    if t == "TO_ATTACK":
        if pd.notna(r.get("atk_frac")):
            bits.append(f"atk_frac={r['atk_frac']}")
        if pd.notna(r.get("drop_pct")):
            bits.append(f"drop_pct={r['drop_pct']}")
        if pd.notna(r.get("rebound_pct")):
            bits.append(f"rebound_pct={r['rebound_pct']}")
    if t == "STOP_LOSS" and pd.notna(r.get("reason")):
        bits.append(f"reason={r['reason']}")
    return " ".join(bits) if len(bits) > 1 else t


def _events_on_date(ev: pd.DataFrame, d: pd.Timestamp) -> str:
    if ev is None or ev.empty:
        return ""
    sub = ev[pd.to_datetime(ev["Date"]) == d]
    if sub.empty:
        return ""
    return "+".join(_fmt_row_event(r) for _, r in sub.iterrows())


def build_timeline(
    dates_out: pd.DatetimeIndex,
    qld_full: pd.Series,
    w42: list[float],
    w5: list[float],
    ev42: pd.DataFrame,
    ev5: pd.DataFrame,
    run_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    ath = qld_full.loc[run_dates].cummax()
    out_set = set(dates_out)
    rows = []
    for i, d in enumerate(run_dates):
        if d not in out_set:
            continue
        q = float(qld_full.iloc[i])
        a = float(ath.iloc[i])
        dd = q / a - 1.0
        rows.append({
            "Date": d,
            "QLD_Close": round(q, 4),
            "QLD_ATH": round(a, 4),
            "QLD_dd_pct": round(dd * 100, 2),
            "S42_tqqq_pct_eod": round(w42[i] * 100, 2),
            "S5_tqqq_pct_eod": round(w5[i] * 100, 2),
            "S42_event": _events_on_date(ev42, d),
            "S5_event": _events_on_date(ev5, d),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--root-start", default="2002-10-01",
                    help="백테스트 전체 시작(이벤트·비중 초기화 일치)")
    ap.add_argument("--entry-depth", type=float, default=None,
                    help="S5 entry_depth_stop_mult (미지정 시 drop_deepen)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qqq = load_extended_daily("QQQ")
    qld = load_extended_daily("QLD")
    tqqq = load_extended_daily("TQQQ")
    c = qqq.index.intersection(qld.index).intersection(tqqq.index)
    Qfull, Lfull, Tfull = qqq.loc[c], qld.loc[c], tqqq.loc[c]

    m = Qfull.index >= pd.Timestamp(args.root_start)
    Q, L, T = Qfull.loc[m], Lfull.loc[m], Tfull.loc[m]

    kw = {**S5_ANCHOR, "stop_factor": 0.75}
    if args.entry_depth is not None:
        kw["entry_depth_stop_mult"] = float(args.entry_depth)

    w42: list[float] = []
    w5: list[float] = []
    _, ev42 = strategy_tiered(Q, L, T, CAP, daily_tqqq_weight_out=w42)
    _, ev5 = strategy_s5(Q, L, T, CAP, daily_tqqq_weight_out=w5, **kw)

    mout = (Q.index >= pd.Timestamp(args.start)) & (Q.index <= pd.Timestamp(args.end))
    dates_out = Q.index[mout]

    df = build_timeline(
        dates_out,
        L["Close"].reindex(Q.index),
        w42,
        w5,
        ev42,
        ev5,
        Q.index,
    )
    tag = f"{args.start.replace('-','')}_{args.end.replace('-','')}"
    suf = f"_ed{args.entry_depth}" if args.entry_depth is not None else ""
    path = OUT_DIR / f"s42_s5_daily_timeline_{tag}{suf}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
