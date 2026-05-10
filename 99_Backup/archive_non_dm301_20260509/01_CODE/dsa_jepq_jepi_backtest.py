"""
dsa_jepq_jepi_backtest.py
=========================
DSA(strategy_s5, benchmark DSA와 동일 파라미터)에서 NORMAL 방어자산을
QQQ / JEPQ / JEPI 로 바꿔 동일 QLD·TQQQ 신호로 비교.

• QQQ·QLD·TQQQ: yahoo_extended 확장 일봉
• JEPQ: 02_DATA/yahoo_extended/JEPQ/JEPQ_daily.csv
• JEPI: 02_DATA/yahoo/JEPI/JEPI_daily.csv 없으면 Yahoo(max)로 수신 후 저장

비교 구간: 위 다섯 시계열의 교집합 (JEPQ 상장 2022-05-04 이후로 제한).

출력: 03_RESULT/dsa_jepq_jepi_compare*.png/json
  --start / --end 로 평가 구간 지정 가능.

실행 예:
  python3 01_CODE/dsa_jepq_jepi_backtest.py
  python3 01_CODE/dsa_jepq_jepi_backtest.py --start 2023-06-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import load_extended_daily, strategy_s5  # noqa: E402
from evaluation_metrics import full_metrics  # noqa: E402

DATA_DIR = _ROOT / "02_DATA"
OUT_DIR = _ROOT / "03_RESULT"

S5_ANCHOR_COMMON = dict(
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

INIT = 100_000.0


def parse_args():
    p = argparse.ArgumentParser(description="DSA QQQ vs JEPQ vs JEPI")
    p.add_argument(
        "--start",
        type=str,
        default="",
        help="평가 시작일 YYYY-MM-DD (포함). 비우면 공통구간 첫날",
    )
    p.add_argument(
        "--end",
        type=str,
        default="",
        help="평가 종료일 YYYY-MM-DD (포함). 비우면 공통구간 마지막날",
    )
    return p.parse_args()


def slice_common(
    common: pd.DatetimeIndex,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> pd.DatetimeIndex:
    c = common.sort_values()
    if start_ts is not None:
        c = c[c >= start_ts]
    if end_ts is not None:
        c = c[c <= end_ts]
    return c


def load_jepq() -> pd.DataFrame:
    csv_path = DATA_DIR / "yahoo_extended" / "JEPQ" / "JEPQ_daily.csv"
    if not csv_path.exists():
        raise SystemExit(f"JEPQ CSV 없음: {csv_path}")
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close"]]


def load_or_fetch_yahoo_plain(ticker: str) -> pd.DataFrame:
    """수정주가 일봉. 02_DATA/yahoo/{ticker}/{ticker}_daily.csv 캐시."""
    dest_dir = DATA_DIR / "yahoo" / ticker
    csv_path = dest_dir / f"{ticker}_daily.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        df.index.name = "Date"
        return df[["Open", "High", "Low", "Close"]]

    import yfinance as yf

    dest_dir.mkdir(parents=True, exist_ok=True)
    raw = yf.download(
        ticker,
        period="max",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise SystemExit(f"{ticker} Yahoo 다운로드 결과가 비었습니다.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    raw.index.name = "Date"
    raw.to_csv(csv_path)
    return raw[["Open", "High", "Low", "Close"]].copy()


def buy_and_hold(close_df: pd.DataFrame, name: str, cap: float = INIT) -> pd.Series:
    c = close_df["Close"]
    sh = cap / c.iloc[0]
    return pd.Series(sh * c.values, index=c.index, name=name)


def main():
    args = parse_args()
    start_ts = pd.Timestamp(args.start.strip()) if args.start.strip() else None
    end_ts = pd.Timestamp(args.end.strip()) if args.end.strip() else None
    range_requested = bool(args.start.strip() or args.end.strip())

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qqq = load_extended_daily("QQQ")
    qld = load_extended_daily("QLD")
    tqqq = load_extended_daily("TQQQ")
    jepq = load_jepq()
    jepi = load_or_fetch_yahoo_plain("JEPI")

    common = (
        qqq.index.intersection(qld.index)
        .intersection(tqqq.index)
        .intersection(jepq.index)
        .intersection(jepi.index)
    )
    common = slice_common(common, start_ts, end_ts)
    if len(common) < 50:
        raise SystemExit(
            f"공통 거래일이 너무 적습니다: {len(common)} "
            f"(요청 구간: start={start_ts}, end={end_ts})"
        )

    Q = qqq.loc[common]
    L = qld.loc[common]
    T = tqqq.loc[common]
    Jq = jepq.loc[common]
    Ji = jepi.loc[common]

    kw = dict(stop_factor=0.75, **S5_ANCHOR_COMMON)
    dsa_q, ev_q = strategy_s5(Q, L, T, INIT, series_name="DSA 방어=QQQ", **kw)
    dsa_jq, ev_jq = strategy_s5(Jq, L, T, INIT, series_name="DSA 방어=JEPQ", **kw)
    dsa_ji, ev_ji = strategy_s5(Ji, L, T, INIT, series_name="DSA 방어=JEPI", **kw)

    series_map = {
        "DSA 방어=QQQ": dsa_q,
        "DSA 방어=JEPQ": dsa_jq,
        "DSA 방어=JEPI": dsa_ji,
        "B&H QQQ": buy_and_hold(Q, "B&H QQQ"),
        "B&H JEPQ": buy_and_hold(Jq, "B&H JEPQ"),
        "B&H JEPI": buy_and_hold(Ji, "B&H JEPI"),
    }

    out_stem = (
        f"dsa_jepq_jepi_compare_{common[0].strftime('%Y%m%d')}_{common[-1].strftime('%Y%m%d')}"
        if range_requested
        else "dsa_jepq_jepi_compare"
    )

    print("=" * 72)
    print("DSA 방어자산 비교: QQQ vs JEPQ vs JEPI")
    if range_requested:
        print(f"  요청 필터: start={start_ts}, end={end_ts}")
    print(f"  기간: {common[0].date()} ~ {common[-1].date()}  ({len(common):,} 거래일)")
    print(
        "  DSA: β=0.5, min/max drop 10%/20%, exp(b=2) 25%→100%, "
        "stop_factor=0.75, trail=-15%"
    )
    print("=" * 72)

    rows = []
    for label, s in series_map.items():
        m = full_metrics(s)
        rows.append({"label": label, **{k: float(m[k]) for k in m}})
        print(
            f"  {label:18s}  CAGR={m['cagr'] * 100:6.2f}%  "
            f"Sharpe={m['sharpe']:5.2f}  MDD={m['mdd'] * 100:6.2f}%  "
            f"총수익={(s.iloc[-1] / s.iloc[0] - 1) * 100:7.2f}%"
        )

    colors = {
        "DSA 방어=QQQ": "#2563EB",
        "DSA 방어=JEPQ": "#7C3AED",
        "DSA 방어=JEPI": "#EA580C",
        "B&H QQQ": "#93C5FD",
        "B&H JEPQ": "#C4B5FD",
        "B&H JEPI": "#FDBA74",
    }

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for label, s in series_map.items():
        lw = 1.8 if label.startswith("DSA") else 1.0
        ax.plot(s.index, s.values / INIT, label=label, color=colors[label], lw=lw, alpha=0.95)
    ax.set_title(
        "DSA NORMAL: QQQ vs JEPQ vs JEPI (동일 QLD/TQQQ 신호)\n"
        f"{common[0].date()} ~ {common[-1].date()}"
    )
    ax.set_ylabel("배수 (초기=1)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    png_path = OUT_DIR / f"{out_stem}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    out_json = {
        "start": str(common[0].date()),
        "end": str(common[-1].date()),
        "filter_start": str(start_ts.date()) if start_ts is not None else None,
        "filter_end": str(end_ts.date()) if end_ts is not None else None,
        "n_days": int(len(common)),
        "dsa_params": {**S5_ANCHOR_COMMON, "stop_factor": 0.75},
        "metrics": rows,
        "dsa_events": {"QQQ": len(ev_q), "JEPQ": len(ev_jq), "JEPI": len(ev_ji)},
    }
    json_path = OUT_DIR / f"{out_stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print(f"\n  저장: {png_path}")
    print(f"  저장: {json_path}")


if __name__ == "__main__":
    main()
