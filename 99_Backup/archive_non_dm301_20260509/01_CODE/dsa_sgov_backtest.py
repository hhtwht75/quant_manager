"""
dsa_sgov_backtest.py
====================
DSA 전략(strategy_s5 앵커, benchmark_s2_s5_anchor_table.py 와 동일 파라미터)에서
NORMAL(방어) 구간 보유자산을 QQQ vs SGOV(단기국채)로만 바꿔 동일 기간·동일 신호로 비교.

• SGOV: 데이터는 02_DATA/yahoo/SGOV/SGOV_daily.csv 가 없으면 Yahoo에서 내려받아 저장.
• 비교 구간: QQQ·QLD·TQQQ 확장 일봉과 SGOV 일봉의 **교집합** (SGOV 상장 이후로 제한됨).

출력: 콘솔 메트릭, 03_RESULT/dsa_sgov_compare*.json/png
  --start / --end 를 주면 파일명에 구간 태그가 붙음 (예: dsa_sgov_compare_20190101_20251231.png)

실행 예:
  python3 01_CODE/dsa_sgov_backtest.py
  python3 01_CODE/dsa_sgov_backtest.py --start 2022-01-01 --end 2024-12-31
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

from backtest_switching import (  # noqa: E402
    load_extended_daily,
    strategy_s5,
)
from evaluation_metrics import full_metrics  # noqa: E402

DATA_DIR = _ROOT / "02_DATA"
OUT_DIR = _ROOT / "03_RESULT"

S5_MAX_DROP = 0.20
S5_ANCHOR_COMMON = dict(
    beta=0.5,
    min_drop=0.10,
    max_drop=S5_MAX_DROP,
    trailing_stop_pct=-0.15,
    use_stop_loss=True,
    position_mode="exp",
    exp_frac_lo=0.25,
    exp_frac_hi=1.00,
    exp_base=2.0,
)

INIT = 100_000.0


def parse_args():
    p = argparse.ArgumentParser(description="DSA QQQ vs SGOV 방어자산 비교")
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


def load_or_fetch_sgov() -> pd.DataFrame:
    """Yahoo 일봉(OHLC). 로컬 CSV가 있으면 재사용."""
    dest_dir = DATA_DIR / "yahoo" / "SGOV"
    csv_path = dest_dir / "SGOV_daily.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        df.index.name = "Date"
        return df[["Open", "High", "Low", "Close"]]

    import yfinance as yf

    dest_dir.mkdir(parents=True, exist_ok=True)
    raw = yf.download(
        "SGOV",
        period="max",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise SystemExit("SGOV Yahoo 다운로드 결과가 비었습니다.")
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
    sgov = load_or_fetch_sgov()

    common = (
        qqq.index.intersection(qld.index)
        .intersection(tqqq.index)
        .intersection(sgov.index)
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
    S = sgov.loc[common]

    dsa_qqq, ev_q = strategy_s5(
        Q, L, T, INIT,
        stop_factor=0.75,
        series_name="DSA 방어=QQQ",
        **S5_ANCHOR_COMMON,
    )
    dsa_sgov, ev_s = strategy_s5(
        S, L, T, INIT,
        stop_factor=0.75,
        series_name="DSA 방어=SGOV",
        **S5_ANCHOR_COMMON,
    )

    bh_q = buy_and_hold(Q, "B&H QQQ")
    bh_s = buy_and_hold(S, "B&H SGOV")

    series_map = {
        "DSA 방어=QQQ": dsa_qqq,
        "DSA 방어=SGOV": dsa_sgov,
        "B&H QQQ": bh_q,
        "B&H SGOV": bh_s,
    }

    out_stem = (
        f"dsa_sgov_compare_{common[0].strftime('%Y%m%d')}_{common[-1].strftime('%Y%m%d')}"
        if range_requested
        else "dsa_sgov_compare"
    )

    print("=" * 72)
    print("DSA 방어자산 비교: QQQ vs SGOV")
    if range_requested:
        print(f"  요청 필터: start={start_ts}, end={end_ts}")
    print(f"  기간: {common[0].date()} ~ {common[-1].date()}  ({len(common):,} 거래일)")
    print(f"  DSA 파라미터: β=0.5, min_drop=10%, max_drop=20%, exp(b=2) 25%→100%, "
          f"stop_factor=0.75, trail=-15%")
    print("=" * 72)

    rows = []
    for label, s in series_map.items():
        m = full_metrics(s)
        rows.append({"label": label, **{k: float(m[k]) for k in m}})
        print(
            f"  {label:16s}  CAGR={m['cagr']*100:6.2f}%  "
            f"Sharpe={m['sharpe']:5.2f}  MDD={m['mdd']*100:6.2f}%  "
            f"총수익={(s.iloc[-1]/s.iloc[0]-1)*100:7.2f}%"
        )

    # 차트
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = {"DSA 방어=QQQ": "#2563EB", "DSA 방어=SGOV": "#16A34A",
              "B&H QQQ": "#93C5FD", "B&H SGOV": "#86EFAC"}
    for label, s in series_map.items():
        ax.plot(s.index, s.values / INIT, label=label, color=colors[label], lw=1.4)
    ax.set_title(
        f"DSA NORMAL 보유: QQQ vs SGOV (동일 QLD/TQQQ 신호)\n"
        f"{common[0].date()} ~ {common[-1].date()}"
    )
    ax.set_ylabel("배수 (초기=1)")
    ax.legend(loc="upper left", fontsize=9)
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
        "events_qqq": len(ev_q),
        "events_sgov": len(ev_s),
    }
    json_path = OUT_DIR / f"{out_stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print(f"\n  저장: {png_path}")
    print(f"  저장: {json_path}")


if __name__ == "__main__":
    main()
