#!/usr/bin/env python3
"""에피소드 2 (2016-06-02 ~ 2017-06-02) 일자별 보유·체결 레코드 CSV 생성."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import (  # noqa: E402
    load_extended_daily,
    strategy_rsi_ma_based_switching,
    strategy_rsi_ma_based_switching_qqq_only,
)

OUT_DIR = _ROOT / "03_RESULT" / "sensitivity"
CAP = 100_000.0
EP_T0 = pd.Timestamp("2016-06-02")
EP_T1 = pd.Timestamp("2017-06-02")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qq = load_extended_daily("QQQ")
    ql = load_extended_daily("QLD")
    tq = load_extended_daily("TQQQ")
    c = qq.index.intersection(ql.index).intersection(tq.index)
    c = c[c >= pd.Timestamp("2002-10-01")]
    Q, L, T = qq.loc[c], ql.loc[c], tq.loc[c]

    aud_qq: list = []
    aud_qld: list = []
    strategy_rsi_ma_based_switching_qqq_only(Q, L, T, CAP, daily_audit=aud_qq)
    strategy_rsi_ma_based_switching(Q, L, T, CAP, daily_audit=aud_qld)

    df_q = pd.DataFrame(aud_qq)
    df_l = pd.DataFrame(aud_qld)
    df_q = df_q.rename(columns={"hold": "hold_RMQQ", "trade": "trade_RMQQ", "nav": "nav_RMQQ"})
    df_l = df_l.rename(columns={"hold": "hold_RMABS_QLD", "trade": "trade_RMABS_QLD", "nav": "nav_RMABS_QLD"})

    m = (df_q["Date"] >= EP_T0) & (df_q["Date"] <= EP_T1)
    df_q = df_q.loc[m].reset_index(drop=True)
    m2 = (df_l["Date"] >= EP_T0) & (df_l["Date"] <= EP_T1)
    df_l = df_l.loc[m2].reset_index(drop=True)

    out = df_q.merge(df_l, on="Date", how="outer").sort_values("Date")
    outp = OUT_DIR / "rmabs_episode2_20160602_20170602_daily_trades.csv"
    out.to_csv(outp, index=False, encoding="utf-8-sig")

    trx = out[
        (out["trade_RMQQ"].fillna("") != "")
        | (out["trade_RMABS_QLD"].fillna("") != "")
    ]
    print("CSV:", outp)
    print("행 수(전체 일자):", len(out))
    print("매매 문자열 비어있지 않은 일자:", len(trx))
    if len(trx):
        print("\n체결 발생일만:")
        print(trx.to_string(index=False))


if __name__ == "__main__":
    main()
