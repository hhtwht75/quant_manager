from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "02_DATA"

YEARS = range(2016, 2026)


def load_yahoo_daily(ticker: str) -> pd.DataFrame:
    """Load split-adjusted daily OHLC from Yahoo Finance CSV (no split correction needed)."""
    csv_path = DATA_DIR / "yahoo" / ticker / f"{ticker}_daily.csv"
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close"]]


def load_extended_daily(ticker: str) -> pd.DataFrame:
    """Load extended (1999~) daily Close from yahoo_extended directory.
    Pre-inception dates are synthetic (calibrated leveraged model on QQQ).
    Post-inception dates are real Yahoo Finance data.
    Returns DataFrame with Open/High/Low/Close where Open=High=Low=Close (Close only available).
    """
    csv_path = DATA_DIR / "yahoo_extended" / ticker / f"{ticker}_daily.csv"
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    df.index.name = "Date"
    # Expose Close in OHLC format (O/H/L filled with Close for backtest compatibility)
    close = df["Close"]
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close},
                        index=df.index)


def load_daily_data(leverage_dir: str, ticker: str) -> pd.DataFrame:
    frames = []
    for year in YEARS:
        csv_path = DATA_DIR / leverage_dir / ticker / f"{ticker}_{year}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df = df[df["symbol"] == ticker].copy()
        df["index"] = pd.to_datetime(df["index"], utc=True).dt.tz_convert("America/New_York")
        df.set_index("index", inplace=True)
        frames.append(df)

    raw = pd.concat(frames).sort_index()
    daily = raw.groupby(raw.index.date).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
    )
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "Date"

    daily = _adjust_for_splits(daily)
    return daily


def _adjust_for_splits(daily: pd.DataFrame, gap_threshold: float = 0.30) -> pd.DataFrame:
    """Detect forward splits via large overnight gaps and adjust all prices
    to the latest (post-all-splits) scale."""
    price_cols = ["Open", "High", "Low", "Close"]
    prev_close = daily["Close"].shift(1)
    gap = (daily["Open"] - prev_close) / prev_close

    split_dates = daily.index[gap < -gap_threshold]
    if split_dates.empty:
        return daily

    adj = daily.copy()
    for sd in split_dates:
        idx = daily.index.get_loc(sd)
        if idx == 0:
            continue
        ratio = daily["Close"].iloc[idx - 1] / daily["Open"].iloc[idx]
        adj.loc[adj.index < sd, price_cols] /= ratio

    return adj


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def strategy_buy_and_hold(defensive: pd.DataFrame, initial_capital: float,
                          series_name: str = "BuyHold") -> pd.Series:
    shares = initial_capital / defensive["Close"].iloc[0]
    portfolio = defensive["Close"] * shares
    return portfolio.rename(series_name)


def strategy_switching(
    defensive: pd.DataFrame,
    qld: pd.DataFrame,
    tqqq: pd.DataFrame,
    initial_capital: float,
    shallow_drop: float = -0.10,
    deep_drop: float = -0.20,
    shallow_bounce: float = -0.05,
    deep_bounce: float = -0.10,
    series_name: str = "Switching",
):
    dates = defensive.index
    portfolio_values = []
    switch_events = []

    capital = initial_capital
    def_shares = capital / defensive["Close"].iloc[0]
    tqqq_shares = 0.0
    state = "NORMAL"

    ath = qld["Close"].iloc[0]
    touched_minus_10 = False
    touched_minus_20 = False

    for i, date in enumerate(dates):
        qld_close  = qld["Close"].iloc[i]
        def_close  = defensive["Close"].iloc[i]
        tqqq_close = tqqq["Close"].iloc[i]

        if qld_close > ath:
            ath = qld_close

        drawdown = (qld_close - ath) / ath

        if drawdown <= shallow_drop:
            touched_minus_10 = True
        if drawdown <= deep_drop:
            touched_minus_20 = True

        if drawdown >= 0:
            if state != "NORMAL":
                total_value = def_shares * def_close + tqqq_shares * tqqq_close
                def_shares = total_value / def_close
                tqqq_shares = 0.0
                state = "NORMAL"
                switch_events.append(
                    {"Date": date, "type": "TO_NORMAL", "value": total_value}
                )
            touched_minus_10 = False
            touched_minus_20 = False

        elif state == "NORMAL" and touched_minus_10 and not touched_minus_20 and drawdown >= shallow_bounce:
            total_value = def_shares * def_close
            half_value = total_value / 2
            def_shares = half_value / def_close
            tqqq_shares = half_value / tqqq_close
            state = "HALF_ATTACK"
            switch_events.append(
                {"Date": date, "type": "TO_HALF_ATTACK", "value": total_value}
            )

        elif touched_minus_20 and drawdown >= deep_bounce and state != "FULL_ATTACK":
            total_value = def_shares * def_close + tqqq_shares * tqqq_close
            tqqq_shares = total_value / tqqq_close
            def_shares = 0.0
            state = "FULL_ATTACK"
            switch_events.append(
                {"Date": date, "type": "TO_FULL_ATTACK", "value": total_value}
            )

        portfolio_val = def_shares * def_close + tqqq_shares * tqqq_close
        portfolio_values.append(portfolio_val)

    portfolio_series = pd.Series(portfolio_values, index=dates, name=series_name)
    events_df = pd.DataFrame(switch_events) if switch_events else pd.DataFrame(
        columns=["Date", "type", "value"]
    )
    return portfolio_series, events_df


def strategy_switching_rsi(
    defensive: pd.DataFrame,
    qld: pd.DataFrame,
    tqqq: pd.DataFrame,
    initial_capital: float,
    shallow_drop: float = -0.10,
    deep_drop: float = -0.20,
    shallow_bounce: float = -0.05,
    deep_bounce: float = -0.10,
    rsi_period: int = 14,
    rsi_threshold: float = 30.0,
    series_name: str = "Switching+RSI",
):
    dates = defensive.index
    rsi_series = compute_rsi(qld["Close"], rsi_period)
    portfolio_values = []
    switch_events = []

    capital = initial_capital
    def_shares = capital / defensive["Close"].iloc[0]
    tqqq_shares = 0.0
    state = "NORMAL"

    ath = qld["Close"].iloc[0]
    touched_minus_10 = False
    touched_minus_20 = False
    rsi_valid_at_touch_10 = False
    rsi_valid_at_touch_20 = False

    for i, date in enumerate(dates):
        qld_close  = qld["Close"].iloc[i]
        def_close  = defensive["Close"].iloc[i]
        tqqq_close = tqqq["Close"].iloc[i]
        rsi = rsi_series.iloc[i]

        if qld_close > ath:
            ath = qld_close

        drawdown = (qld_close - ath) / ath

        if drawdown <= shallow_drop and not touched_minus_10:
            touched_minus_10 = True
            rsi_valid_at_touch_10 = rsi < rsi_threshold
        if drawdown <= deep_drop and not touched_minus_20:
            touched_minus_20 = True
            rsi_valid_at_touch_20 = rsi < rsi_threshold

        if drawdown >= 0:
            if state != "NORMAL":
                total_value = def_shares * def_close + tqqq_shares * tqqq_close
                def_shares = total_value / def_close
                tqqq_shares = 0.0
                state = "NORMAL"
                switch_events.append(
                    {"Date": date, "type": "TO_NORMAL", "value": total_value}
                )
            touched_minus_10 = False
            touched_minus_20 = False
            rsi_valid_at_touch_10 = False
            rsi_valid_at_touch_20 = False

        elif (
            state == "NORMAL"
            and touched_minus_10
            and not touched_minus_20
            and drawdown >= shallow_bounce
            and rsi_valid_at_touch_10
            and rsi >= rsi_threshold
        ):
            total_value = def_shares * def_close
            half_value = total_value / 2
            def_shares = half_value / def_close
            tqqq_shares = half_value / tqqq_close
            state = "HALF_ATTACK"
            switch_events.append(
                {"Date": date, "type": "TO_HALF_ATTACK", "value": total_value}
            )

        elif (
            touched_minus_20
            and drawdown >= deep_bounce
            and state != "FULL_ATTACK"
            and rsi_valid_at_touch_20
            and rsi >= rsi_threshold
        ):
            total_value = def_shares * def_close + tqqq_shares * tqqq_close
            tqqq_shares = total_value / tqqq_close
            def_shares = 0.0
            state = "FULL_ATTACK"
            switch_events.append(
                {"Date": date, "type": "TO_FULL_ATTACK", "value": total_value}
            )

        portfolio_val = def_shares * def_close + tqqq_shares * tqqq_close
        portfolio_values.append(portfolio_val)

    portfolio_series = pd.Series(portfolio_values, index=dates, name=series_name)
    events_df = pd.DataFrame(switch_events) if switch_events else pd.DataFrame(
        columns=["Date", "type", "value"]
    )
    return portfolio_series, events_df



def strategy_rsi_ma_based_switching(
    qqq: pd.DataFrame,
    qld: pd.DataFrame,
    tqqq: pd.DataFrame,
    initial_capital: float,
    *,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    ma_days: int = 200,
    ma_break_multiplier: float = 1.03,
    ma_breakdown_multiplier: float = 0.97,
    trailing_stop_pct: float = -0.15,
    series_name: str = "RMABS",
    daily_audit: list | None = None,
    defensive_asset: pd.DataFrame | None = None,
    event_prefix: str = "RMABS",
    qld_ma_down_hold_event_token: str = "QQQ",
    defensive_audit_hold: str | None = None,
    warmup_hold_cash: bool = False,
):
    """RSI_and_Moving_Average_Based_Switching (RMABS).

    시그널은 **QQQ** 종가 기준 RSI14·MA200.

    기본 방어 레인 종가 재평가는 QQQ.**defensive_asset** 를 주면 해당 종가(GLD 등)만 방어 레인 현금화에 쓰이고,
    신호 계산은 QQQ 종가 그대로다.

    ``warmup_hold_cash=True`` 이면 MA200 유효·첫 레짐 결정(첫 이벤트) 전까지 **현금만** 보유하고
    순자산을 ``initial_capital`` 로 유지한다(이자 없음). 기본은 예전과 같이 첫날부터 방어종가로 100% 배분.

    0~3는 기존 RMABS와 동일하되 “QQQ 보유”를 “방어자산 보유”로 읽는다.

    """

    dates = qqq.index
    if not dates.equals(qld.index) or not dates.equals(tqqq.index):
        raise ValueError("RMABS: qqq/qld/tqqq index mismatch")

    sig_close = qqq["Close"].astype(float)

    ext_def = defensive_asset is not None
    if ext_def:
        if not defensive_asset.index.equals(qqq.index):
            raise ValueError("defensive_asset 인덱스가 qqq 와 동일해야 합니다")
        hold_close = defensive_asset["Close"].astype(float)
    else:
        hold_close = sig_close

    audit_def = defensive_audit_hold if defensive_audit_hold is not None else (
        "QQQ" if not ext_def else "DEF"
    )
    evt_init_hold = f"{event_prefix}_INIT_RULE0_{qld_ma_down_hold_event_token}"
    evt_ql_ma = f"{event_prefix}_QLD_TO_{qld_ma_down_hold_event_token}_MA_DOWN"

    rsi_series = compute_rsi(sig_close, rsi_period)
    rsi_vals = rsi_series.values
    rsi_prev = rsi_series.shift(1).values
    ma200 = sig_close.rolling(ma_days, min_periods=ma_days).mean()
    ma_arr = ma200.values

    portfolio_values: list[float] = []
    switch_events: list[dict] = []

    mode = "warmup_QQ"
    rsi_armed_below = False
    rsi_cycle_done = False
    entry_tqq = float("nan")
    peak_since_tqq_entry = float("nan")

    if warmup_hold_cash:
        cash_usd = float(initial_capital)
        q_shares = 0.0
        l_shares = 0.0
        t_shares = 0.0
    else:
        cash_usd = 0.0
        d_open = float(hold_close.iloc[0])
        q_shares = initial_capital / d_open
        l_shares = 0.0
        t_shares = 0.0

    for i, date in enumerate(dates):
        day_trades: list[str] = []

        qq_c = float(sig_close.iloc[i])
        def_c = float(hold_close.iloc[i])
        lc = float(qld["Close"].iloc[i])
        tc = float(tqqq["Close"].iloc[i])

        prev_rsi = float(rsi_prev[i]) if not pd.isna(rsi_prev[i]) else float("nan")
        curr_rsi = float(rsi_vals[i]) if not pd.isna(rsi_vals[i]) else float("nan")

        cb30 = False
        ca30 = False
        if not (pd.isna(prev_rsi) or pd.isna(curr_rsi)):
            cb30 = prev_rsi >= rsi_oversold and curr_rsi < rsi_oversold
            ca30 = prev_rsi <= rsi_oversold and curr_rsi > rsi_oversold

        ma_i = ma_arr[i]
        ma_valid = not pd.isna(ma_i)
        ma_gate_up = ma_valid and qq_c > ma_i * ma_break_multiplier
        ma_deep_down_qld_to_qq = (
            ma_valid
            and qq_c <= ma_i * ma_breakdown_multiplier + 1e-9
        )

        if mode == "warmup_QQ" and warmup_hold_cash:
            pv_before = cash_usd
        else:
            pv_before = q_shares * def_c + l_shares * lc + t_shares * tc

        use_rsi = mode in ("warmup_QQ", "QQQ", "QLD")
        if use_rsi:
            if cb30:
                rsi_armed_below = True
            elif rsi_armed_below and ca30:
                rsi_armed_below = False
                rsi_cycle_done = True

        if mode == "warmup_QQ" and ma_valid:
            mode = "QLD" if qq_c > ma_i else "QQQ"
            if mode == "QLD":
                total = pv_before
                if warmup_hold_cash:
                    cash_usd = 0.0
                q_shares = 0.0
                l_shares = total / lc
                t_shares = 0.0
                ty = f"{event_prefix}_INIT_RULE0_QLD"
                switch_events.append({"Date": date, "type": ty, "value": total})
                day_trades.append(ty)
            else:
                total = pv_before
                if warmup_hold_cash:
                    cash_usd = 0.0
                    q_shares = total / def_c
                    l_shares = 0.0
                    t_shares = 0.0
                switch_events.append(
                    {"Date": date, "type": evt_init_hold, "value": total}
                )
                day_trades.append(evt_init_hold)

        if mode == "TQQ":
            peak_since_tqq_entry = max(peak_since_tqq_entry, tc)
            below_cost = tc < entry_tqq - 1e-9
            trail_hit = peak_since_tqq_entry > 0 and (tc / peak_since_tqq_entry - 1.0) <= trailing_stop_pct

            if below_cost or trail_hit:
                total = t_shares * tc
                t_shares = 0.0
                l_shares = total / lc
                mode = "QLD"
                entry_tqq = float("nan")
                peak_since_tqq_entry = float("nan")
                rsi_armed_below = False
                rsi_cycle_done = False
                evt = (
                    f"{event_prefix}_EXIT_BELOW_COST"
                    if below_cost
                    else f"{event_prefix}_EXIT_TRAIL"
                )
                switch_events.append({"Date": date, "type": evt, "value": total})
                day_trades.append(evt)

        elif mode == "QLD" and l_shares > 0:
            if ma_deep_down_qld_to_qq:
                total = l_shares * lc
                l_shares = 0.0
                q_shares = total / def_c
                mode = "QQQ"
                switch_events.append(
                    {"Date": date, "type": evt_ql_ma, "value": total}
                )
                day_trades.append(evt_ql_ma)
            elif rsi_cycle_done and ma_gate_up and t_shares == 0:
                total_l = l_shares * lc
                l_shares = 0.0
                t_shares = total_l / tc
                entry_tqq = tc
                peak_since_tqq_entry = tc
                rsi_cycle_done = False
                rsi_armed_below = False
                mode = "TQQ"
                ty = f"{event_prefix}_TO_TQQQ"
                switch_events.append({"Date": date, "type": ty, "value": pv_before})
                day_trades.append(ty)

        elif mode == "warmup_QQ":
            pass
        elif mode == "QQQ" and q_shares > 0:
            if rsi_cycle_done and ma_valid and ma_gate_up and t_shares == 0:
                total_q = q_shares * def_c
                q_shares = 0.0
                t_shares = total_q / tc
                entry_tqq = tc
                peak_since_tqq_entry = tc
                rsi_cycle_done = False
                rsi_armed_below = False
                mode = "TQQ"
                ty = f"{event_prefix}_TO_TQQQ"
                switch_events.append({"Date": date, "type": ty, "value": pv_before})
                day_trades.append(ty)

        if mode == "warmup_QQ":
            nav_close = (
                cash_usd
                if warmup_hold_cash
                else q_shares * def_c + l_shares * lc + t_shares * tc
            )
        else:
            nav_close = q_shares * def_c + l_shares * lc + t_shares * tc
        portfolio_values.append(nav_close)
        if daily_audit is not None:
            if mode == "warmup_QQ" and warmup_hold_cash:
                hh = "CASH"
            elif t_shares > 1e-12:
                hh = "TQQQ"
            elif l_shares > 1e-12:
                hh = "QLD"
            elif q_shares > 1e-12:
                hh = audit_def
            else:
                hh = "—"
            daily_audit.append(
                {
                    "Date": pd.Timestamp(date),
                    "hold": hh,
                    "trade": "; ".join(day_trades),
                    "nav": float(nav_close),
                }
            )

    portfolio_series = pd.Series(portfolio_values, index=dates, name=series_name)
    events_df = (
        pd.DataFrame(switch_events)
        if switch_events
        else pd.DataFrame(columns=["Date", "type", "value"])
    )
    return portfolio_series, events_df


def strategy_rsi_ma_based_switching_gold(
    qqq: pd.DataFrame,
    gold: pd.DataFrame,
    qld: pd.DataFrame,
    tqqq: pd.DataFrame,
    initial_capital: float,
    **kwargs,
) -> tuple[pd.Series, pd.DataFrame]:
    """RMABS-GOLD: 신호=QQQ, 방어 레인=금 추종(예 SPDR GLD). 이벤트 접두 ``RMABSG``."""

    kw = dict(kwargs)
    kw.setdefault("series_name", "RMABS-GOLD")
    return strategy_rsi_ma_based_switching(
        qqq,
        qld,
        tqqq,
        initial_capital,
        defensive_asset=gold,
        defensive_audit_hold="GLD",
        event_prefix="RMABSG",
        qld_ma_down_hold_event_token="GLD",
        **kw,
    )


def strategy_rsi_ma_based_switching_qqq_only(
    qqq: pd.DataFrame,
    qld: pd.DataFrame,
    tqqq: pd.DataFrame,
    initial_capital: float,
    *,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    ma_days: int = 200,
    ma_break_multiplier: float = 1.03,
    trailing_stop_pct: float = -0.15,
    series_name: str = "RMABS-QQQ",
    daily_audit: list | None = None,
):
    """RMABS-QQQ: 시그널·방어 자산 모두 **QQQ** 종가.

    0. **시작 QQQ 100%** (전 기간 동일 종목으로 RSI·가격 신호 처리).
    1. RSI14 하향 30 돌파 후 30 상향 재교차 후, **QQQ 종가 > MA200 × ma_break_multiplier** → 전량 **TQQQ**.
    2. **TQQQ → QQQ** (하나 만족 시): 종가 < 진입가 또는 고점 대비 트레일링 **trailing_stop_pct**.

    ``qld``는 인덱스 정렬용으로만 검증한다.

    이벤트 ``type`` 은 접두사 ``RMQQ_`` 로 QLD변형과 구분한다.
    """

    dates = qqq.index
    if not dates.equals(qld.index) or not dates.equals(tqqq.index):
        raise ValueError("RMABS-QQQ: qqq/qld/tqqq index mismatch")

    rsi_series = compute_rsi(qqq["Close"], rsi_period)
    rsi_vals = rsi_series.values
    rsi_prev = rsi_series.shift(1).values
    ma200 = qqq["Close"].astype(float).rolling(ma_days, min_periods=ma_days).mean()
    ma_arr = ma200.values

    portfolio_values: list[float] = []
    switch_events: list[dict] = []

    mode = "QQ"
    rsi_armed_below = False
    rsi_cycle_done = False
    entry_tqq = float("nan")
    peak_since_tqq_entry = float("nan")

    q0 = float(qqq["Close"].iloc[0])
    q_shares = initial_capital / q0
    t_shares = 0.0

    for i, date in enumerate(dates):
        day_trades: list[str] = []

        qq_c = float(qqq["Close"].iloc[i])
        tc = float(tqqq["Close"].iloc[i])

        prev_rsi = float(rsi_prev[i]) if not pd.isna(rsi_prev[i]) else float("nan")
        curr_rsi = float(rsi_vals[i]) if not pd.isna(rsi_vals[i]) else float("nan")

        cb30 = False
        ca30 = False
        if not (pd.isna(prev_rsi) or pd.isna(curr_rsi)):
            cb30 = prev_rsi >= rsi_oversold and curr_rsi < rsi_oversold
            ca30 = prev_rsi <= rsi_oversold and curr_rsi > rsi_oversold

        ma_i = ma_arr[i]
        ma_valid = not pd.isna(ma_i)
        ma_gate_up = ma_valid and qq_c > ma_i * ma_break_multiplier

        pv_before = q_shares * qq_c + t_shares * tc

        if mode == "QQ":
            if cb30:
                rsi_armed_below = True
            elif rsi_armed_below and ca30:
                rsi_armed_below = False
                rsi_cycle_done = True

            if rsi_cycle_done and ma_gate_up and t_shares == 0 and q_shares > 0:
                total_q = q_shares * qq_c
                q_shares = 0.0
                t_shares = total_q / tc
                entry_tqq = tc
                peak_since_tqq_entry = tc
                rsi_cycle_done = False
                rsi_armed_below = False
                mode = "TQQ"
                switch_events.append(
                    {"Date": date, "type": "RMQQ_TO_TQQQ", "value": pv_before}
                )
                day_trades.append("RMQQ_TO_TQQQ")

        elif mode == "TQQ" and t_shares > 0:
            peak_since_tqq_entry = max(peak_since_tqq_entry, tc)
            below_cost = tc < entry_tqq - 1e-9
            trail_hit = peak_since_tqq_entry > 0 and (tc / peak_since_tqq_entry - 1.0) <= trailing_stop_pct

            if below_cost or trail_hit:
                total = t_shares * tc
                q_shares = total / qq_c
                t_shares = 0.0
                mode = "QQ"
                entry_tqq = float("nan")
                peak_since_tqq_entry = float("nan")
                rsi_armed_below = False
                rsi_cycle_done = False
                evt = (
                    "RMQQ_EXIT_BELOW_COST"
                    if below_cost
                    else "RMQQ_EXIT_TRAIL"
                )
                switch_events.append({"Date": date, "type": evt, "value": total})
                day_trades.append(evt)

        nav_close = q_shares * qq_c + t_shares * tc
        portfolio_values.append(nav_close)
        if daily_audit is not None:
            hh = "TQQQ" if t_shares > 1e-12 else "QQQ"
            daily_audit.append(
                {
                    "Date": pd.Timestamp(date),
                    "hold": hh,
                    "trade": "; ".join(day_trades),
                    "nav": float(nav_close),
                }
            )

    portfolio_series = pd.Series(portfolio_values, index=dates, name=series_name)
    events_df = (
        pd.DataFrame(switch_events)
        if switch_events
        else pd.DataFrame(columns=["Date", "type", "value"])
    )
    return portfolio_series, events_df


def strategy_switching_stoploss(
    qqq: pd.DataFrame,
    qld: pd.DataFrame,
    tqqq: pd.DataFrame,
    initial_capital: float,
    shallow_drop: float = -0.10,
    deep_drop: float = -0.20,
    shallow_bounce: float = -0.05,
    deep_bounce: float = -0.10,
):
    """Strategy 2 + stop-loss:
    HALF_ATTACK exits back to NORMAL when drawdown falls to shallow_drop again.
    FULL_ATTACK exits back to NORMAL when drawdown falls to deep_drop again.
    Touch flags are preserved after stop-loss, allowing re-entry on next bounce.
    """
    dates = qqq.index
    portfolio_values = []
    switch_events = []

    qqq_shares = initial_capital / qqq["Close"].iloc[0]
    tqqq_shares = 0.0
    state = "NORMAL"
    ath = qld["Close"].iloc[0]
    touched_10 = False
    touched_20 = False

    for i, date in enumerate(dates):
        qld_close = qld["Close"].iloc[i]
        qqq_close = qqq["Close"].iloc[i]
        tqqq_close = tqqq["Close"].iloc[i]

        if qld_close > ath:
            ath = qld_close
        drawdown = (qld_close - ath) / ath

        if drawdown <= shallow_drop:
            touched_10 = True
        if drawdown <= deep_drop:
            touched_20 = True

        total_value = qqq_shares * qqq_close + tqqq_shares * tqqq_close

        if drawdown >= 0:
            if state != "NORMAL":
                qqq_shares = total_value / qqq_close
                tqqq_shares = 0.0
                state = "NORMAL"
                switch_events.append({"Date": date, "type": "TO_NORMAL", "value": total_value})
            touched_10 = False
            touched_20 = False

        elif state == "HALF_ATTACK" and drawdown <= shallow_drop:
            # Stop-loss: fell back to -10%
            qqq_shares = total_value / qqq_close
            tqqq_shares = 0.0
            state = "NORMAL"
            switch_events.append({"Date": date, "type": "STOPLOSS_HALF", "value": total_value})

        elif state == "FULL_ATTACK" and drawdown <= deep_drop:
            # Stop-loss: fell back to -20%
            qqq_shares = total_value / qqq_close
            tqqq_shares = 0.0
            state = "NORMAL"
            switch_events.append({"Date": date, "type": "STOPLOSS_FULL", "value": total_value})

        elif state == "NORMAL" and touched_10 and not touched_20 and drawdown >= shallow_bounce:
            half = total_value / 2
            qqq_shares = half / qqq_close
            tqqq_shares = half / tqqq_close
            state = "HALF_ATTACK"
            switch_events.append({"Date": date, "type": "TO_HALF_ATTACK", "value": total_value})

        elif touched_20 and drawdown >= deep_bounce and state != "FULL_ATTACK":
            tqqq_shares = total_value / tqqq_close
            qqq_shares = 0.0
            state = "FULL_ATTACK"
            switch_events.append({"Date": date, "type": "TO_FULL_ATTACK", "value": total_value})

        portfolio_values.append(qqq_shares * qqq_close + tqqq_shares * tqqq_close)

    portfolio_series = pd.Series(portfolio_values, index=dates, name="StopLoss")
    events_df = pd.DataFrame(switch_events) if switch_events else pd.DataFrame(
        columns=["Date", "type", "value"]
    )
    return portfolio_series, events_df


def strategy_s4_trailing(
    defensive: pd.DataFrame,
    qld: pd.DataFrame,
    tqqq: pd.DataFrame,
    initial_capital: float,
    shallow_drop: float = -0.10,
    deep_drop: float = -0.20,
    shallow_bounce: float = -0.05,
    deep_bounce: float = -0.10,
    half_stop: float = -0.15,         # QLD -15% while HALF_ATTACK → exit (flags kept)
    trailing_stop_pct: float = -0.15,  # trailing stop from TQQQ peak in TRAILING mode
    half_frac: float = 0.50,           # fraction converted to TQQQ on HALF_ATTACK entry
    full_frac: float = 1.00,           # fraction converted to TQQQ on FULL_ATTACK entry
    series_name: str = "S4 TrailExit",
):
    """Strategy 4: S2-identical entry (QLD signal), but exits differently.

    Exit changes vs S2:
    A) Instead of exiting to QQQ when QLD recovers its ATH, enters TRAILING mode:
       - Tracks TQQQ from the ATH-recovery moment (trail_entry price)
       - TRAIL_FLOOR: TQQQ drops below trail_entry → exit (lock in attack gains)
       - TRAIL_EXIT : TQQQ drops 10% from its peak in TRAILING → exit
    B) HALF_ATTACK stop at -15%: QLD reaches -15% while in HALF_ATTACK
       → exit to QQQ, but KEEP touched flags (re-entry still possible)

    States: NORMAL → HALF_ATTACK / FULL_ATTACK → TRAILING → NORMAL
    """
    dates = defensive.index
    portfolio_values = []
    switch_events = []

    def_shares  = initial_capital / defensive["Close"].iloc[0]
    tqqq_shares = 0.0
    state       = "NORMAL"

    ath        = qld["Close"].iloc[0]
    touched_10 = False
    touched_20 = False

    tqqq_trail_peak  = 0.0   # highest TQQQ price since entering TRAILING
    tqqq_trail_entry = 0.0   # TQQQ price at moment of entering TRAILING (floor)

    for i, date in enumerate(dates):
        qld_close  = qld["Close"].iloc[i]
        def_close  = defensive["Close"].iloc[i]
        tqqq_close = tqqq["Close"].iloc[i]

        if qld_close > ath:
            ath = qld_close
        dd = (qld_close - ath) / ath

        if dd <= shallow_drop:
            touched_10 = True
        if dd <= deep_drop:
            touched_20 = True

        total_value = def_shares * def_close + tqqq_shares * tqqq_close

        # ── TRAILING state ─────────────────────────────────────────────────────
        if state == "TRAILING":
            if tqqq_close > tqqq_trail_peak:
                tqqq_trail_peak = tqqq_close

            # Rule 1: Floor — TQQQ below trailing entry price → lock in gains
            if tqqq_close <= tqqq_trail_entry:
                def_shares  = total_value / def_close
                tqqq_shares = 0.0
                state       = "NORMAL"
                touched_10  = False
                touched_20  = False
                switch_events.append({"Date": date, "type": "TRAIL_FLOOR", "value": total_value})

            # Trailing stop — TQQQ 10% below its peak since TRAILING entry
            elif tqqq_close <= tqqq_trail_peak * (1.0 + trailing_stop_pct):
                def_shares  = total_value / def_close
                tqqq_shares = 0.0
                state       = "NORMAL"
                touched_10  = False
                touched_20  = False
                switch_events.append({"Date": date, "type": "TRAIL_EXIT", "value": total_value})

        # ── HALF / FULL ATTACK ─────────────────────────────────────────────────
        elif state in ("HALF_ATTACK", "FULL_ATTACK"):
            if dd >= 0:
                # QLD ATH recovery → enter TRAILING instead of immediate QQQ exit
                tqqq_trail_peak  = tqqq_close
                tqqq_trail_entry = tqqq_close   # floor = current TQQQ price
                state = "TRAILING"
                switch_events.append({"Date": date, "type": "TO_TRAILING", "value": total_value})

            elif state == "HALF_ATTACK" and dd <= half_stop:
                # Rule 2: QLD hit -15% while HALF → exit, but KEEP touch flags
                def_shares  = total_value / def_close
                tqqq_shares = 0.0
                state       = "NORMAL"
                # touched_10 / touched_20 intentionally NOT reset here
                switch_events.append({"Date": date, "type": "HALF_STOP", "value": total_value})

            elif state == "HALF_ATTACK" and touched_20 and dd >= deep_bounce:
                # Upgrade HALF → FULL
                tqqq_shares = total_value * full_frac / tqqq_close
                def_shares  = total_value * (1.0 - full_frac) / def_close
                state       = "FULL_ATTACK"
                switch_events.append({"Date": date, "type": "TO_FULL_ATTACK", "value": total_value})

        # ── NORMAL ─────────────────────────────────────────────────────────────
        elif state == "NORMAL":
            if dd >= 0:
                touched_10 = False
                touched_20 = False
            elif touched_10 and not touched_20 and dd >= shallow_bounce:
                tqqq_shares = total_value * half_frac / tqqq_close
                def_shares  = total_value * (1.0 - half_frac) / def_close
                state       = "HALF_ATTACK"
                switch_events.append({"Date": date, "type": "TO_HALF_ATTACK", "value": total_value})
            elif touched_20 and dd >= deep_bounce:
                tqqq_shares = total_value * full_frac / tqqq_close
                def_shares  = total_value * (1.0 - full_frac) / def_close
                state       = "FULL_ATTACK"
                switch_events.append({"Date": date, "type": "TO_FULL_ATTACK", "value": total_value})

        portfolio_values.append(def_shares * def_close + tqqq_shares * tqqq_close)

    portfolio_series = pd.Series(portfolio_values, index=dates, name=series_name)
    events_df = pd.DataFrame(switch_events) if switch_events else pd.DataFrame(
        columns=["Date", "type", "value"]
    )
    return portfolio_series, events_df


def strategy_qld_touch_bounce_full(
    defensive: pd.DataFrame,
    qld: pd.DataFrame,
    tqqq: pd.DataFrame,
    initial_capital: float,
    touch_drop: float = -0.14,       # QLD ATH 대비 이 하락을 한 번 이상 터치해야 함
    bounce_dd: float = -0.07,       # 위 조건 후 dd가 이 값 이상(덜 깊게) 반등 시 진입
    stop_dd: float = -0.14,         # 보유 중 QLD가 다시 이 깊이 이하 → 손절 → QQQ
    trailing_stop_pct: float = -0.15,
    series_name: str = "QLD-14/-7 Full",
) -> tuple:
    """QLD ATH 스윙: −14% 터치 후 −7%까지 반등 시 포트 100% TQQQ.

    • 보유 중 QLD가 다시 ATH 대비 −14% 이하 → QQQ로 손절 (터치 플래그 유지, 반등 시 재진입 가능).
    • QLD가 전고점 돌파(dd≥0) → TRAILING: TQQQ 진입가 이하 또는 고점 대비 trail % 하락 시 청산 후
      NORMAL + 터치 플래그 리셋.

    States: NORMAL → IN_TQQQ → TRAILING → NORMAL
    """
    dates = defensive.index
    portfolio_values = []
    switch_events = []

    def_shares  = initial_capital / defensive["Close"].iloc[0]
    tqqq_shares = 0.0
    state       = "NORMAL"

    ath = qld["Close"].iloc[0]
    touched = False

    tqqq_trail_peak  = 0.0
    tqqq_trail_entry = 0.0

    for i, date in enumerate(dates):
        qld_close  = qld["Close"].iloc[i]
        def_close  = defensive["Close"].iloc[i]
        tqqq_close = tqqq["Close"].iloc[i]

        if qld_close > ath:
            ath = qld_close
        dd = (qld_close - ath) / ath

        if dd <= touch_drop:
            touched = True

        total_value = def_shares * def_close + tqqq_shares * tqqq_close

        if state == "TRAILING":
            if tqqq_close > tqqq_trail_peak:
                tqqq_trail_peak = tqqq_close
            if tqqq_close <= tqqq_trail_entry:
                def_shares  = total_value / def_close
                tqqq_shares = 0.0
                state       = "NORMAL"
                touched     = False
                switch_events.append({"Date": date, "type": "TRAIL_FLOOR", "value": total_value})
            elif tqqq_close <= tqqq_trail_peak * (1.0 + trailing_stop_pct):
                def_shares  = total_value / def_close
                tqqq_shares = 0.0
                state       = "NORMAL"
                touched     = False
                switch_events.append({"Date": date, "type": "TRAIL_EXIT", "value": total_value})

        elif state == "IN_TQQQ":
            if dd >= 0:
                tqqq_trail_peak  = tqqq_close
                tqqq_trail_entry = tqqq_close
                state = "TRAILING"
                switch_events.append({"Date": date, "type": "TO_TRAILING", "value": total_value})
            elif dd <= stop_dd:
                def_shares  = total_value / def_close
                tqqq_shares = 0.0
                state       = "NORMAL"
                switch_events.append({"Date": date, "type": "STOP_AT_TOUCH", "value": total_value})

        elif state == "NORMAL":
            if dd >= 0:
                touched = False
            elif touched and dd >= bounce_dd:
                tqqq_shares = total_value / tqqq_close
                def_shares  = 0.0
                state       = "IN_TQQQ"
                switch_events.append({"Date": date, "type": "TO_FULL_TQQQ", "value": total_value})

        portfolio_values.append(def_shares * def_close + tqqq_shares * tqqq_close)

    portfolio_series = pd.Series(portfolio_values, index=dates, name=series_name)
    events_df = pd.DataFrame(switch_events) if switch_events else pd.DataFrame(
        columns=["Date", "type", "value"]
    )
    return portfolio_series, events_df


def strategy_s5_2(
    defensive: pd.DataFrame,
    qld: pd.DataFrame,
    tqqq: pd.DataFrame,
    initial_capital: float,
    beta: float = 0.50,
    min_drop: float = 0.11,
    drop_l1: float = -0.10,
    drop_l2: float = -0.15,
    drop_l3: float = -0.20,
    bounce_l1: float = -0.05,
    bounce_l2: float = -0.075,
    bounce_l3: float = -0.10,
    frac_l1: float = 0.25,
    frac_l2: float = 0.50,
    frac_l3: float = 1.00,
    l1_stop: float = -0.15,
    trailing_stop_pct: float = -0.15,
    series_name: str = "S5-2",
) -> tuple:
    """S5-2: S4-2와 동일한 티어·손절·트레일 상태머신 + NORMAL 진입만 S5식 연속 게이트.

    • ATH·터치 플래그·L1/L2/L3·트레일: strategy_tiered 와 동일.
    • NORMAL → L1/L2/L3 신규 진입 시에만 추가 조건:
        swing_drop = (ATH − swing_low)/ATH > min_drop   (기본 11%: −10%만 찍은 얕은 스윙은 제외)
        rebound    = (ATH − QLD)/ATH < beta × swing_drop,  QLD < ATH
      (업그레이드 L1→L2→L3 는 S4-2 와 동일, 게이트 없음)
    • TRAILING 청산 후 swing_low 를 현재 QLD 로 리셋 (사이클 정리).
    """
    dates = defensive.index
    dc = defensive["Close"].values
    qc = qld["Close"].values
    tc = tqqq["Close"].values

    def_shares = initial_capital / dc[0]
    tqqq_shares = 0.0
    state = "NORMAL"

    ath = qc[0]
    swing_low = qc[0]
    touched_10 = touched_15 = touched_20 = False
    tqqq_trail_peak = 0.0
    tqqq_trail_entry = 0.0

    portfolio = []
    events = []

    for i in range(len(dates)):
        date = dates[i]
        dcp = dc[i]
        qcp = qc[i]
        tcp = tc[i]

        if qcp > ath:
            ath = qcp
            swing_low = qcp
        if qcp < swing_low:
            swing_low = qcp

        dd = qcp / ath - 1.0
        swing_drop = (ath - swing_low) / ath
        rebound_s5 = (ath - qcp) / ath
        s5_gate_norm = (
            swing_drop > min_drop
            and rebound_s5 < beta * swing_drop
            and qcp < ath
        )

        if dd <= drop_l1:
            touched_10 = True
        if dd <= drop_l2:
            touched_15 = True
        if dd <= drop_l3:
            touched_20 = True

        tv = def_shares * dcp + tqqq_shares * tcp

        if state == "TRAILING":
            if tcp > tqqq_trail_peak:
                tqqq_trail_peak = tcp
            if tcp <= tqqq_trail_entry:
                def_shares = tv / dcp
                tqqq_shares = 0.0
                state = "NORMAL"
                touched_10 = touched_15 = touched_20 = False
                swing_low = qcp
                events.append({"Date": date, "type": "TRAIL_FLOOR", "value": tv})
            elif tcp <= tqqq_trail_peak * (1.0 + trailing_stop_pct):
                def_shares = tv / dcp
                tqqq_shares = 0.0
                state = "NORMAL"
                touched_10 = touched_15 = touched_20 = False
                swing_low = qcp
                events.append({"Date": date, "type": "TRAIL_EXIT", "value": tv})

        elif state == "L1":
            if dd >= 0:
                tqqq_trail_peak = tcp
                tqqq_trail_entry = tcp
                state = "TRAILING"
                events.append({"Date": date, "type": "TO_TRAILING_from_L1", "value": tv})
            elif dd <= l1_stop:
                def_shares = tv / dcp
                tqqq_shares = 0.0
                state = "NORMAL"
                events.append({"Date": date, "type": "L1_STOP", "value": tv})
            elif touched_20 and dd >= bounce_l3:
                tqqq_shares = tv * frac_l3 / tcp
                def_shares = tv * (1.0 - frac_l3) / dcp
                state = "L3"
                events.append({"Date": date, "type": "TO_L3_from_L1", "value": tv})
            elif touched_15 and not touched_20 and dd >= bounce_l2:
                tqqq_shares = tv * frac_l2 / tcp
                def_shares = tv * (1.0 - frac_l2) / dcp
                state = "L2"
                events.append({"Date": date, "type": "TO_L2_from_L1", "value": tv})

        elif state == "L2":
            if dd >= 0:
                tqqq_trail_peak = tcp
                tqqq_trail_entry = tcp
                state = "TRAILING"
                events.append({"Date": date, "type": "TO_TRAILING_from_L2", "value": tv})
            elif dd < drop_l3:
                def_shares = tv / dcp
                tqqq_shares = 0.0
                state = "NORMAL"
                events.append({"Date": date, "type": "L2_STOP", "value": tv})
            elif touched_20 and dd >= bounce_l3:
                tqqq_shares = tv * frac_l3 / tcp
                def_shares = tv * (1.0 - frac_l3) / dcp
                state = "L3"
                events.append({"Date": date, "type": "TO_L3_from_L2", "value": tv})

        elif state == "L3":
            if dd >= 0:
                tqqq_trail_peak = tcp
                tqqq_trail_entry = tcp
                state = "TRAILING"
                events.append({"Date": date, "type": "TO_TRAILING_from_L3", "value": tv})

        elif state == "NORMAL":
            if dd >= 0:
                touched_10 = touched_15 = touched_20 = False
                swing_low = qcp

            elif touched_20 and dd >= bounce_l3 and s5_gate_norm:
                tqqq_shares = tv * frac_l3 / tcp
                def_shares = tv * (1.0 - frac_l3) / dcp
                state = "L3"
                events.append({"Date": date, "type": "TO_L3_from_NORMAL", "value": tv})

            elif touched_15 and not touched_20 and dd >= bounce_l2 and s5_gate_norm:
                tqqq_shares = tv * frac_l2 / tcp
                def_shares = tv * (1.0 - frac_l2) / dcp
                state = "L2"
                events.append({"Date": date, "type": "TO_L2_from_NORMAL", "value": tv})

            elif touched_10 and not touched_15 and dd >= bounce_l1 and s5_gate_norm:
                tqqq_shares = tv * frac_l1 / tcp
                def_shares = tv * (1.0 - frac_l1) / dcp
                state = "L1"
                events.append({"Date": date, "type": "TO_L1_from_NORMAL", "value": tv})

        portfolio.append(def_shares * dcp + tqqq_shares * tcp)

    port_series = pd.Series(portfolio, index=dates, name=series_name)
    ev_df = (
        pd.DataFrame(events)
        if events
        else pd.DataFrame(columns=["Date", "type", "value"])
    )
    return port_series, ev_df


def _s5_tiered_attack_fraction(
    drop: float,
    min_drop: float,
    mid_depth: float,
    deep_depth: float,
    frac_shallow: float,
    frac_mid: float,
    frac_full: float,
) -> float:
    """Piecewise TQQQ fraction vs swing depth (S4-2 style): shallow → small, deep → full."""
    if drop <= min_drop:
        return 0.0
    if drop <= mid_depth:
        return frac_shallow
    if drop <= deep_depth:
        return frac_mid
    return frac_full


def _s5_exp_attack_fraction(
    drop: float,
    min_drop: float,
    max_drop: float,
    frac_lo: float = 0.25,
    frac_hi: float = 1.0,
    exp_base: float = 2.0,
) -> float:
    """Map drop to TQQQ fraction; t∈[0,1] from (min_drop,max_drop).

    frac(t) = frac_lo + (frac_hi − frac_lo) × (b^t − 1) / (b − 1),  b = exp_base > 1.
    t=0 → frac_lo, t=1 → frac_hi.  b=4 with 0.25/1.0 matches legacy 0.25×4^t.
    """
    if max_drop <= min_drop:
        raise ValueError("max_drop must be > min_drop for exp sizing")
    if exp_base <= 1.0:
        raise ValueError("exp_base must be > 1")
    if drop <= min_drop:
        return 0.0
    t = (drop - min_drop) / (max_drop - min_drop)
    if t > 1.0:
        t = 1.0
    scale = (exp_base**t - 1.0) / (exp_base - 1.0)
    return float(frac_lo + (frac_hi - frac_lo) * scale)


def _s5_s4_aligned_attack_stop_hit(
    entry_atk_frac: float,
    dd_q: float,
    *,
    l1_stop_dd: float,
    l2_stop_dd: float,
    l2_strict_ref: float,
    frac_l1_boundary: float = 0.25,
    eps: float = 1e-9,
) -> bool:
    """S4-2 공격 구간 손절과 동일한 띠: 진입 시 TQQQ 비중으로 L1/L2/L3 대응.

    • entry_atk_frac ≤ frac_l1_boundary (25% 띠) → dd_q ≤ l1_stop_dd
    • 그 외 ~100% 미만 → L2: l2_stop_dd == l2_strict_ref 이면 dd_q < ref, 아니면 dd_q ≤ l2_stop_dd
    • entry_atk_frac ≥ 1−eps → 손절 없음 (S4-2의 L3)
    dd_q = QLD/ATH − 1 (음수).
    """
    if entry_atk_frac >= 1.0 - eps:
        return False
    if entry_atk_frac <= frac_l1_boundary + eps:
        return dd_q <= l1_stop_dd
    if abs(l2_stop_dd - l2_strict_ref) < 1e-14:
        return dd_q < l2_strict_ref
    return dd_q <= l2_stop_dd


def strategy_s5(
    defensive: pd.DataFrame,
    qld: pd.DataFrame,
    tqqq: pd.DataFrame,
    initial_capital: float,
    beta: float = 0.50,          # entry: rebound < beta × drop
    max_drop: float = 0.20,      # linear: cap; exp: depth where frac → exp_frac_hi
    stop_factor: float = 0.75,   # linear + use_stop_loss: rebound > stop_factor × drop_entry
    trailing_stop_pct: float = -0.15,  # TRAILING mode: exit if TQQQ < peak × (1+trail)
    min_drop: float = 0.0,       # minimum swing drop (ath−trough)/ath to allow entry
    use_stop_loss: bool = True,  # True: exit ATTACK on stop (see attack_stop_mode)
    # ATTACK 손절: drop_deepen=스윙 저점 추가 악화(기본); s4_dd=QLD ATH 대비 dd 띠(S4-2); both=둘 중 하나
    attack_stop_mode: str = "drop_deepen",  # "drop_deepen" | "s4_dd" | "both"
    s4_l1_stop_dd: float = -0.15,
    s4_l2_stop_dd: float = -0.20,
    s4_l2_strict_ref: float = -0.20,  # s4_l2_stop_dd 와 같으면 dd < ref (S4-2 L2와 동일)
    s4_frac_l1_boundary: float = 0.25,
    # exp/tiered + drop_deepen(또는 both): None이면 기존 drop>drop_entry. 지정 시 ATH dd가
    #   진입 스윙 깊이(drop_entry)의 mult배 이상 악화될 때만 손절 (예 mult=1.5, 스윙 12% → dd<=-18%)
    entry_depth_stop_mult: float | None = None,
    position_mode: str = "exp",  # "exp" | "linear" | "tiered"
    exp_frac_lo: float = 0.25,   # TQQQ fraction at min_drop (exp mode)
    exp_frac_hi: float = 1.00,   # TQQQ fraction at max_drop (exp mode)
    exp_base: float = 2.0,       # exp mode: b in (b^t−1)/(b−1); 2=완만, 4=구버전과 동일 곡선
    tier_mid_drop: float = 0.15,
    tier_deep_drop: float = 0.20,
    tier_frac_shallow: float = 0.25,
    tier_frac_mid: float = 0.50,
    tier_frac_full: float = 1.00,
    series_name: str = "S5",
    daily_tqqq_weight_out: list | None = None,
) -> tuple:
    """Strategy 5: QLD swing-based entry; continuous entry, linear / exp / tiered sizing.

    Entry logic
    -----------
    • Track QLD ATH and swing_low; drop = (ath−swing_low)/ath, rebound = (ath−qld)/ath.
    • Enter ATTACK when: rebound < beta × drop  AND  drop > min_drop  AND  qld < ath
    • Position size at entry:
        - position_mode="exp": t=clip((drop−min)/(max−min),0,1);
          atk_frac = exp_frac_lo + (exp_frac_hi−exp_frac_lo)×(exp_base^t−1)/(exp_base−1)
        - position_mode="linear": atk_frac = min(drop / max_drop, 1.0)
        - position_mode="tiered": piecewise (S4-2-style steps)

    Exit (ATTACK) when use_stop_loss
    --------------------------------
    attack_stop_mode="drop_deepen" (기본):
    • position_mode="linear": rebound > stop_factor × drop_entry
    • position_mode="exp" or "tiered": (기본) swing 저점이 진입보다 깊어짐 — drop > drop_entry
      entry_depth_stop_mult 가 주어지면 위 대신: dd_q <= −mult×drop_entry (진입 시 스윙 깊이의 배수)

    attack_stop_mode="s4_dd": S4-2와 같은 **현재 QLD의 ATH 대비 dd** 임계.
    진입 시 확정된 atk_frac으로 띠 구분: ≤25%→l1_stop, (25%,100%)→l2 규칙, 100%→중간 SL 없음.

    attack_stop_mode="both": 위 두 조건 중 하나.

    Then ATH → TRAILING; TRAIL_FLOOR / TRAIL_EXIT → NORMAL; swing_low reset on exit.

    States: NORMAL ↔ ATTACK → TRAILING → NORMAL
    """
    _asm = attack_stop_mode.lower()
    if _asm not in ("drop_deepen", "s4_dd", "both"):
        raise ValueError(
            "attack_stop_mode must be 'drop_deepen', 's4_dd', or 'both', "
            f"got {attack_stop_mode!r}"
        )

    dates = defensive.index
    portfolio_values = []
    switch_events    = []

    # Portfolio initialisation
    def_shares  = initial_capital / defensive["Close"].iloc[0]
    tqqq_shares = 0.0
    state       = "NORMAL"
    atk_frac    = 0.0   # fraction in TQQQ at entry
    entry_atk_frac = 0.0

    # QLD tracking
    ath        = qld["Close"].iloc[0]
    swing_low  = qld["Close"].iloc[0]
    drop_entry = 0.0   # drop at time of ATTACK entry (for stop-loss calc)

    # TRAILING tracking
    tqqq_trail_peak  = 0.0
    tqqq_trail_entry = 0.0

    for i, date in enumerate(dates):
        qld_close  = qld["Close"].iloc[i]
        def_close  = defensive["Close"].iloc[i]
        tqqq_close = tqqq["Close"].iloc[i]

        # ── Update ATH & swing_low ──────────────────────────────────────────
        if qld_close > ath:
            ath       = qld_close
            swing_low = qld_close   # reset cycle at new ATH

        if qld_close < swing_low:
            swing_low = qld_close

        drop    = (ath - swing_low) / ath             # >= 0
        rebound = max(0.0, (ath - qld_close) / ath)  # >= 0, <= drop

        total_value = def_shares * def_close + tqqq_shares * tqqq_close

        # ── TRAILING ───────────────────────────────────────────────────────
        if state == "TRAILING":
            if tqqq_close > tqqq_trail_peak:
                tqqq_trail_peak = tqqq_close

            exited = False
            if tqqq_close <= tqqq_trail_entry:
                switch_events.append({"Date": date, "type": "TRAIL_FLOOR", "value": total_value})
                exited = True
            elif tqqq_close <= tqqq_trail_peak * (1.0 + trailing_stop_pct):
                switch_events.append({"Date": date, "type": "TRAIL_EXIT", "value": total_value})
                exited = True

            if exited:
                def_shares  = total_value / def_close
                tqqq_shares = 0.0
                state       = "NORMAL"
                swing_low   = qld_close  # reset cycle → prevents immediate re-entry

        # ── ATTACK ─────────────────────────────────────────────────────────
        elif state == "ATTACK":
            # ATH recovery → enter TRAILING
            if qld_close >= ath:
                tqqq_trail_entry = tqqq_close
                tqqq_trail_peak  = tqqq_close
                state = "TRAILING"
                switch_events.append({"Date": date, "type": "TO_TRAILING", "value": total_value})

            # Stop loss
            elif use_stop_loss:
                pm = position_mode.lower()
                mode = attack_stop_mode.lower()
                dd_q = qld_close / ath - 1.0
                sl_hit = False
                reasons: list[str] = []

                if mode in ("drop_deepen", "both"):
                    if pm == "linear":
                        if rebound > stop_factor * drop_entry:
                            sl_hit = True
                            reasons.append("rebound")
                    elif pm in ("exp", "tiered"):
                        if entry_depth_stop_mult is not None:
                            thr = -float(entry_depth_stop_mult) * drop_entry
                            if dd_q <= thr - 1e-15:
                                sl_hit = True
                                reasons.append(
                                    f"entry_depth_x{entry_depth_stop_mult}"
                                )
                        elif drop > drop_entry + 1e-12:
                            sl_hit = True
                            reasons.append("drop_deepen")
                    else:
                        pass

                if mode in ("s4_dd", "both"):
                    if _s5_s4_aligned_attack_stop_hit(
                        entry_atk_frac,
                        dd_q,
                        l1_stop_dd=s4_l1_stop_dd,
                        l2_stop_dd=s4_l2_stop_dd,
                        l2_strict_ref=s4_l2_strict_ref,
                        frac_l1_boundary=s4_frac_l1_boundary,
                    ):
                        sl_hit = True
                        reasons.append("s4_dd")

                if sl_hit:
                    def_shares  = total_value / def_close
                    tqqq_shares = 0.0
                    state       = "NORMAL"
                    swing_low   = qld_close
                    switch_events.append({
                        "Date": date, "type": "STOP_LOSS", "value": total_value,
                        "reason": "+".join(reasons) if reasons else "stop",
                    })

        # ── NORMAL ─────────────────────────────────────────────────────────
        elif state == "NORMAL":
            # Entry: significant dip + partial recovery within [beta × drop]
            if drop > min_drop and rebound < beta * drop and qld_close < ath:
                pm = position_mode.lower()
                if pm == "linear":
                    atk_frac = min(drop / max_drop, 1.0)
                elif pm == "exp":
                    atk_frac = _s5_exp_attack_fraction(
                        drop, min_drop, max_drop, exp_frac_lo, exp_frac_hi, exp_base,
                    )
                elif pm == "tiered":
                    atk_frac = _s5_tiered_attack_fraction(
                        drop,
                        min_drop,
                        tier_mid_drop,
                        tier_deep_drop,
                        tier_frac_shallow,
                        tier_frac_mid,
                        tier_frac_full,
                    )
                else:
                    raise ValueError(
                        "position_mode must be 'linear', 'exp', or 'tiered', "
                        f"got {position_mode!r}"
                    )
                tqqq_shares = total_value * atk_frac    / tqqq_close
                def_shares  = total_value * (1.0 - atk_frac) / def_close
                state       = "ATTACK"
                drop_entry  = drop
                entry_atk_frac = atk_frac
                switch_events.append({"Date": date, "type": "TO_ATTACK",
                                       "value": total_value,
                                       "drop_pct": round(drop * 100, 2),
                                       "rebound_pct": round(rebound * 100, 2),
                                       "atk_frac": round(atk_frac, 4),
                                       "sizing": pm})

        tv_end = def_shares * def_close + tqqq_shares * tqqq_close
        if daily_tqqq_weight_out is not None:
            daily_tqqq_weight_out.append(
                (tqqq_shares * tqqq_close) / tv_end if tv_end > 1e-18 else 0.0
            )
        portfolio_values.append(tv_end)

    portfolio_series = pd.Series(portfolio_values, index=dates, name=series_name)
    events_df = pd.DataFrame(switch_events) if switch_events else pd.DataFrame(
        columns=["Date", "type", "value"]
    )
    return portfolio_series, events_df


def compute_statistics(portfolio: pd.Series, label: str, switch_count: int = 0) -> dict:
    total_return = portfolio.iloc[-1] / portfolio.iloc[0] - 1
    n_days = (portfolio.index[-1] - portfolio.index[0]).days
    n_years = n_days / 365.25
    cagr = (portfolio.iloc[-1] / portfolio.iloc[0]) ** (1 / n_years) - 1

    daily_returns = portfolio.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    risk_free_daily = (1 + 0.04) ** (1 / 252) - 1
    sharpe = (daily_returns.mean() - risk_free_daily) / daily_returns.std() * np.sqrt(252)

    rolling_max = portfolio.cummax()
    drawdown = (portfolio - rolling_max) / rolling_max
    mdd = drawdown.min()

    stats = {
        "전략": label,
        "최종 자산": f"${portfolio.iloc[-1]:,.0f}",
        "총 수익률": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}",
        "MDD": f"{mdd:.2%}",
        "연간 변동성": f"{volatility:.2%}",
        "샤프 비율": f"{sharpe:.2f}",
    }
    if switch_count > 0:
        stats["스위칭 횟수"] = switch_count
    return stats


def plot_results(
    bh_portfolio: pd.Series,
    sw_portfolio: pd.Series,
    sw_rsi_portfolio: pd.Series,
    sw_sl_portfolio: pd.Series,
    events_df: pd.DataFrame,
    events_rsi_df: pd.DataFrame,
    events_sl_df: pd.DataFrame,
    qld: pd.DataFrame,
    out_path=None,
):
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(16, 13), height_ratios=[3, 1, 1], sharex=True
    )
    fig.suptitle(
        "S1: QQQ Hold  |  S2: Switching  |  S3: Switching+RSI  |  S4: Switching+StopLoss",
        fontsize=13, fontweight="bold",
    )

    ax1.plot(bh_portfolio.index, bh_portfolio.values, label="S1: QQQ Buy & Hold", linewidth=1.2, color="#2196F3")
    ax1.plot(sw_portfolio.index, sw_portfolio.values, label="S2: Switching", linewidth=1.2, color="#FF5722")
    ax1.plot(sw_rsi_portfolio.index, sw_rsi_portfolio.values, label="S3: Switching+RSI", linewidth=1.2, color="#9C27B0")
    ax1.plot(sw_sl_portfolio.index, sw_sl_portfolio.values, label="S4: Switching+StopLoss", linewidth=1.2, color="#4CAF50")

    for evdf in [events_df, events_rsi_df, events_sl_df]:
        if evdf.empty:
            continue
        half  = evdf[evdf["type"] == "TO_HALF_ATTACK"]
        full  = evdf[evdf["type"] == "TO_FULL_ATTACK"]
        norm  = evdf[evdf["type"] == "TO_NORMAL"]
        sl_h  = evdf[evdf["type"] == "STOPLOSS_HALF"]
        sl_f  = evdf[evdf["type"] == "STOPLOSS_FULL"]
        for subset, marker, color in [
            (half, "^", "#FFC107"), (full, "^", "#F44336"), (norm, "v", "#4CAF50"),
            (sl_h, "x", "#FF6F00"), (sl_f, "x", "#B71C1C"),
        ]:
            if not subset.empty:
                ax1.scatter(pd.to_datetime(subset["Date"]), subset["value"],
                            marker=marker, color=color, s=45, zorder=5, alpha=0.75,
                            edgecolors="black", linewidths=0.3)

    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    qld_ath = qld["Close"].cummax()
    qld_dd = (qld["Close"] - qld_ath) / qld_ath * 100
    ax2.fill_between(qld.index, qld_dd, 0, alpha=0.4, color="#9C27B0")
    ax2.axhline(y=-10, color="orange", linestyle="--", linewidth=0.8, alpha=0.7, label="-10%")
    ax2.axhline(y=-20, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="-20%")
    ax2.axhline(y=-5, color="green", linestyle="--", linewidth=0.8, alpha=0.7, label="-5%")
    ax2.set_ylabel("QLD Drawdown (%)")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    rsi = compute_rsi(qld["Close"])
    ax3.plot(qld.index, rsi, linewidth=0.8, color="#607D8B")
    ax3.axhline(y=30, color="green", linestyle="--", linewidth=0.8, alpha=0.7, label="RSI 30")
    ax3.axhline(y=70, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="RSI 70")
    ax3.fill_between(qld.index, rsi, 30, where=rsi < 30, alpha=0.3, color="green")
    ax3.set_ylabel("QLD RSI (14)")
    ax3.set_xlabel("Date")
    ax3.set_ylim(0, 100)
    ax3.legend(loc="lower left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if out_path is None:
        out_path = BASE_DIR / "03_RESULT" / "switching_backtest.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {out_path}")
    plt.close(fig)


def extract_episodes(events_df: pd.DataFrame) -> list:
    """Extract attack episodes: from first non-NORMAL entry to TO_NORMAL."""
    episodes = []
    start, start_type = None, None
    for _, row in events_df.iterrows():
        t = row["type"]
        d = pd.to_datetime(row["Date"])
        if t in ("TO_HALF_ATTACK", "TO_FULL_ATTACK") and start is None:
            start = d
            start_type = t
        elif t == "TO_NORMAL" and start is not None:
            episodes.append({"start": start, "end": d, "entry_type": start_type})
            start = None
    return episodes


def analyze_episodes(
    episodes: list,
    bh_portfolio: pd.Series,
    sw_portfolio: pd.Series,
    sw_rsi_portfolio: pd.Series,
    sw_sl_portfolio: pd.Series,
    events_rsi_df: pd.DataFrame,
    events_sl_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute S1/S2/S3/S4 returns for each S2 attack episode."""

    def _get_return(pf: pd.Series, s, e):
        idx = pf.index
        s_i = idx[idx >= s][0] if (idx >= s).any() else idx[-1]
        e_i = idx[idx >= e][0] if (idx >= e).any() else idx[-1]
        if s_i == e_i:
            return 0.0
        return float(pf.loc[e_i] / pf.loc[s_i] - 1)

    def _attack_dates(evdf):
        dates = set()
        if not evdf.empty:
            for _, r in evdf.iterrows():
                if r["type"] not in ("TO_NORMAL", "STOPLOSS_HALF", "STOPLOSS_FULL"):
                    dates.add(pd.to_datetime(r["Date"]).date())
        return dates

    rsi_attack_dates = _attack_dates(events_rsi_df)
    sl_attack_dates  = _attack_dates(events_sl_df)

    rows = []
    for i, ep in enumerate(episodes, 1):
        s, e = ep["start"], ep["end"]
        r1 = _get_return(bh_portfolio,    s, e)
        r2 = _get_return(sw_portfolio,    s, e)
        r3 = _get_return(sw_rsi_portfolio, s, e)
        r4 = _get_return(sw_sl_portfolio, s, e)

        s3_active = any(s.date() <= d <= e.date() for d in rsi_attack_dates)
        s4_active = any(s.date() <= d <= e.date() for d in sl_attack_dates)

        rows.append({
            "#": i,
            "start": s.strftime("%Y-%m-%d"),
            "end": e.strftime("%Y-%m-%d"),
            "days": (e - s).days,
            "entry": ep["entry_type"].replace("TO_", ""),
            "S1_QQQ":   r1,
            "S2_Switch": r2,
            "S3_RSI":   r3,
            "S4_SL":    r4,
            "S2_excess": r2 - r1,
            "S3_excess": r3 - r1,
            "S4_excess": r4 - r1,
            "S3_active": s3_active,
            "S4_active": s4_active,
            "S2_beats_S1": r2 > r1,
            "S3_beats_S1": r3 > r1,
            "S4_beats_S1": r4 > r1,
        })
    return pd.DataFrame(rows)


def print_episode_table(ep_df: pd.DataFrame, title: str):
    W = 115
    print(f"\n{'=' * W}")
    print(f"EPISODE ANALYSIS  [{title}]")
    print(f"{'=' * W}")
    hdr = (f"  {'#':>3}  {'Start':>12}  {'End':>12}  {'Days':>5}  {'Entry':>14}"
           f"  {'S1(QQQ)':>9}  {'S2(Sw)':>9}  {'S3(RSI)':>9}  {'S4(SL)':>9}"
           f"  {'S2-S1':>8}  {'S3-S1':>8}  {'S4-S1':>8}")
    print(hdr)
    print("-" * W)
    for _, r in ep_df.iterrows():
        s2m = "✓" if r["S2_beats_S1"] else "✗"
        s3m = ("✓" if r["S3_beats_S1"] else "✗") if r["S3_active"] else " -"
        s4m = ("✓" if r["S4_beats_S1"] else "✗") if r["S4_active"] else " -"
        print(
            f"  {r['#']:>3}  {r['start']:>12}  {r['end']:>12}  {r['days']:>5}  {r['entry']:>14}"
            f"  {r['S1_QQQ']:>+8.1%}  {r['S2_Switch']:>+8.1%}  {r['S3_RSI']:>+8.1%}  {r['S4_SL']:>+8.1%}"
            f"  {r['S2_excess']:>+7.1%}{s2m}  {r['S3_excess']:>+7.1%}{s3m}  {r['S4_excess']:>+7.1%}{s4m}"
        )

    print("-" * W)
    n = len(ep_df)
    s2_win = ep_df["S2_beats_S1"].sum()
    s4_ep  = ep_df[ep_df["S4_active"]]
    s4_win = s4_ep["S4_beats_S1"].sum() if not s4_ep.empty else 0
    s3_ep  = ep_df[ep_df["S3_active"]]
    s3_win = s3_ep["S3_beats_S1"].sum() if not s3_ep.empty else 0

    print(f"\n  Total episodes (S2): {n}")
    print(f"  S2 beat S1: {s2_win}/{n} ({s2_win/n:.0%})  |  avg excess: {ep_df['S2_excess'].mean():+.2%}")
    if not s3_ep.empty:
        print(f"  S3 active: {len(s3_ep)}/{n}  → beat S1: {s3_win}/{len(s3_ep)} ({s3_win/len(s3_ep):.0%})"
              f"  avg excess: {s3_ep['S3_excess'].mean():+.2%}")
    if not s4_ep.empty:
        print(f"  S4 active: {len(s4_ep)}/{n}  → beat S1: {s4_win}/{len(s4_ep)} ({s4_win/len(s4_ep):.0%})"
              f"  avg excess: {s4_ep['S4_excess'].mean():+.2%}")
    print(f"{'=' * W}")


def plot_episode_comparison(ep_df: pd.DataFrame, title: str, out_path: Path):
    n = len(ep_df)
    fig_h = max(6, 0.5 * n + 3)
    fig, axes = plt.subplots(1, 2, figsize=(18, fig_h))
    fig.suptitle(f"Episode Analysis: Returns During Each S2 Attack Period\n[{title}]",
                 fontsize=13, fontweight="bold")

    labels = [f"#{r['#']} {r['start'][:7]}~{r['end'][:7]}" for _, r in ep_df.iterrows()]
    y = np.arange(n)
    h = 0.20

    # ── Left: absolute returns ───────────────────────────────────────────────
    ax = axes[0]
    ax.barh(y + 1.5*h, ep_df["S1_QQQ"]    * 100, h, label="S1 QQQ",           color="#2196F3", alpha=0.85)
    ax.barh(y + 0.5*h, ep_df["S2_Switch"] * 100, h, label="S2 Switching",      color="#FF5722", alpha=0.85)
    ax.barh(y - 0.5*h, ep_df["S3_RSI"]   * 100, h, label="S3 RSI",            color="#9C27B0", alpha=0.85)
    ax.barh(y - 1.5*h, ep_df["S4_SL"]    * 100, h, label="S4 Stop-Loss",      color="#4CAF50", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Return (%)")
    ax.set_title("Absolute Return During Each Episode")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    # ── Right: excess returns vs S1 ──────────────────────────────────────────
    ax2 = axes[1]
    for offset, col_key, active_key, color_pos, color_neg, label in [
        (0.5*h,  "S2_excess", None,        "#FF5722", "#FFCCBC", "S2-S1"),
        (0,      "S3_excess", "S3_active", "#9C27B0", "#E1BEE7", "S3-S1"),
        (-0.5*h, "S4_excess", "S4_active", "#4CAF50", "#C8E6C9", "S4-S1"),
    ]:
        colors = []
        for i, (_, r) in enumerate(ep_df.iterrows()):
            is_active = (active_key is None) or r[active_key]
            colors.append(color_pos if (r[col_key] >= 0 and is_active) else
                          (color_neg if is_active else "#E0E0E0"))
        ax2.barh(y + offset, ep_df[col_key] * 100, h, color=colors, alpha=0.9, label=label)

    for i, (_, r) in enumerate(ep_df.iterrows()):
        if not r["S3_active"]:
            ax2.annotate("skip", xy=(0, i), fontsize=5, color="#9C27B0", va="center", ha="left")
        if not r["S4_active"]:
            ax2.annotate("skip", xy=(0, i - 0.5*h), fontsize=5, color="#4CAF50", va="center", ha="left")

    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Excess Return vs S1 QQQ (%)")
    ax2.set_title("Excess Return vs Buy & Hold\n(gray = RSI/SL filter blocked; 'skip' = not active)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="x", alpha=0.3)

    win_s2 = ep_df["S2_beats_S1"].sum()
    s3_ep = ep_df[ep_df["S3_active"]]
    s4_ep = ep_df[ep_df["S4_active"]]
    win_s3 = s3_ep["S3_beats_S1"].sum() if not s3_ep.empty else 0
    win_s4 = s4_ep["S4_beats_S1"].sum() if not s4_ep.empty else 0
    lines = [
        f"S2 win: {win_s2}/{n} ({win_s2/n:.0%})  avg: {ep_df['S2_excess'].mean():+.1%}",
    ]
    if not s3_ep.empty:
        lines.append(f"S3 active {len(s3_ep)}/{n}  win: {win_s3}/{len(s3_ep)} ({win_s3/len(s3_ep):.0%})  avg: {s3_ep['S3_excess'].mean():+.1%}")
    if not s4_ep.empty:
        lines.append(f"S4 active {len(s4_ep)}/{n}  win: {win_s4}/{len(s4_ep)} ({win_s4/len(s4_ep):.0%})  avg: {s4_ep['S4_excess'].mean():+.1%}")
    ax2.text(0.98, 0.02, "\n".join(lines), transform=ax2.transAxes, fontsize=8,
             va="bottom", ha="right",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Episode chart saved to {out_path}")
    plt.close(fig)


_ALLOC = {
    "NORMAL":       "QQQ 100%",
    "HALF_ATTACK":  "QQQ 50% + TQQQ 50%",
    "FULL_ATTACK":  "TQQQ 100%",
    "ATTACK":       "TQQQ 100%",      # legacy S4
    "TRAILING":     "TQQQ 보유 (트레일링 모드)",  # new S4
}


def evt_type_uses_qqq_signal(key: str) -> bool:
    """Returns True if this strategy uses QQQ (not QLD) as its signal asset."""
    return key in ("S4T",) or "트레일링" in key or "TRAILING" in key.upper() or key == "S4 (QQQ시그널)"

_EVT_META = {
    "TO_HALF_ATTACK": ("▲ HALF ATTACK 진입",  "HALF_ATTACK", "QLD -10% 이하 터치 후 -5%까지 반등"),
    "TO_FULL_ATTACK": ("▲ FULL ATTACK 진입",  "FULL_ATTACK", "QLD -20% 이하 터치 후 -10%까지 반등"),
    "TO_NORMAL":      ("▼ 방어 복귀 (고점 회복)", "NORMAL",   "QLD 전 고점 완전 회복"),
    "STOPLOSS_HALF":  ("⚠ 손절 (HALF→QQQ)",   "NORMAL",   "QLD 다시 -10%로 하락 → 손절 발동"),
    "STOPLOSS_FULL":  ("⚠ 손절 (FULL→QQQ)",   "NORMAL",   "QLD 다시 -20%로 하락 → 손절 발동"),
    # S4 (trailing exit) events — signal = QLD (same as S2)
    "TO_TRAILING": ("▶ 트레일링 모드 진입",          "TRAILING", "QLD 전고점 회복 → TQQQ 트레일링스탑 개시 (매도 보류)"),
    "TRAIL_EXIT":  ("⚠ 트레일링 청산 (고점 -15%)",  "NORMAL",   "TQQQ 고점 대비 -15% 하락 → QQQ 전환"),
    "TRAIL_FLOOR": ("⚠ 트레일링 플로어 청산",        "NORMAL",   "TQQQ 트레일링 진입가 이하 하락 → 수익 보전 후 QQQ 전환"),
    "HALF_STOP":   ("⚠ HALF 손절 (QLD -15%)",      "NORMAL",   "HALF 상태에서 QLD -15% 도달 → QQQ 전환 (스위칭 로직 유지)"),
    # (legacy S4 QQQ-signal events kept for reference)
    "TO_ATTACK":   ("▲ TQQQ 100% 진입",            "ATTACK",   "QQQ -10% 이하 터치 후 전고점 완전 회복"),
    "HARD_STOP":   ("⚠ 하드스탑 (진입가 -10%)",    "NORMAL",   "TQQQ 진입가 대비 -10% 하락 → 청산"),
    "TRAIL_STOP":  ("⚠ 트레일링스탑 (고점 -15%)",  "NORMAL",   "TQQQ 보유 중 고점 대비 -15% 하락 → 청산"),
}


def generate_report(
    qld: pd.DataFrame,
    events_dict: dict,
    title: str,
    out_path: Path,
    initial_capital: float = 100_000,
    signal_per_strategy: dict = None,   # {key: DataFrame} overrides qld per strategy
):
    # Pre-compute signal series for each unique signal df
    _signal_cache = {}

    def _get_signal_series(sig_df: pd.DataFrame):
        key_id = id(sig_df)
        if key_id not in _signal_cache:
            rsi   = compute_rsi(sig_df["Close"])
            ath   = sig_df["Close"].cummax()
            dd    = (sig_df["Close"] - ath) / ath
            _signal_cache[key_id] = (sig_df, ath, dd, rsi)
        return _signal_cache[key_id]

    def _signal_info(sig_df, date):
        _, ath_s, dd_s, rsi_s = _get_signal_series(sig_df)
        d   = pd.to_datetime(date)
        idx = sig_df.index
        loc = d if d in idx else (idx[idx <= d][-1] if len(idx[idx <= d]) else idx[0])
        price = float(sig_df.loc[loc, "Close"])
        ath   = float(ath_s.loc[loc])
        dd    = float(dd_s.loc[loc])
        r     = float(rsi_s.loc[loc]) if loc in rsi_s.index else float("nan")
        return price, ath, dd, r

    strategy_labels = {
        "S1":  "전략 1: QQQ Buy & Hold",
        "S2":  "전략 2: QLD 시그널 스위칭 (RSI 없음)",
        "S3":  "전략 3: QLD 시그널 + RSI 필터",
        "S4":  "전략 4: QLD 시그널 + TQQQ 트레일링 청산 (하드스탑 없음)",
        "S4T": "전략 4: QQQ 시그널 + 트레일링스탑",
    }

    W = 95
    lines = []
    lines.append("=" * W)
    lines.append("  백테스트 스위칭 이벤트 상세 보고서")
    lines.append(f"  기간: {qld.index[0].date()} ~ {qld.index[-1].date()}"
                 f"  │  초기 자본: ${initial_capital:,.0f}  │  {title}")
    lines.append("=" * W)
    lines.append("")
    lines.append("  시그널 자산: QLD (2x QQQ)  │  방어 자산: QQQ  │  공격 자산: TQQQ")
    lines.append("")
    lines.append("  [전략 요약]")
    lines.append("  S1: 단순 QQQ 보유 — 스위칭 없음")
    for key, evdf in events_dict.items():
        lines.append(f"  {strategy_labels.get(key, key)}  →  총 {len(evdf)}건")
    lines.append("")

    for key, evdf in events_dict.items():
        # Determine which signal df to use for this strategy
        sig_df = (signal_per_strategy or {}).get(key, qld)
        is_s4t = evt_type_uses_qqq_signal(key)

        lines.append("━" * W)
        lbl_header = strategy_labels.get(key, key)
        sig_name   = "QQQ" if is_s4t else "QLD"
        lines.append(f"  {lbl_header}  (시그널: {sig_name})")
        lines.append("━" * W)

        if evdf.empty:
            lines.append("  이벤트 없음 (Buy & Hold)")
            lines.append("")
            continue

        state = "NORMAL"
        for i, (_, row) in enumerate(evdf.iterrows(), 1):
            date     = pd.to_datetime(row["Date"])
            evt_type = row["type"]
            value    = float(row["value"])

            price, ath, dd, r = _signal_info(sig_df, date)
            label, to_state, trigger = _EVT_META.get(
                evt_type, (evt_type, state, "")
            )

            from_alloc = _ALLOC.get(state, state)
            to_alloc   = _ALLOC.get(to_state, to_state)

            lines.append("")
            lines.append(f"  #{i:02d}  {date.strftime('%Y-%m-%d')}  {label}")
            lines.append(f"       {sig_name} 가격  : ${price:>8.2f}")
            lines.append(f"       ATH(고점) : ${ath:>8.2f}  │  고점대비 : {dd:>+7.2%}  │  RSI(14) : {r:>5.1f}")
            lines.append(f"       포트폴리오: ${value:>12,.0f}")
            lines.append(f"       전환      : {from_alloc}  →  {to_alloc}")
            lines.append(f"       트리거    : {trigger}")
            _is_rsi_strategy = (key in ("S3",) or "RSI" in key.upper()
                                    or key.startswith("S3 "))
            if _is_rsi_strategy and evt_type in ("TO_HALF_ATTACK", "TO_FULL_ATTACK"):
                lines.append(f"       RSI 조건  : 터치 시 RSI < 30  +  현재 RSI ≥ 30 확인 (현재 RSI: {r:.1f})")

            state = to_state

        lines.append("")

    lines.append("=" * W)
    text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Report saved → {out_path}")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# JEPQ vs QQQ Defensive Asset Comparison
# ─────────────────────────────────────────────────────────────────────────────

_COMPARE_STYLES = {
    "S1 QQQ":  ("#9CA3AF", "--", 1.0),
    "S2 QQQ":  ("#2563EB", "-",  1.4),
    "S3 QQQ":  ("#1E3A8A", "-",  1.4),
    "S4 QQQ":  ("#7C3AED", "-",  1.6),
    "S1 JEPQ": ("#D97706", "--", 1.0),
    "S2 JEPQ": ("#DC2626", "-",  1.4),
    "S3 JEPQ": ("#16A34A", "-",  1.4),
    "S4 JEPQ": ("#831843", "-",  1.6),
}


def plot_results_compare(
    portfolios: dict,
    events_s2_qqq: pd.DataFrame,
    events_s3_qqq: pd.DataFrame,
    events_s2_jepq: pd.DataFrame,
    events_s3_jepq: pd.DataFrame,
    qld: pd.DataFrame,
    title: str,
    out_path: Path,
    extra_events: dict = None,   # {"S4 QQQ": df, "S4 JEPQ": df}
):
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(17, 14), height_ratios=[3, 1, 1], sharex=True
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for name, series in portfolios.items():
        color, ls, lw = _COMPARE_STYLES.get(name, ("#999", "-", 1.0))
        ax1.plot(series.index, series.values, label=name,
                 color=color, linestyle=ls, linewidth=lw)

    # event markers for S2-QQQ events (reference baseline)
    for evdf, alpha in [(events_s2_qqq, 0.85), (events_s2_jepq, 0.45)]:
        if evdf.empty:
            continue
        for etype, marker, color in [
            ("TO_HALF_ATTACK", "^", "#F59E0B"),
            ("TO_FULL_ATTACK", "^", "#EF4444"),
            ("TO_NORMAL",      "v", "#22C55E"),
        ]:
            sub = evdf[evdf["type"] == etype]
            if not sub.empty:
                ax1.scatter(pd.to_datetime(sub["Date"]), sub["value"],
                            marker=marker, color=color, s=50, zorder=5,
                            alpha=alpha, edgecolors="black", linewidths=0.4)

    # S4 event markers (trailing mode + exits)
    if extra_events:
        for _evdf in extra_events.values():
            if _evdf.empty:
                continue
            for etype, marker, color in [
                ("TO_TRAILING",    "D",  "#7C3AED"),
                ("TRAIL_EXIT",     "x",  "#9D174D"),
                ("TRAIL_FLOOR",    "x",  "#BE123C"),
                ("HALF_STOP",      "v",  "#B45309"),
                ("TO_HALF_ATTACK", "^",  "#F59E0B"),
                ("TO_FULL_ATTACK", "^",  "#EF4444"),
            ]:
                sub = _evdf[_evdf["type"] == etype]
                if not sub.empty:
                    ax1.scatter(pd.to_datetime(sub["Date"]), sub["value"],
                                marker=marker, color=color, s=45, zorder=5,
                                alpha=0.65, edgecolors="black", linewidths=0.3)

    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # QLD drawdown
    qld_ath = qld["Close"].cummax()
    qld_dd  = (qld["Close"] - qld_ath) / qld_ath * 100
    ax2.fill_between(qld.index, qld_dd, 0, alpha=0.4, color="#7C3AED")
    ax2.axhline(y=-10, color="orange",   linestyle="--", linewidth=0.8, alpha=0.7, label="-10%")
    ax2.axhline(y=-20, color="red",      linestyle="--", linewidth=0.8, alpha=0.7, label="-20%")
    ax2.axhline(y=-5,  color="green",    linestyle="--", linewidth=0.8, alpha=0.7, label="-5%")
    ax2.set_ylabel("QLD Drawdown (%)")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # QLD RSI
    rsi = compute_rsi(qld["Close"])
    ax3.plot(qld.index, rsi, linewidth=0.8, color="#607D8B")
    ax3.axhline(y=30, color="green", linestyle="--", linewidth=0.8, alpha=0.7, label="RSI 30")
    ax3.axhline(y=70, color="red",   linestyle="--", linewidth=0.8, alpha=0.7, label="RSI 70")
    ax3.fill_between(qld.index, rsi, 30, where=rsi < 30, alpha=0.3, color="green")
    ax3.set_ylabel("QLD RSI (14)")
    ax3.set_xlabel("Date")
    ax3.set_ylim(0, 100)
    ax3.legend(loc="lower left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Compare chart saved → {out_path}")
    plt.close(fig)


def run_backtest_jepq_compare(
    qqq: pd.DataFrame, qld: pd.DataFrame, tqqq: pd.DataFrame, jepq: pd.DataFrame,
    initial_capital: float = 100_000,
    start_date: str = "2022-06-01",
):
    # ── Align dates ────────────────────────────────────────────────────────────
    common = (qqq.index
              .intersection(qld.index)
              .intersection(tqqq.index)
              .intersection(jepq.index))
    common = common[common >= pd.Timestamp(start_date)]

    qqq_  = qqq.loc[common]
    qld_  = qld.loc[common]
    tqqq_ = tqqq.loc[common]
    jepq_ = jepq.loc[common]

    period = f"{common[0].date()} ~ {common[-1].date()}  ({len(common)} 거래일)"
    title  = f"S1-S4 방어자산 QQQ vs JEPQ 비교  |  {period}"
    print(f"\nData period: {period}")

    # ── QQQ-defensive strategies ───────────────────────────────────────────────
    print("Running S1 QQQ..."); bh_qqq = strategy_buy_and_hold(qqq_, initial_capital, series_name="S1 QQQ")
    print("Running S2 QQQ..."); sw_qqq, ev_sw_qqq = strategy_switching(qqq_, qld_, tqqq_, initial_capital, series_name="S2 QQQ")
    print("Running S3 QQQ..."); sw_rsi_qqq, ev_rsi_qqq = strategy_switching_rsi(qqq_, qld_, tqqq_, initial_capital, series_name="S3 QQQ")
    print("Running S4 QQQ..."); s4_qqq, ev_s4_qqq = strategy_s4_trailing(qqq_, qld_, tqqq_, initial_capital, series_name="S4 QQQ")

    # ── JEPQ-defensive strategies ─────────────────────────────────────────────
    print("Running S1 JEPQ..."); bh_jepq = strategy_buy_and_hold(jepq_, initial_capital, series_name="S1 JEPQ")
    print("Running S2 JEPQ..."); sw_jepq, ev_sw_jepq = strategy_switching(jepq_, qld_, tqqq_, initial_capital, series_name="S2 JEPQ")
    print("Running S3 JEPQ..."); sw_rsi_jepq, ev_rsi_jepq = strategy_switching_rsi(jepq_, qld_, tqqq_, initial_capital, series_name="S3 JEPQ")
    print("Running S4 JEPQ..."); s4_jepq, ev_s4_jepq = strategy_s4_trailing(jepq_, qld_, tqqq_, initial_capital, series_name="S4 JEPQ")

    # ── Statistics — two groups ────────────────────────────────────────────────
    W = 120
    col_w = 14

    def _print_group(label, stats_list):
        print(f"\n  ▶ {label}")
        print("  " + "-" * (W - 2))
        header = f"  {'':20s}" + "".join(f"{s['전략']:>{col_w}s}" for s in stats_list)
        print(header)
        print("  " + "-" * (W - 2))
        for key in ["최종 자산", "총 수익률", "CAGR", "MDD", "연간 변동성", "샤프 비율"]:
            row = f"  {key:20s}" + "".join(f"{str(s.get(key, '')):>{col_w}s}" for s in stats_list)
            print(row)
        cnt = f"  {'스위칭 횟수':20s}" + "".join(f"{str(s.get('스위칭 횟수', '-')):>{col_w}s}" for s in stats_list)
        print(cnt)

    print("\n" + "=" * W)
    print(f"  S1-S4  QQQ vs JEPQ 방어자산 비교  [{period}]")
    print(f"  ※ 수정종가(adjusted close) — 주가분할 + 배당 재투자 반영")
    print("=" * W)

    _print_group("QQQ 방어자산", [
        compute_statistics(bh_qqq,     "S1 QQQ"),
        compute_statistics(sw_qqq,     "S2 QQQ",  switch_count=len(ev_sw_qqq)),
        compute_statistics(sw_rsi_qqq, "S3 QQQ",  switch_count=len(ev_rsi_qqq)),
        compute_statistics(s4_qqq,     "S4 QQQ",  switch_count=len(ev_s4_qqq)),
    ])
    _print_group("JEPQ 방어자산", [
        compute_statistics(bh_jepq,     "S1 JEPQ"),
        compute_statistics(sw_jepq,     "S2 JEPQ", switch_count=len(ev_sw_jepq)),
        compute_statistics(sw_rsi_jepq, "S3 JEPQ", switch_count=len(ev_rsi_jepq)),
        compute_statistics(s4_jepq,     "S4 JEPQ", switch_count=len(ev_s4_jepq)),
    ])
    print("=" * W)

    # Event summary (S2/S4 only — S3 usually too few)
    for label, evdf in [
        ("S2 QQQ", ev_sw_qqq), ("S3 QQQ", ev_rsi_qqq),
        ("S4 QQQ", ev_s4_qqq), ("S4 JEPQ", ev_s4_jepq),
    ]:
        if not evdf.empty:
            print(f"\n{label} Events ({len(evdf)} total):")
            print("-" * 55)
            for _, row in evdf.iterrows():
                date_str = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
                print(f"  {date_str}  {row['type']:22s}  ${row['value']:>12,.0f}")

    # ── Chart ─────────────────────────────────────────────────────────────────
    print("\nGenerating compare chart...")
    out_path = BASE_DIR / "03_RESULT" / "switching_backtest_jepq_compare.png"
    portfolios = {
        "S1 QQQ": bh_qqq, "S2 QQQ": sw_qqq, "S3 QQQ": sw_rsi_qqq, "S4 QQQ": s4_qqq,
        "S1 JEPQ": bh_jepq, "S2 JEPQ": sw_jepq, "S3 JEPQ": sw_rsi_jepq, "S4 JEPQ": s4_jepq,
    }
    plot_results_compare(
        portfolios, ev_sw_qqq, ev_rsi_qqq, ev_sw_jepq, ev_rsi_jepq,
        qld_, title, out_path,
        extra_events={"S4 QQQ": ev_s4_qqq, "S4 JEPQ": ev_s4_jepq},
    )

    # ── Report ────────────────────────────────────────────────────────────────
    print("Generating switching report...")
    report_path = BASE_DIR / "03_RESULT" / "switching_backtest_jepq_compare_report.txt"
    generate_report(
        qld_,
        {
            "S2 QQQ (방어=QQQ)":        ev_sw_qqq,
            "S3 QQQ (방어=QQQ, RSI)":   ev_rsi_qqq,
            "S4 QQQ (방어=QQQ, Trail)":  ev_s4_qqq,
            "S2 JEPQ (방어=JEPQ)":       ev_sw_jepq,
            "S3 JEPQ (방어=JEPQ, RSI)":  ev_rsi_jepq,
            "S4 JEPQ (방어=JEPQ, Trail)": ev_s4_jepq,
        },
        title,
        report_path,
        initial_capital,
    )


def plot_results_s1s4(
    bh_portfolio: pd.Series,
    sw_portfolio: pd.Series,
    sw_rsi_portfolio: pd.Series,
    s4_portfolio: pd.Series,
    events_s2: pd.DataFrame,
    events_s3: pd.DataFrame,
    events_s4: pd.DataFrame,
    qld: pd.DataFrame,
    qqq: pd.DataFrame,   # kept for API compat; not used in new 3-panel version
    title: str,
    out_path: Path,
):
    """3-panel chart: portfolio values, QLD drawdown (all signals), QLD RSI."""
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(17, 14), height_ratios=[3, 1, 1], sharex=True
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── Panel 1: portfolio values ─────────────────────────────────────────────
    colors = ["#6B7280", "#2563EB", "#1E3A8A", "#DC2626"]
    for series, color, lbl in zip(
        [bh_portfolio, sw_portfolio, sw_rsi_portfolio, s4_portfolio],
        colors,
        ["S1: QQQ Buy&Hold", "S2: Switching (QLD)", "S3: Switching+RSI (QLD)",
         "S4: Trailing Stop (QQQ)"],
    ):
        ax1.plot(series.index, series.values, label=lbl, color=color, linewidth=1.3)

    # S2/S3 event markers
    for evdf in [events_s2, events_s3]:
        if evdf.empty:
            continue
        for etype, marker, color in [
            ("TO_HALF_ATTACK", "^", "#F59E0B"),
            ("TO_FULL_ATTACK", "^", "#EF4444"),
            ("TO_NORMAL",      "v", "#22C55E"),
        ]:
            sub = evdf[evdf["type"] == etype]
            if not sub.empty:
                ax1.scatter(pd.to_datetime(sub["Date"]), sub["value"],
                            marker=marker, color=color, s=45, zorder=5,
                            alpha=0.7, edgecolors="black", linewidths=0.4)
    # S4 event markers
    for etype, marker, color in [
        ("TO_HALF_ATTACK", "^",  "#F59E0B"),
        ("TO_FULL_ATTACK", "^",  "#EF4444"),
        ("TO_TRAILING",    "D",  "#7C3AED"),   # diamond = trailing mode start
        ("TRAIL_EXIT",     "x",  "#9D174D"),   # trailing stop exit
        ("TRAIL_FLOOR",    "x",  "#BE123C"),   # floor exit
        ("HALF_STOP",      "v",  "#B45309"),   # HALF -15% stop
    ]:
        sub = events_s4[events_s4["type"] == etype] if not events_s4.empty else pd.DataFrame()
        if not sub.empty:
            ax1.scatter(pd.to_datetime(sub["Date"]), sub["value"],
                        marker=marker, color=color, s=60, zorder=5,
                        alpha=0.85, edgecolors="black", linewidths=0.5)

    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── Panel 2: QLD drawdown (all strategies use QLD signal) ────────────────
    qld_ath = qld["Close"].cummax()
    qld_dd  = (qld["Close"] - qld_ath) / qld_ath * 100
    ax2.fill_between(qld.index, qld_dd, 0, alpha=0.35, color="#2563EB")
    ax2.axhline(y=-10, color="orange", linestyle="--", linewidth=0.8, alpha=0.7, label="-10%")
    ax2.axhline(y=-20, color="red",    linestyle="--", linewidth=0.8, alpha=0.7, label="-20%")
    ax2.axhline(y=-5,  color="green",  linestyle="--", linewidth=0.8, alpha=0.7, label="-5%")
    ax2.set_ylabel("QLD Drawdown (%)\n(S2/S3/S4 signal)")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: QLD RSI (S3 condition) ──────────────────────────────────────
    rsi = compute_rsi(qld["Close"])
    ax3.plot(qld.index, rsi, linewidth=0.8, color="#607D8B")
    ax3.axhline(y=30, color="green", linestyle="--", linewidth=0.8, alpha=0.7, label="RSI 30")
    ax3.axhline(y=70, color="red",   linestyle="--", linewidth=0.8, alpha=0.7, label="RSI 70")
    ax3.fill_between(qld.index, rsi, 30, where=rsi < 30, alpha=0.3, color="green")
    ax3.set_ylabel("QLD RSI (14)\n(S3 condition)")
    ax3.set_xlabel("Date")
    ax3.set_ylim(0, 100)
    ax3.legend(loc="lower left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"S1-S4 chart saved → {out_path}")
    plt.close(fig)


def run_backtest_qqq_s1s4(
    qqq: pd.DataFrame, qld: pd.DataFrame, tqqq: pd.DataFrame,
    initial_capital: float = 100_000,
    start_date: str = "2010-02-11",
):
    """Compare S1-S4 (new S4 = QQQ trailing stop), all using QQQ as defensive."""
    common = (qqq.index.intersection(qld.index).intersection(tqqq.index))
    common = common[common >= pd.Timestamp(start_date)]

    qqq_  = qqq.loc[common]
    qld_  = qld.loc[common]
    tqqq_ = tqqq.loc[common]

    period = f"{common[0].date()} ~ {common[-1].date()}  ({len(common)} 거래일)"
    title  = f"S1-S4 (새 전략4: S2 진입 + QLD 전고점 회복 시 TQQQ 트레일링 -15%)  |  {period}"
    print(f"\nData period: {period}")

    print("Running S1: QQQ Buy & Hold...")
    bh = strategy_buy_and_hold(qqq_, initial_capital, series_name="S1 B&H")

    print("Running S2: Switching (QLD signal)...")
    sw, ev_sw = strategy_switching(qqq_, qld_, tqqq_, initial_capital, series_name="S2 Switching")

    print("Running S3: Switching + RSI (QLD signal)...")
    sw_rsi, ev_rsi = strategy_switching_rsi(qqq_, qld_, tqqq_, initial_capital, series_name="S3 +RSI")

    print("Running S4: Trail Exit (QLD signal, no hard stop)...")
    s4, ev_s4 = strategy_s4_trailing(qqq_, qld_, tqqq_, initial_capital, series_name="S4 TrailExit")

    # ── Statistics ────────────────────────────────────────────────────────────
    all_stats = [
        compute_statistics(bh,     "S1 (QQQ B&H)"),
        compute_statistics(sw,     "S2 (Switching)",  switch_count=len(ev_sw)),
        compute_statistics(sw_rsi, "S3 (+RSI)",       switch_count=len(ev_rsi)),
        compute_statistics(s4,     "S4 (Trailing)",   switch_count=len(ev_s4)),
    ]

    print("\n" + "=" * 95)
    print(f"S1–S4 비교  [{period}]")
    print("  S1: QQQ 보유  │  S2: QLD 시그널 스위칭  │  S3: S2 + RSI 필터  │  S4: QQQ 시그널 + 트레일링스탑")
    print("=" * 95)
    col_w = 18
    header = f"{'':22s}" + "".join(f"{s['전략']:>{col_w}s}" for s in all_stats)
    print(header)
    print("-" * 95)
    for key in ["최종 자산", "총 수익률", "CAGR", "MDD", "연간 변동성", "샤프 비율"]:
        row = f"{key:22s}" + "".join(f"{str(s.get(key,'')):>{col_w}s}" for s in all_stats)
        print(row)
    cnt_row = f"{'스위칭 횟수':22s}" + "".join(
        f"{str(s.get('스위칭 횟수', '-')):>{col_w}s}" for s in all_stats
    )
    print(cnt_row)
    print("=" * 95)

    for label, evdf in [("S2", ev_sw), ("S3", ev_rsi), ("S4 (trailing)", ev_s4)]:
        if not evdf.empty:
            print(f"\n{label} Events ({len(evdf)} total):")
            print("-" * 55)
            for _, row in evdf.iterrows():
                date_str = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
                print(f"  {date_str}  {row['type']:22s}  ${row['value']:>12,.0f}")

    # ── Chart ─────────────────────────────────────────────────────────────────
    print("\nGenerating S1-S4 chart...")
    out_path = BASE_DIR / "03_RESULT" / "switching_backtest_s1s4_new.png"
    plot_results_s1s4(
        bh, sw, sw_rsi, s4,
        ev_sw, ev_rsi, ev_s4,
        qld_, qqq_, title, out_path
    )

    # ── Report ────────────────────────────────────────────────────────────────
    print("Generating report...")
    report_path = BASE_DIR / "03_RESULT" / "switching_backtest_s1s4_new_report.txt"
    generate_report(
        qld_,
        {"S2": ev_sw, "S3": ev_rsi, "S4 (QQQ시그널)": ev_s4},
        title,
        report_path,
        initial_capital,
        signal_per_strategy={"S4 (QQQ시그널)": qqq_},
    )


def run_backtest(qqq: pd.DataFrame, qld: pd.DataFrame, tqqq: pd.DataFrame,
                 initial_capital: float, title: str, out_filename: str):
    common_dates = qqq.index.intersection(qld.index).intersection(tqqq.index)
    qqq = qqq.loc[common_dates]
    qld = qld.loc[common_dates]
    tqqq = tqqq.loc[common_dates]

    print(f"Data period: {common_dates[0].date()} ~ {common_dates[-1].date()}  ({len(common_dates)} trading days)")

    print("\nRunning Strategy 1: QQQ Buy & Hold...")
    bh_portfolio = strategy_buy_and_hold(qqq, initial_capital)

    print("Running Strategy 2: QLD Signal Switching...")
    sw_portfolio, events_df = strategy_switching(qqq, qld, tqqq, initial_capital)

    print("Running Strategy 3: Switching + RSI Filter...")
    sw_rsi_portfolio, events_rsi_df = strategy_switching_rsi(qqq, qld, tqqq, initial_capital)

    print("Running Strategy 4: Switching + Stop-Loss...")
    sw_sl_portfolio, events_sl_df = strategy_switching_stoploss(qqq, qld, tqqq, initial_capital)

    print("\n" + "=" * 100)
    print(f"BACKTEST RESULTS  [{title}]")
    print("=" * 100)

    stats_bh  = compute_statistics(bh_portfolio,    "QQQ Buy&Hold")
    stats_sw  = compute_statistics(sw_portfolio,    "Switching",    switch_count=len(events_df))
    stats_rsi = compute_statistics(sw_rsi_portfolio,"Switch+RSI",   switch_count=len(events_rsi_df))
    stats_sl  = compute_statistics(sw_sl_portfolio, "Switch+SL",    switch_count=len(events_sl_df))

    header = f"{'':20s} {'Strategy 1':>16s} {'Strategy 2':>16s} {'Strategy 3':>16s} {'Strategy 4':>16s}"
    print(header)
    print("-" * 100)
    for key in stats_bh:
        print(f"{key:20s} {str(stats_bh[key]):>16s} {str(stats_sw.get(key,'')):>16s}"
              f" {str(stats_rsi.get(key,'')):>16s} {str(stats_sl.get(key,'')):>16s}")
    print(f"{'스위칭/이벤트 수':20s} {'':>16s}"
          f" {len(events_df):>16d} {len(events_rsi_df):>16d} {len(events_sl_df):>16d}")
    print("=" * 100)

    for lbl, evdf in [("S2", events_df), ("S3 (RSI)", events_rsi_df), ("S4 (StopLoss)", events_sl_df)]:
        if not evdf.empty:
            print(f"\n{lbl} Events ({len(evdf)} total):")
            print("-" * 55)
            for _, row in evdf.iterrows():
                date_str = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
                print(f"  {date_str}  {row['type']:22s}  ${row['value']:>12,.0f}")

    # ── Episode-level analysis ───────────────────────────────────────────────
    episodes = extract_episodes(events_df)
    if episodes:
        ep_df = analyze_episodes(
            episodes, bh_portfolio, sw_portfolio, sw_rsi_portfolio, sw_sl_portfolio,
            events_rsi_df, events_sl_df
        )
        print_episode_table(ep_df, title)
        ep_out = BASE_DIR / "03_RESULT" / out_filename.replace(".png", "_episodes.png")
        plot_episode_comparison(ep_df, title, ep_out)

    print("\nGenerating switching report...")
    report_path = BASE_DIR / "03_RESULT" / out_filename.replace(".png", "_report.txt")
    generate_report(
        qld,
        {"S2": events_df, "S3": events_rsi_df, "S4": events_sl_df},
        title,
        report_path,
        initial_capital,
    )

    print("\nGenerating cumulative chart...")
    out_path = BASE_DIR / "03_RESULT" / out_filename
    plot_results(bh_portfolio, sw_portfolio, sw_rsi_portfolio, sw_sl_portfolio,
                 events_df, events_rsi_df, events_sl_df, qld, out_path=out_path)


def main():
    initial_capital = 100_000

    # ── Case A: Intraday (minute-bar) data, 2016~2025 ──────────────────────
    print("\n" + "█" * 80)
    print("CASE A: Intraday data (2016~2025)")
    print("█" * 80)
    print("Loading intraday data...")
    qqq_intra = load_daily_data("1x", "QQQ")
    qld_intra = load_daily_data("2x", "QLD")
    tqqq_intra = load_daily_data("3x", "TQQQ")
    run_backtest(qqq_intra, qld_intra, tqqq_intra, initial_capital,
                 title="Intraday 2016~2025",
                 out_filename="switching_backtest_intraday.png")

    # ── Case B: Yahoo Finance daily data, 2010~2026 ─────────────────────────
    print("\n" + "█" * 80)
    print("CASE B: Yahoo Finance daily data (2010~2026)")
    print("  Note: QLD listed 2006-06-21 / TQQQ listed 2010-02-11")
    print("  → Common period starts 2010-02-11")
    print("█" * 80)
    print("Loading Yahoo Finance daily data...")
    qqq_yf = load_yahoo_daily("QQQ")
    qld_yf = load_yahoo_daily("QLD")
    tqqq_yf = load_yahoo_daily("TQQQ")
    run_backtest(qqq_yf, qld_yf, tqqq_yf, initial_capital,
                 title="Yahoo Daily 2010~2026",
                 out_filename="switching_backtest_yahoo.png")

    # ── Case D: S1-S4 (new S4 trailing stop), QQQ only, 2010~present ─────────
    print("\n" + "█" * 80)
    print("CASE D: S1-S4 비교 (새 전략4: S2 진입 + 트레일링 청산)")
    print("  S4: S2와 동일 진입 (QLD -10%/-20% 터치 후 반등)")
    print("  기존 S2 '전고점 회복 시 즉시 QQQ 전환' 대신 → TQQQ 트레일링스탑 -15%로 수익 극대화")
    print("█" * 80)
    run_backtest_qqq_s1s4(qqq_yf, qld_yf, tqqq_yf, initial_capital=initial_capital)

    # ── Case C: S1-S4, QQQ vs JEPQ defensive, 2022-06 ~ present ─────────────
    print("\n" + "█" * 80)
    print("CASE C: S1-S4  QQQ vs JEPQ 방어자산 비교  (2022-06 ~ present)")
    print("  ※ QQQ, QLD, TQQQ, JEPQ 모두 Yahoo Finance 수정종가 (auto_adjust=True)")
    print("  ※ 배당 재투자 포함한 총 수익률 반영")
    print("█" * 80)
    print("Loading Yahoo Finance daily data for JEPQ comparison...")
    jepq_yf = load_yahoo_daily("JEPQ")
    run_backtest_jepq_compare(
        qqq_yf, qld_yf, tqqq_yf, jepq_yf,
        initial_capital=initial_capital,
        start_date="2022-06-01",
    )


if __name__ == "__main__":
    main()
