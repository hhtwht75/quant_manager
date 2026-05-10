"""
5레그 종가 입력(sig / safe / bounce / defense / agg) 과 다중 레짐 FSM 백테스트.

플랜과 일치하는 **영업일 1바 평가 순서**(고정):

1. ``ma_ok``: 당일 MA200 유효 여부 확인.
2. **규칙 4(전역 최우선)**: ``ma_ok`` 이고 **당일 종가가 ``0.97·MA`` 를 하방 돌파**할 때 NAV 전액 ``safe_asset`` 로 재배분.
   **RMABS RSI 경로**에서 ``use_safe_ma_rule=True`` 이고 레짐이 ``SAFE``/``SAFE_BOUNCE`` 이면 규칙 4 미적용.
   **DMABS**에서는 규칙 4 재적용(MA 교차로 반등 레그 전환 후 재하방 시 전액 안전).
3. 규칙 4 불발 시 **``agg_state``**: ``agg_high`` 갱신 후 (6-1) 진입가 미만 또는 (6-2) 종가 <
   ``agg_high * trail_stop`` 이면 **``defense_state``**(방어 레그 100%).
4. **stress (SAFE 또는 SAFE_BOUNCE)**: ``sig > 1.03 * MA`` 이면 전액 ``agg_state`` (기존 bounce→agg 동일 장벽).
5. **stress RSI/DMABS**: RSI 경로에서는 안전 보유량의 50% **반등종가** 매수; DMABS(``MA5_CROSS_MA120_FULL``)에서는
   **안전→반등 레그 100%** (**``SAFE_MA5_GT_MA120_FULL_BOUNCE``**).
6. **클래식 ``bounce_state``**(IDLE에서만 진입)·``sig > 1.03 * MA`` 이면 **``agg_state``**.
7. **``idle``**: ``ma_ok`` 아니면 idle 유지. ``ma_ok`` 이면 장벽 ``sig > 1.03 * MA`` → agg, 아니면 bounce.
8. **``defense_state``**(방어): 위 전이 외 추가 룰 없음.

Stress 구간: ``SAFE``(순 safe 100%), ``SAFE_BOUNCE``(safe+반등 혼합). IDLE·BOUNCE·DEFENSE·AGG 는 단일 수량 종목 보유.



MA200: ``sig_asset`` 종가 전 구간 확장 로드 후 롤링 200. 캐시는 CSV + meta JSON
(``full_data_sha256`` 전종가 바이너리 해시, ``window``, 틱컱, 종료일)로 무결성·재현에 사용.

실행 단일구간::

    python3 01_CODE/fsm_four_asset_strategy.py \\
        --sig QQQ --bounce QQQ --defense QLD --agg TQQQ --root 2002-10-01

무작위 윈도(전체 교집합에서 **거래일 ≥ ceil(mc_years×252)** 인 구간을 **mc_iters** 번)::

    python3 01_CODE/fsm_four_asset_strategy.py \\
        --sig QQQ --bounce QQQ --defense QLD --agg TQQQ \\
        --monte-carlo --mc-seed 42

옵션 ``--monte-carlo`` 사용 시 ``--mc-years`` 기본 3년, ``--mc-iters`` 기본 1000.
기본적으로 **RMABS-QLD · RMABS-QQQ** 도 전구간 NAV 슬라이스로 같은 무작위 창과 비교한다.
생략: ``--no-mc-rmabs``.


"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from enum import Enum, auto
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import (  # noqa: E402
    compute_rsi,
    load_extended_daily,
    load_yahoo_daily,
    strategy_rsi_ma_based_switching,
    strategy_rsi_ma_based_switching_qqq_only,
)
from evaluation_metrics import full_metrics  # noqa: E402
from rmabs_gold_simulation import load_gold_series  # noqa: E402

CACHE_SUB = _ROOT / "02_DATA" / "cache" / "ma200"
_OUT_MC = _ROOT / "03_RESULT" / "sensitivity"


def load_daily_extended_or_yahoo(ticker: str) -> pd.DataFrame:
    """yahoo_extended 우선 없으면 yahoo(예: SGOV 단기국채)."""
    t = ticker.upper()
    p = _ROOT / "02_DATA" / "yahoo_extended" / t / f"{t}_daily.csv"
    return load_extended_daily(t) if p.exists() else load_yahoo_daily(t)


class Regime(Enum):
    IDLE = auto()
    BOUNCE = auto()  # 반등 레그 단일 보유 (구 레거시 명칭 DEF)
    DEFENSE = auto()  # 방어 레그 단일 보유 (구 NOR)
    AGG = auto()
    SAFE = auto()
    SAFE_BOUNCE = auto()  # 안전 + 반등 혼합 (구 SAFE_DEF)


MA_WINDOW = 200
MULT_DOWN = 0.97  # 규칙4: 종가 하방 돌파(s_{t-1}≥MULT_DOWN·m_{t-1} → s_t<MULT·m_t)
MULT_BOUNCE_UP_TO_AGG = 1.03  # bounce 레짐에서 sig > 배수·MA → agg
DEFAULT_TRAIL = 0.85
CAPITAL_START = 100_000.0
RSI_PERIOD_DEFAULT = 14
RSI_CROSS_DOWN_DEFAULT = 30.0
StressBleedMode = Literal["RSI_HALF", "MA5_CROSS_MA120_FULL"]


def _core_metrics(nav: pd.Series) -> dict[str, float]:
    """CAGR · MDD · Sharpe · Sortino · Ulcer · 총수익률."""
    m = full_metrics(nav)
    return {
        "cagr": float(m["cagr"]),
        "mdd": float(m["mdd"]),
        "sharpe": float(m["sharpe"]),
        "sortino": float(m["sortino"]),
        "ulcer": float(m["ulcer"]),
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
    }


def benchmark_buy_hold(close: pd.Series, capital: float) -> pd.Series:
    """구간 시작 종가 기준 매수 후 보유(리밸런스 없음)."""
    px = close.astype(float).dropna()
    if len(px) < 2:
        return pd.Series(dtype=float)
    shares = capital / float(px.iloc[0])
    return (shares * px).rename("NAV_bh")


def benchmark_half_qq_tqq(c_qq: pd.Series, c_tq: pd.Series, capital: float) -> pd.Series:
    """초기 50%/50% 각각 매수 후 보유(중간 리밸 없음)."""
    qq = c_qq.astype(float).rename("qq").to_frame().join(c_tq.astype(float).rename("tq"), how="inner")
    if len(qq) < 2:
        return pd.Series(dtype=float)
    half = capital * 0.5
    s_q = half / float(qq["qq"].iloc[0])
    s_t = half / float(qq["tq"].iloc[0])
    nav = (s_q * qq["qq"] + s_t * qq["tq"]).rename("NAV_bh50")
    return nav.loc[nav.index]


def benchmark_panel_for_slice(
    win_ix: pd.DatetimeIndex,
    bench_source: pd.DataFrame,
    capital: float,
) -> dict[str, pd.Series]:
    """
    ``bench_source``: FSM 과 동일 ``Date`` 인덱스 행열
    ``QQQ_Close``, ``QLD_Close``, ``TQQQ_Close``.
    """
    sl = bench_source.reindex(win_ix).dropna(how="any")
    if len(sl) < 2:
        return {}
    ix = sl.index
    cq = sl["QQQ_Close"].loc[ix].astype(float)
    cl = sl["QLD_Close"].loc[ix].astype(float)
    ct = sl["TQQQ_Close"].loc[ix].astype(float)
    return {
        "QQQ_bh": benchmark_buy_hold(cq, capital),
        "QLD_bh": benchmark_buy_hold(cl, capital),
        "mix_qq50_tqq50_bh": benchmark_half_qq_tqq(cq, ct, capital),
        "TQQQ_bh": benchmark_buy_hold(ct, capital),
    }


def _flatten_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": float(v) for k, v in metrics.items()}


def _distribution_block(df_col: pd.Series) -> dict[str, float]:
    x = df_col.astype(float).values
    if len(x) == 0:
        return {}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "p50": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
    }


def summarize_mc_distributions(mc_df: pd.DataFrame) -> dict[str, object]:
    """MC 표 열 패턴별 분포. ``distribution_user_5metrics`` 는 CAGR·MDD·Sharpe·Sortino·Ulcer 만."""
    block_prefixes = {
        "fsm_strategy": "strat_",
        "benchmark_qqq_100_buy_hold": "bh_qqq_",
        "benchmark_qld_100_buy_hold": "bh_qld_",
        "benchmark_mix_qqq50_tqqq50_buy_hold": "bh_mix50_",
        "benchmark_tqqq_100_buy_hold": "bh_tqqq_",
        "rmabs_qld_full_nav_slice": "rmabs_qld_",
        "rmabs_qqq_full_nav_slice": "rmabs_qq_",
    }
    panels: dict[str, dict[str, dict[str, float]]] = {}
    for blk, pref in block_prefixes.items():
        panels[blk] = {}
        for col in sorted(c for c in mc_df.columns if c.startswith(pref)):
            mk = col[len(pref) :]
            panels[blk][mk] = _distribution_block(mc_df[col])

    want = ["cagr", "mdd", "sharpe", "sortino", "ulcer"]
    slim: dict[str, dict[str, dict[str, float]]] = {}
    for blk in block_prefixes:
        slim[blk] = {m: panels[blk][m] for m in want if m in panels[blk]}

    return {
        "distribution_panels": panels,
        "distribution_user_5metrics": slim,
    }


def mc_min_trading_days(years_float: float) -> int:
    """N년 이상 → 최소 거래일 수 (연 252 근사, MA 윈도우보다 길게)."""
    return max(MA_WINDOW + 10, math.ceil(max(years_float, 1e-6) * 252.0))


def run_random_windows(
    aligned_full: pd.DataFrame,
    ma_full: pd.Series,
    bench_source: pd.DataFrame | None,
    *,
    n_years: float,
    n_iters: int,
    rng: np.random.Generator,
    trail_stop: float,
    capital: float,
    nav_rmabs_qld: pd.Series | None,
    nav_rmabs_qqq: pd.Series | None,
    rsi_period: int = RSI_PERIOD_DEFAULT,
    rsi_cross_below: float = RSI_CROSS_DOWN_DEFAULT,
) -> tuple[pd.DataFrame, dict]:
    """무작위 윈도 MC. 동일 창 벤치 + (옵션) RMABS-QLD·RMABS-QQQ 전구간 NAV 슬라이스."""
    ix = aligned_full.index.sort_values()
    n_dates = len(ix)
    lo = mc_min_trading_days(n_years)
    if n_dates < lo:
        raise ValueError(f"교집합 거래일 {n_dates} < 최소 {lo} 일")

    rows: list[dict[str, object]] = []
    for _ in range(n_iters):
        s_pos = int(rng.integers(0, n_dates - lo + 1))
        l_max = n_dates - s_pos
        l_win = int(rng.integers(lo, l_max + 1))
        sl = aligned_full.iloc[s_pos : s_pos + l_win]
        win_ix = sl.index
        mav_sl = ma_full.reindex(win_ix)

        nav, _ev = run_fsm_backtest(
            sl,
            mav_sl,
            trail_stop=trail_stop,
            initial_capital=capital,
            rsi_period=rsi_period,
            rsi_cross_below=rsi_cross_below,
        )
        strat_m = _core_metrics(nav)

        row: dict[str, object] = {
            "window_start": str(pd.Timestamp(win_ix[0]).date()),
            "window_end": str(pd.Timestamp(win_ix[-1]).date()),
            "trading_days": int(l_win),
            "calendar_years_approx": round(
                (pd.Timestamp(win_ix[-1]) - pd.Timestamp(win_ix[0])).days / 365.25, 4
            ),
        }
        row.update(_flatten_metrics("strat", strat_m))

        if bench_source is not None:
            pans = benchmark_panel_for_slice(win_ix, bench_source, capital)
            for pmap in (
                ("QQQ_bh", "bh_qqq"),
                ("QLD_bh", "bh_qld"),
                ("mix_qq50_tqq50_bh", "bh_mix50"),
                ("TQQQ_bh", "bh_tqqq"),
            ):
                k_bm, pref = pmap
                if k_bm not in pans or len(pans[k_bm]) < 2:
                    continue
                row.update(_flatten_metrics(pref, _core_metrics(pans[k_bm])))

        if nav_rmabs_qld is not None and nav_rmabs_qqq is not None:
            sub_ld = nav_rmabs_qld.reindex(win_ix)
            sub_qq = nav_rmabs_qqq.reindex(win_ix)
            if (
                len(sub_ld) >= 2
                and len(sub_qq) >= 2
                and not sub_ld.isna().any()
                and not sub_qq.isna().any()
            ):
                row.update(_flatten_metrics("rmabs_qld", _core_metrics(sub_ld)))
                row.update(_flatten_metrics("rmabs_qq", _core_metrics(sub_qq)))
        rows.append(row)

    df_out = pd.DataFrame(rows)

    distro = summarize_mc_distributions(df_out)
    blob: dict[str, object] = {
        "n_iters": int(n_iters),
        "mc_years_param": float(n_years),
        "min_trading_days_rule": int(lo),
        "full_span_trading_days": int(n_dates),
        "distribution_panels": distro["distribution_panels"],
        "distribution_user_5metrics": distro["distribution_user_5metrics"],
        "benchmarks_documentation": [
            "QQQ 100% buy and hold",
            "QLD 100% buy and hold",
            "50% capital QQQ buy + 50% capital TQQQ buy at window start then hold no rebalance",
            "TQQQ 100% buy and hold",
        ],
        "rmabs_comparison": (
            {
                "method": (
                    "전구간 1회 RMABS 백테스트 NAV를 동일 (start,end) 구간으로 슬라이스 후 "
                    "full_metrics (monte_carlo_bench_dsa_rmabs.py 와 동일한 슬라이스 방식)"
                ),
                "strategies": [
                    "RMABS-QLD: strategy_rsi_ma_based_switching (QQQ 시그널, 방어 레인 QLD·QQQ 규칙)",
                    "RMABS-QQQ: strategy_rsi_ma_based_switching_qqq_only",
                ],
            }
            if nav_rmabs_qld is not None
            else None
        ),
    }
    return df_out, blob


def full_close_sha256(close: pd.Series) -> str:
    """종가 float64 원시 바이너리 SHA256 (캐시 무결성·소스 버전 식별)."""
    vals = np.ascontiguousarray(close.astype(np.float64).values)
    return hashlib.sha256(vals.tobytes()).hexdigest()


def get_or_build_ma200(
    sig_extended: pd.DataFrame,
    sig_ticker: str,
    *,
    force_rebuild: bool = False,
) -> tuple[pd.Series, Path]:
    """
    종가 롤링 MA200. 캐시 CSV + meta JSON.
    메타에는 ``full_data_sha256``, ``window``, ``first_date``, ``last_date``, ``n_rows`` 포함.
    """
    close = sig_extended["Close"].astype(float).sort_index()
    ix = close.index
    ma200 = close.rolling(MA_WINDOW, min_periods=MA_WINDOW).mean().astype(float)
    digest = full_close_sha256(close)
    digest16 = digest[:16]

    CACHE_SUB.mkdir(parents=True, exist_ok=True)
    csv_path = CACHE_SUB / f"{sig_ticker.upper()}_{MA_WINDOW}_{digest16}_ma.csv"
    meta_path = CACHE_SUB / f"{sig_ticker.upper()}_{MA_WINDOW}_{digest16}_meta.json"

    meta: dict[str, object] = {
        "sig_ticker": sig_ticker.upper(),
        "window": MA_WINDOW,
        "full_data_sha256": digest,
        "sha_key_suffix16": digest16,
        "parameter_ma_window": MA_WINDOW,
    }

    if csv_path.exists() and meta_path.exists() and not force_rebuild:
        try:
            cand = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
            cand.index = pd.to_datetime(cand.index).normalize()
            if len(cand) == len(ix) and cand.index.equals(ix):
                return cand["ma200"].astype(float), csv_path
        except Exception:
            pass

    out = pd.DataFrame({"sig_close": close.values, "ma200": ma200.values}, index=ix)
    out.index.name = "Date"
    out.reset_index().to_csv(csv_path, index=False)

    meta["n_rows"] = len(ix)
    meta["first_date"] = str(ix[0].date())
    meta["last_date"] = str(ix[-1].date())
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # 로드 검증 메타 불일치 시 재실행 가능하도록 sha 기록 확인용
    return ma200.astype(float), csv_path


def _nav_eod(
    regime: Regime,
    units: float,
    u_safe: float,
    u_sbounce: float,
    sfx: float,
    bv: float,
    dv: float,
    av: float,
) -> float:
    if regime == Regime.IDLE:
        return units
    if regime == Regime.DEFENSE:
        return units * dv
    if regime == Regime.AGG:
        return units * av
    if regime == Regime.BOUNCE:
        return units * bv
    if regime in (Regime.SAFE, Regime.SAFE_BOUNCE):
        return u_safe * sfx + u_sbounce * bv
    return float("nan")


def run_fsm_backtest(
    aligned: pd.DataFrame,
    ma200: pd.Series,
    *,
    trail_stop: float = DEFAULT_TRAIL,
    initial_capital: float = CAPITAL_START,
    rsi_period: int = RSI_PERIOD_DEFAULT,
    rsi_cross_below: float = RSI_CROSS_DOWN_DEFAULT,
    rsi_override: pd.Series | None = None,
    use_safe_ma_rule: bool = True,
    bootstrap_eod_date: pd.Timestamp | str | None = None,
    bootstrap_capital: float | None = None,
    end_date: pd.Timestamp | str | None = None,
    bootstrap_initial_regime: Literal["AUTO", "DEFENSE"] = "AUTO",
    stress_bleed_mode: StressBleedMode = "RSI_HALF",
    mult_down: float = MULT_DOWN,
    mult_bounce_up_to_agg: float = MULT_BOUNCE_UP_TO_AGG,
    stress_ma_fast: int = 5,
    stress_ma_slow: int = 120,
    merge_agg_into_bounce_trailing: bool = False,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    ``aligned`` 에 ``Close_sig``, ``Close_safe``, ``Close_bounce``, ``Close_defense``, ``Close_agg`` 필요.
    ``rsi_override``: 단위 테스트용 RSI 직렬입력.
    ``use_safe_ma_rule``: True=RMABS-4tier 계열(**안전 레그**는 ``Close_safe``: 금·현금)·규칙4→SAFE·스트레스 블리드; False=MABS(규칙4→BOUNCE 100% 단일종목 레거시).
    ``stress_bleed_mode``: ``RSI_HALF``(기본) 안전 일부 매수 종가는 ``Close_bounce``; DMABS는 MA5>M120 교차 시 **안전 전량→반등 레그**(``Close_bounce``).
    규칙4 발동 조건은 **종가 하방 교차**: 전 거래일 ``sig≥0.97·MA`` 이었고 당일 ``sig<0.97·MA`` 인 경우(처음부터 MA 유효·전전일 포함해
    교차 정의 불가하면 발동 안 함).

    ``bootstrap_eod_date``: 지정 시 해당 **영업일 종가 직후** 포지션을 고정한 뒤 **그다음 영업일부터** 시뮬레이션.
    ``bootstrap_initial_regime`` 가 ``AUTO``(기본): RMABS=``SAFE``, MABS=``BOUNCE``.
    ``DEFENSE``면 종가 방어 레그 **100%** ``regime=DEFENSE``.
    ``bootstrap_capital`` 기본은 ``initial_capital``.
    ``end_date``가 있으면 그 날짜(포함)까지만 루프·산출.
    ``stress_ma_fast``·``stress_ma_slow``: ``MA5_CROSS_MA120_FULL`` 일 때 신호종가 롤링 MA 교차 길이(일).
    ``mult_down``·``mult_bounce_up_to_agg``: 규칙4 하방 배수와 상향 AGG 진입 장벽(기본값은 모듈 상수).
    ``merge_agg_into_bounce_trailing``: True이면 **AGG 레짐 제거**. 상향 신호·트레일은 전부 ``Close_bounce``
    만 사용(종가 ``Close_agg`` 는 열 존재만 필요). 반등 종가 진입 즉시 ``agg_init``/트레일과 동등한 상태로 간주한다.
    """
    required = {"Close_sig", "Close_safe", "Close_bounce", "Close_defense", "Close_agg"}
    miss = required - set(aligned.columns)
    if miss:
        raise ValueError(f"aligned 에 컬럼 필요: {sorted(miss)} — _align_five 사용")

    ix = aligned.index.sort_values()
    mav = ma200.reindex(ix).astype(float)

    regime = Regime.IDLE
    units = float(initial_capital)
    u_safe = 0.0
    u_sbounce = 0.0
    agg_init = 0.0
    agg_high = 0.0
    prev_rsi = float("nan")
    prev_sv = float("nan")
    prev_mah = float("nan")
    prev_ma_ok = False

    sig = aligned["Close_sig"].astype(float)
    csafe = aligned["Close_safe"].astype(float)
    cb = aligned["Close_bounce"].astype(float)
    cdef = aligned["Close_defense"].astype(float)
    cag = aligned["Close_agg"].astype(float)

    rsi_sr = (
        rsi_override.reindex(ix).astype(float)
        if rsi_override is not None
        else compute_rsi(sig, rsi_period).reindex(ix).astype(float)
    )

    ma5_sr = sig.rolling(int(stress_ma_fast), min_periods=int(stress_ma_fast)).mean().reindex(ix).astype(float)
    ma120_sr = sig.rolling(int(stress_ma_slow), min_periods=int(stress_ma_slow)).mean().reindex(ix).astype(float)
    prev_ma5 = float("nan")
    prev_ma120 = float("nan")

    loop_ix = ix
    if bootstrap_eod_date is not None:
        d_boot = pd.Timestamp(bootstrap_eod_date).normalize()
        if d_boot not in ix:
            raise ValueError(f"bootstrap_eod_date {d_boot.date()} 가 aligned 인덱스에 없음")
        cap_b = float(bootstrap_capital) if bootstrap_capital is not None else float(initial_capital)
        bir = str(bootstrap_initial_regime).upper()

        sfx0 = float(csafe.loc[d_boot])
        bv0 = float(cb.loc[d_boot])
        dv0 = float(cdef.loc[d_boot])

        if bir == "DEFENSE":
            if dv0 <= 0:
                raise ValueError(f"{d_boot}: Close_defense<=0 DEFENSE 부트스트랩 불가")
            regime = Regime.DEFENSE
            units = cap_b / dv0
            u_safe = 0.0
            u_sbounce = 0.0
        elif bir == "AUTO":
            if use_safe_ma_rule:
                if sfx0 <= 0:
                    raise ValueError(f"{d_boot}: Close_safe<=0 부트스트랩 불가")
                regime = Regime.SAFE
                u_safe = cap_b / sfx0
                u_sbounce = 0.0
                units = 0.0
            else:
                if bv0 <= 0:
                    raise ValueError(f"{d_boot}: Close_bounce<=0 부트스트랩 불가")
                regime = Regime.BOUNCE
                units = cap_b / bv0
                u_safe = 0.0
                u_sbounce = 0.0
        else:
            raise ValueError(f"bootstrap_initial_regime: {bir!r} — AUTO 또는 DEFENSE 만 허용")
        if merge_agg_into_bounce_trailing and regime == Regime.BOUNCE:
            agg_init = float(bv0)
            agg_high = float(bv0)
        else:
            agg_init = 0.0
            agg_high = 0.0
        mah0 = float(mav.loc[d_boot]) if pd.notna(mav.loc[d_boot]) else float("nan")
        prev_mah = mah0
        prev_ma_ok = not pd.isna(mah0)
        prev_sv = float(sig.loc[d_boot])
        pr0 = float(rsi_sr.loc[d_boot]) if pd.notna(rsi_sr.loc[d_boot]) else float("nan")
        prev_rsi = pr0
        pm5b = ma5_sr.loc[d_boot]
        prev_ma5 = float(pm5b) if pd.notna(pm5b) else float("nan")
        pm12b = ma120_sr.loc[d_boot]
        prev_ma120 = float(pm12b) if pd.notna(pm12b) else float("nan")
        loop_ix = ix[ix > d_boot]
        if end_date is not None:
            d_end = pd.Timestamp(end_date).normalize()
            loop_ix = loop_ix[loop_ix <= d_end]
    elif end_date is not None:
        d_end = pd.Timestamp(end_date).normalize()
        loop_ix = ix[ix <= d_end]

    nav_list: list[float] = []
    ev_rows: list[dict[str, object]] = []

    for d in loop_ix:
        sv = float(sig.loc[d])
        sfx = float(csafe.loc[d])
        bv = float(cb.loc[d])
        dv = float(cdef.loc[d])
        av = float(cag.loc[d])
        mah = float(mav.loc[d]) if pd.notna(mav.loc[d]) else float("nan")
        ma_ok = not pd.isna(mah)
        rsi_t = float(rsi_sr.loc[d]) if pd.notna(rsi_sr.loc[d]) else float("nan")
        m5 = float(ma5_sr.loc[d]) if pd.notna(ma5_sr.loc[d]) else float("nan")
        m12 = float(ma120_sr.loc[d]) if pd.notna(ma120_sr.loc[d]) else float("nan")

        nav_pre = _nav_eod(regime, units, u_safe, u_sbounce, sfx, bv, dv, av)
        transitions: list[str] = []
        regime_prev = regime

        rule4_cross_down = ma_ok and prev_ma_ok and (sv < mult_down * mah) and (
            prev_sv >= mult_down * prev_mah
        )
        rule4_suppress_stress_safe = (
            use_safe_ma_rule
            and stress_bleed_mode != "MA5_CROSS_MA120_FULL"
            and regime in (Regime.SAFE, Regime.SAFE_BOUNCE)
        )
        rule4_fire = rule4_cross_down and not rule4_suppress_stress_safe

        if rule4_fire:
            agg_init = 0.0
            agg_high = 0.0
            tot = nav_pre
            if use_safe_ma_rule:
                if sfx <= 0:
                    raise ValueError(f"{d}: Close_safe<=0 규칙4 불가")
                regime = Regime.SAFE
                u_safe = tot / sfx
                u_sbounce = 0.0
                units = 0.0
                if regime_prev not in (Regime.SAFE, Regime.SAFE_BOUNCE):
                    transitions.append("GLOBAL_LT_097MA_SAFE")
                elif regime_prev == Regime.SAFE_BOUNCE:
                    transitions.append("GLOBAL_LT_097MA_SAFE_FLUSH_MIX")
            else:
                if bv <= 0:
                    raise ValueError(f"{d}: Close_bounce<=0 MABS 규칙4 불가")
                regime = Regime.BOUNCE
                units = tot / bv
                u_safe = 0.0
                u_sbounce = 0.0
                if merge_agg_into_bounce_trailing:
                    agg_init = bv
                    agg_high = bv
                if regime_prev != Regime.BOUNCE:
                    transitions.append("GLOBAL_LT_097MA_BOUNCE")
        elif merge_agg_into_bounce_trailing and regime == Regime.BOUNCE:
            if agg_init > 1e-12:
                agg_high = max(agg_high, bv)
                if bv < agg_init - 1e-12:
                    regime = Regime.DEFENSE
                    units = nav_pre / dv
                    u_safe = 0.0
                    u_sbounce = 0.0
                    agg_init = 0.0
                    agg_high = 0.0
                    transitions.append("BOUNCE_LT_INIT_DEFENSE")
                elif agg_high > 1e-12 and bv < agg_high * trail_stop - 1e-12:
                    regime = Regime.DEFENSE
                    units = nav_pre / dv
                    u_safe = 0.0
                    u_sbounce = 0.0
                    agg_init = 0.0
                    agg_high = 0.0
                    transitions.append("BOUNCE_TRAIL_DEFENSE")
        elif not merge_agg_into_bounce_trailing and regime == Regime.AGG:
            agg_high = max(agg_high, av)
            if av < agg_init - 1e-12:
                regime = Regime.DEFENSE
                units = nav_pre / dv
                u_safe = 0.0
                u_sbounce = 0.0
                agg_init = 0.0
                agg_high = 0.0
                transitions.append("AGG_LT_INIT_DEFENSE")
            elif agg_high > 0 and av < agg_high * trail_stop - 1e-12:
                regime = Regime.DEFENSE
                units = nav_pre / dv
                u_safe = 0.0
                u_sbounce = 0.0
                agg_init = 0.0
                agg_high = 0.0
                transitions.append("AGG_TRAIL_DEFENSE")
        elif regime in (Regime.SAFE, Regime.SAFE_BOUNCE):
            if ma_ok and sv > mult_bounce_up_to_agg * mah:
                if merge_agg_into_bounce_trailing:
                    if bv <= 0:
                        raise ValueError(f"{d}: Close_bounce<=0 BOUNCE(+트레일) 전이 불가")
                    tot_b = nav_pre
                    regime = Regime.BOUNCE
                    units = tot_b / bv
                    agg_init = bv
                    agg_high = bv
                    u_safe = 0.0
                    u_sbounce = 0.0
                    transitions.append("SIG_GT_MULT_MA_TO_BOUNCE_TRAIL")
                else:
                    if av <= 0:
                        raise ValueError(f"{d}: Close_agg<=0 AGG 전이 불가")
                    tot_agg = nav_pre
                    regime = Regime.AGG
                    units = tot_agg / av
                    agg_init = av
                    agg_high = av
                    u_safe = 0.0
                    u_sbounce = 0.0
                    transitions.append("SIG_GT_MULT_MA_TO_AGG")
            else:
                if stress_bleed_mode == "MA5_CROSS_MA120_FULL":
                    ma_dual_ok = (
                        not pd.isna(m5)
                        and not pd.isna(m12)
                        and not pd.isna(prev_ma5)
                        and not pd.isna(prev_ma120)
                    )
                    cross_up_m5_ma120 = ma_dual_ok and (prev_ma5 <= prev_ma120 + 1e-12) and (m5 > m12)
                    if cross_up_m5_ma120 and u_safe > 1e-12 and bv > 0:
                        tot_mx = nav_pre
                        u_sbounce = tot_mx / bv
                        u_safe = 0.0
                        transitions.append("SAFE_MA5_GT_MA120_FULL_BOUNCE")
                    regime = Regime.SAFE_BOUNCE if u_sbounce > 1e-12 else Regime.SAFE
                else:
                    if (
                        not pd.isna(rsi_t)
                        and not pd.isna(prev_rsi)
                        and prev_rsi >= rsi_cross_below
                        and rsi_t < rsi_cross_below
                        and u_safe > 1e-12
                        and bv > 0
                    ):
                        qty_half = u_safe * 0.5
                        proceeds = qty_half * sfx
                        add_bounce = proceeds / bv
                        u_sbounce += add_bounce
                        u_safe -= qty_half
                        transitions.append("SAFE_RSI30_HALVE_TO_BOUNCE")
                    regime = Regime.SAFE_BOUNCE if u_sbounce > 1e-12 else Regime.SAFE
        elif regime == Regime.BOUNCE and ma_ok and sv > mult_bounce_up_to_agg * mah:
            if merge_agg_into_bounce_trailing:
                if bv <= 0:
                    raise ValueError(f"{d}: Close_bounce<=0 BOUNCE 장벽 재시드 불가")
                agg_init = bv
                agg_high = bv
                transitions.append("SIG_GT_MULT_MA_RESEED_BOUNCE_TRAIL")
            else:
                if av <= 0:
                    raise ValueError(f"{d}: Close_agg<=0 AGG 전이 불가")
                regime = Regime.AGG
                units = nav_pre / av
                u_safe = 0.0
                u_sbounce = 0.0
                agg_init = av
                agg_high = av
                transitions.append("SIG_GT_MULT_MA_TO_AGG")
        elif regime == Regime.IDLE:
            if not ma_ok:
                pass
            elif sv > mult_bounce_up_to_agg * mah:
                if merge_agg_into_bounce_trailing:
                    if bv <= 0:
                        raise ValueError(f"{d}: Close_bounce<=0 IDLE→BOUNCE+트레일 불가")
                    regime = Regime.BOUNCE
                    units = nav_pre / bv
                    u_safe = 0.0
                    u_sbounce = 0.0
                    agg_init = bv
                    agg_high = bv
                    transitions.append("IDLE_GT_103MA_BOUNCE_TRAIL")
                else:
                    if av <= 0:
                        raise ValueError(f"{d}: Close_agg<=0 IDLE→AGG 불가")
                    regime = Regime.AGG
                    units = nav_pre / av
                    u_safe = 0.0
                    u_sbounce = 0.0
                    agg_init = av
                    agg_high = av
                    transitions.append("IDLE_GT_103MA_AGG")
            else:
                regime = Regime.BOUNCE
                units = nav_pre / bv
                u_safe = 0.0
                u_sbounce = 0.0
                if merge_agg_into_bounce_trailing:
                    agg_init = bv
                    agg_high = bv
                transitions.append("IDLE_LE_103MA_BOUNCE")

        nav_eod = _nav_eod(regime, units, u_safe, u_sbounce, sfx, bv, dv, av)
        nav_list.append(nav_eod)
        ev_rows.append(
            {
                "Date": d,
                "regime_after": regime.name,
                "transitions": "; ".join(transitions) if transitions else "",
                "nav_eod": float(nav_eod),
                "sig": sv,
                "ma200": mah if ma_ok else None,
                "rsi": float(rsi_t) if not pd.isna(rsi_t) else None,
                "u_safe": float(u_safe),
                "u_stress_bounce": float(u_sbounce),
                "agg_init": agg_init,
                "agg_high": agg_high,
            }
        )
        if not pd.isna(rsi_t):
            prev_rsi = rsi_t
        prev_sv = sv
        prev_mah = mah
        prev_ma_ok = ma_ok
        prev_ma5 = m5
        prev_ma120 = m12

    nav_sr = pd.Series(nav_list, index=loop_ix, name="FSM_NAV")
    return nav_sr, pd.DataFrame(ev_rows)


def _align_five(
    sig: pd.DataFrame,
    safe_df: pd.DataFrame,
    bounce_df: pd.DataFrame,
    defense_df: pd.DataFrame,
    agg_df: pd.DataFrame,
) -> pd.DataFrame:
    ix = (
        sig.index.intersection(safe_df.index)
        .intersection(bounce_df.index)
        .intersection(defense_df.index)
        .intersection(agg_df.index)
    )
    ix = ix.sort_values()
    return pd.DataFrame(
        {
            "Close_sig": sig.loc[ix, "Close"].astype(float),
            "Close_safe": safe_df.loc[ix, "Close"].astype(float),
            "Close_bounce": bounce_df.loc[ix, "Close"].astype(float),
            "Close_defense": defense_df.loc[ix, "Close"].astype(float),
            "Close_agg": agg_df.loc[ix, "Close"].astype(float),
        },
        index=ix,
    )


def _align_four(
    sig: pd.DataFrame,
    bounce_df: pd.DataFrame,
    defense_df: pd.DataFrame,
    agg_df: pd.DataFrame,
) -> pd.DataFrame:
    """하위호환: 안전 레그 종가를 반등 레그와 동일 시계열로 둠(레거시 MABS)."""
    return _align_five(sig, bounce_df, bounce_df, defense_df, agg_df)


def build_bench_source_for_index(ix: pd.DatetimeIndex) -> pd.DataFrame:
    """QQQ · QLD · TQQQ 종가, FSM ``aligned_full`` 과 동일 인덱스(결측 있으면 오류)."""
    qq = load_extended_daily("QQQ")
    ql = load_extended_daily("QLD")
    tq = load_extended_daily("TQQQ")
    ix_sorted = ix.sort_values()
    out = pd.DataFrame(
        {
            "QQQ_Close": qq["Close"].reindex(ix_sorted).astype(float),
            "QLD_Close": ql["Close"].reindex(ix_sorted).astype(float),
            "TQQQ_Close": tq["Close"].reindex(ix_sorted).astype(float),
        },
        index=ix_sorted,
    )
    if out.isna().any().any():
        cols = list(out.columns[out.isna().any()])
        raise ValueError(
            f"벤치마크 종가 결측({cols}) — FSM 구간 일자에 대해 QQQ·QLD·TQQQ 가격 존재 여부 확인"
        )
    return out


def _print_user_metric_row(title: str, block: dict[str, dict[str, float]]) -> None:
    """``block``: metric 이름 → 분포 통계(dict). 출력은 평균값만."""
    def g(k: str) -> float:
        return float(block[k]["mean"]) if k in block else float("nan")

    print(
        f"  {title:28s}"
        f"  CAGR평균={g('cagr') * 100:7.2f}%"
        f"  MDD평균={g('mdd') * 100:7.2f}%"
        f"  Sharpe≈{g('sharpe'):6.3f}"
        f"  Sortino≈{g('sortino'):6.3f}"
        f"  Ulcer≈{g('ulcer'):7.4f}"
    )


def print_mc_distribution_user_summary(blob: dict[str, object]) -> None:
    u5 = blob.get("distribution_user_5metrics")
    if not isinstance(u5, dict):
        print("(distribution_user_5metrics 없음)")
        return
    pairs = (
        ("FSM 전략", "fsm_strategy"),
        ("RMABS-QLD (전구간NAV 슬라이스)", "rmabs_qld_full_nav_slice"),
        ("RMABS-QQQ (전구간NAV 슬라이스)", "rmabs_qqq_full_nav_slice"),
        ("QQQ 100% buy&hold", "benchmark_qqq_100_buy_hold"),
        ("QLD 100% buy&hold", "benchmark_qld_100_buy_hold"),
        ("QQQ50/TQQQ50 초기 매수후 보유", "benchmark_mix_qqq50_tqqq50_buy_hold"),
        ("TQQQ 100% buy&hold", "benchmark_tqqq_100_buy_hold"),
    )
    print("무작위 윈도별 지표 평균(MC 각 시행 1값 → 시행 간 평균):")
    for title, k in pairs:
        blk = u5.get(k)
        if isinstance(blk, dict) and blk:
            _print_user_metric_row(title, blk)


def print_single_slice_metrics(nav: pd.Series, bench_source: pd.DataFrame, ix: pd.DatetimeIndex) -> None:
    """단일 구간: 전략 + 4벤치 동일 칼럼 5지표."""
    strat = _core_metrics(nav)
    print(
        "  FSM 전략"
        f"  CAGR={strat['cagr'] * 100:7.2f}%"
        f"  MDD={strat['mdd'] * 100:7.2f}%"
        f"  Sharpe={strat['sharpe']:6.3f}"
        f"  Sortino={strat['sortino']:6.3f}"
        f"  Ulcer={strat['ulcer']:7.4f}"
    )
    pans = benchmark_panel_for_slice(ix, bench_source, CAPITAL_START)
    labels = (
        ("QQQ 100% buy&hold", "QQQ_bh"),
        ("QLD 100% buy&hold", "QLD_bh"),
        ("QQQ50/TQQQ50 초기 매수후 보유", "mix_qq50_tqq50_bh"),
        ("TQQQ 100% buy&hold", "TQQQ_bh"),
    )
    for tit, kk in labels:
        s = pans.get(kk)
        if s is None or len(s) < 2:
            print(f"  {tit}: (데이터 부족)")
            continue
        m = _core_metrics(s)
        print(
            f"  {tit:28s}"
            f"  CAGR={m['cagr'] * 100:7.2f}%"
            f"  MDD={m['mdd'] * 100:7.2f}%"
            f"  Sharpe={m['sharpe']:6.3f}"
            f"  Sortino={m['sortino']:6.3f}"
            f"  Ulcer={m['ulcer']:7.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="4자산 FSM 백테스트")
    ap.add_argument("--sig", required=True)
    ap.add_argument(
        "--safe",
        dest="safe_tk",
        default="GOLD",
        help="안전자산 티커 또는 GOLD/GC=F/GLD(금 우선)·그 외는 yahoo_extended/yahoo(SGOV 등)",
    )
    ap.add_argument(
        "--bounce",
        dest="bounce_tk",
        required=True,
        help="반등 레그 티커(구 레거시: --def)",
    )
    ap.add_argument(
        "--defense",
        dest="defense_tk",
        required=True,
        help="방어 레그 티커(구 레거시: --nor)",
    )
    ap.add_argument("--agg", dest="agg_tk", required=True)
    ap.add_argument("--root", default="2002-10-01")
    ap.add_argument("--trail", type=float, default=DEFAULT_TRAIL)
    ap.add_argument("--rsi-period", type=int, default=RSI_PERIOD_DEFAULT)
    ap.add_argument("--rsi-cross", type=float, default=RSI_CROSS_DOWN_DEFAULT, help="RSI 하방 교차 레벨")
    ap.add_argument("--force-ma-rebuild", action="store_true")
    ap.add_argument(
        "--monte-carlo",
        action="store_true",
        help="전체 교집합 무작위 윈도(mc-years·mc-iters 반복)",
    )
    ap.add_argument("--mc-years", type=float, default=3.0, help="최소 윈도 거래연(252일/연)")
    ap.add_argument("--mc-iters", type=int, default=1000, help="무작위 윈도 시행 수")
    ap.add_argument("--mc-seed", type=int, default=42)
    ap.add_argument(
        "--mc-out",
        type=str,
        default="",
        help="저장 JSON 경로 비우면 자동 이름",
    )
    ap.add_argument(
        "--no-mc-rmabs",
        action="store_true",
        help="Monte Carlo 시 RMABS-QLD·RMABS-QQQ 비교 생략(샘플링은 FSM 원교집합만)",
    )
    args = ap.parse_args()

    sg = load_extended_daily(args.sig)
    bg = load_extended_daily(args.bounce_tk)
    dg = load_extended_daily(args.defense_tk)
    ag = load_extended_daily(args.agg_tk)

    raw_safe = args.safe_tk.strip()
    stk = raw_safe.upper().replace("GC-F", "GC=F")
    ix4_core = (
        sg.index.intersection(bg.index).intersection(dg.index).intersection(ag.index).sort_values()
    )
    if ix4_core.empty:
        raise SystemExit("sig·bounce·defense·agg 교집합 비어 있음")
    if stk in {"GOLD", "GC=F", "GLD"}:
        g0 = pd.Timestamp(ix4_core.min())
        g1 = pd.Timestamp(ix4_core.max())
        sf, _ = load_gold_series(g0, g1)
    else:
        sf = load_daily_extended_or_yahoo(raw_safe)

    ma_full, csv_cache = get_or_build_ma200(
        sg,
        args.sig.upper(),
        force_rebuild=args.force_ma_rebuild,
    )
    print(f"MA200 캐시: {csv_cache}")

    aligned_full = _align_five(sg, sf, bg, dg, ag)
    mav_full = ma_full.reindex(aligned_full.index)

    if args.monte_carlo:
        lo_need = mc_min_trading_days(args.mc_years)
        if len(aligned_full) < lo_need:
            raise SystemExit(
                f"전체 교집합 거래일 {len(aligned_full)} < 최소 무작위 윈도 {lo_need}"
            )

        nav_rm_ld: pd.Series | None = None
        nav_rm_qq: pd.Series | None = None
        aligned_mc = aligned_full
        mav_mc = mav_full

        if not args.no_mc_rmabs:
            qqq_b = load_extended_daily("QQQ")
            qld_b = load_extended_daily("QLD")
            tqq_b = load_extended_daily("TQQQ")
            ix_tri = (
                aligned_full.index.intersection(qqq_b.index)
                .intersection(qld_b.index)
                .intersection(tqq_b.index)
                .sort_values()
            )
            aligned_mc = aligned_full.reindex(ix_tri).dropna(how="any")
            mav_mc = ma_full.reindex(aligned_mc.index)
            if len(aligned_mc) < lo_need:
                raise SystemExit(
                    f"RMABS 비교용 QQQ·QLD·TQQQ 교집합 거래일 {len(aligned_mc)} < 최소 윈도 {lo_need}"
                )
            if len(aligned_mc) < len(aligned_full):
                print(
                    f"참고: RMABS·벤치는 QQQ/QLD/TQQQ 일자만 사용 → "
                    f"MC 샘플링 {len(aligned_full)}일 → {len(aligned_mc)}일"
                )
            Qa = qqq_b.loc[aligned_mc.index]
            La = qld_b.loc[aligned_mc.index]
            Ta = tqq_b.loc[aligned_mc.index]
            nav_rm_ld, _ = strategy_rsi_ma_based_switching(
                Qa, La, Ta, CAPITAL_START, series_name="RMABS-QLD"
            )
            nav_rm_qq, _ = strategy_rsi_ma_based_switching_qqq_only(
                Qa, La, Ta, CAPITAL_START, series_name="RMABS-QQQ"
            )

        rng = np.random.default_rng(args.mc_seed)
        bench_aligned = build_bench_source_for_index(aligned_mc.index)
        df_mc, summary = run_random_windows(
            aligned_mc,
            mav_mc,
            bench_aligned,
            n_years=args.mc_years,
            n_iters=args.mc_iters,
            rng=rng,
            trail_stop=args.trail,
            capital=CAPITAL_START,
            nav_rmabs_qld=nav_rm_ld,
            nav_rmabs_qqq=nav_rm_qq,
            rsi_period=args.rsi_period,
            rsi_cross_below=args.rsi_cross,
        )

        fx0 = pd.Timestamp(aligned_mc.index[0]).strftime("%Y%m%d")
        fx1 = pd.Timestamp(aligned_mc.index[-1]).strftime("%Y%m%d")
        rmabs_tag = "" if args.no_mc_rmabs else "_rmabs"
        out_name = (
            f"fsm_mc{rmabs_tag}_{args.sig}_s{args.safe_tk}_{args.bounce_tk}_{args.defense_tk}_{args.agg_tk}_"
            f"{fx0}_{fx1}_{args.mc_years:g}yr_n{args.mc_iters}_seed{args.mc_seed}.json"
        )
        out_path = Path(args.mc_out) if args.mc_out.strip() else _OUT_MC / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        csv_side = out_path.with_name(out_path.stem + "_windows.csv")

        blob = dict(summary)
        blob["tickers"] = {
            "sig": args.sig,
            "safe": args.safe_tk,
            "bounce": args.bounce_tk,
            "defense": args.defense_tk,
            "agg": args.agg_tk,
        }
        blob["rsi_period"] = int(args.rsi_period)
        blob["rsi_cross_below"] = float(args.rsi_cross)
        blob["trail_stop"] = float(args.trail)
        blob["capital"] = float(CAPITAL_START)
        blob["mc_include_rmabs_nav_slice"] = not args.no_mc_rmabs
        blob["full_span_dates"] = [
            str(aligned_mc.index[0].date()),
            str(aligned_mc.index[-1].date()),
        ]
        blob["monte_carlo_aligned_trading_days"] = int(len(aligned_mc))
        blob["fsm_four_asset_aligned_trading_days"] = int(len(aligned_full))
        blob["windows_csv_abs"] = str(csv_side.resolve())
        df_mc.to_csv(csv_side, index=False)

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2, ensure_ascii=False)

        print(
            f"Monte Carlo: 샘플 구간 {aligned_mc.index[0].date()} ~ {aligned_mc.index[-1].date()} "
            f"거래일 {len(aligned_mc)} "
            f"(FSM 원 4종 교집합 {len(aligned_full)}일) | "
            f"윈도 최소 거래일 {mc_min_trading_days(args.mc_years)}일 이상"
            + (" | RMABS 슬라이스 비교 포함" if not args.no_mc_rmabs else " | RMABS 비교 생략")
        )
        print(f"시행 수 {args.mc_iters}  seed={args.mc_seed}")
        print_mc_distribution_user_summary(blob)
        print(f"윈도 표: {csv_side}")
        print(f"요약 JSON: {out_path}")
        return

    root = pd.Timestamp(args.root.strip())
    aligned = aligned_full[aligned_full.index >= root]
    if len(aligned) < MA_WINDOW:
        raise SystemExit("교집합 거래일이 MA200 형성보다 짧음")

    mav = ma_full.reindex(aligned.index)

    nav, ev = run_fsm_backtest(
        aligned,
        mav,
        trail_stop=args.trail,
        initial_capital=CAPITAL_START,
        rsi_period=args.rsi_period,
        rsi_cross_below=args.rsi_cross,
    )

    bench_slice = build_bench_source_for_index(aligned.index)
    print(f"기간 {aligned.index[0].date()} ~ {aligned.index[-1].date()}  일수={len(aligned)}")
    print("지표[CAGR · MDD · Sharpe · Sortino · Ulcer] — 전략 및 벤치(동일 일자)·벤치 정의는 JSON/MC와 동일")
    print_single_slice_metrics(nav, bench_slice, aligned.index)

    ret = nav.iloc[-1] / nav.iloc[0] - 1.0
    print(f"종료 NAV={nav.iloc[-1]:,.2f} (초기={CAPITAL_START:,.2f})  총수익={ret*100:.2f}%")
    chg = ev[ev["transitions"].str.len() > 0]
    if not chg.empty:
        print(chg[["Date", "regime_after", "transitions"]].tail(24).to_string(index=False))


if __name__ == "__main__":
    main()
