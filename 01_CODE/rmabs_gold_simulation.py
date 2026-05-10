"""RMABS-GOLD 전구간 1회: 방어 레인 금 가격(Yahoo Finance COMEX GC=F 우선).

Yahoo 의 GC=F 일봉은 약 2000-08-30부터만 제공된다(TradingView 의 GC1! 등 긴 연속선과 다름).
1999 등 그 이전 구간은 **같은 야후 소스의 ^XAU(필라델피아 금·은 지수 성격)** 일간 변동을 이용해,
**첫 GC=F 거래일 t0 에서 종가 레벨만 일치시키고** 역으로 이전 날들을 스케일한다:
``close(d) = GC_F(t0) * ^XAU(d) / ^XAU(t0)`` (d < t0). COMEX 연속선과 동일 시계열은 아니다.

ETF 상장일 제한 회피를 위해 금은 선물 종가 위주 사용, GC=F 불가 시 GLD 폴백.

시그널·규칙은 RMABS-QLD와 동일(신호=QQQ MA200·RSI, QLD/TQQ 레인).

두 전략 모두 전역 옵션 ``warmup_hold_cash=True``: MA200 형성 후 **첫 체결 이벤트** 전까지는
방어종목 매수 없이 **현금만**(이자 미반영·NAV=초기자본).

실행:
  python3 01_CODE/rmabs_gold_simulation.py [--root 1999-03-10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

from backtest_switching import (  # noqa: E402
    load_extended_daily,
    strategy_rsi_ma_based_switching,
    strategy_rsi_ma_based_switching_gold,
)
from evaluation_metrics import fmt_metrics_row, full_metrics  # noqa: E402

OUT_DIR = _ROOT / "03_RESULT" / "sensitivity"
CAP = 100_000.0
WIN_TRADING = 252
MIN_GAP_TRADING = 126


def _yf_history_close(tkr: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.Series:
    s = yf.Ticker(tkr).history(
        start=str(start_ts.date()),
        end=str((end_ts + pd.Timedelta(days=1)).date()),
        auto_adjust=True,
        interval="1d",
        actions=False,
    )
    if s.empty:
        return pd.Series(dtype=float)
    dt = pd.to_datetime(s.index)
    try:
        if getattr(dt, "tz", None) is not None:
            dt = dt.tz_convert(None)
    except (TypeError, AttributeError):
        pass
    s = s.rename_axis("Date")
    s.index = pd.DatetimeIndex(dt.normalize())
    out = s["Close"].astype(float).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _gc_series_with_xau_preface(
    gc: pd.Series,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> tuple[pd.Series, str]:
    """GC=F 가 start_ts 보다 늦게 시작할 때만, 첫 유효 GC 날에서 ^XAU 레벨을 맞춘 접두 일봉을 붙임."""
    if gc.empty:
        return gc, ""
    t0 = gc.index.min()
    if pd.Timestamp(start_ts).normalize() >= t0.normalize():
        return gc, ""

    xa = _yf_history_close("^XAU", start_ts, end_ts)
    if xa.empty or t0 not in xa.index:
        return gc, ""

    g0 = float(gc.loc[t0])
    xa0 = float(xa.loc[t0])
    if xa0 == 0.0 or np.isnan(xa0):
        return gc, ""

    pre = xa.loc[(xa.index >= pd.Timestamp(start_ts).normalize()) & (xa.index < t0)].astype(float)
    if pre.empty:
        return gc, ""

    scaled = pre * (g0 / xa0)
    merged = pd.concat([scaled, gc]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    note = (
        f"GC=F (COMEX) from {t0.date()}; preceding dates use "
        "^XAU anchored to first GC=F close (approx. pre-contract Yahoo window — not TradingView GC1!)"
    )
    return merged, note


def load_gold_series(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    """COMEX 금 GC=F 우선. GC=F 시작 이전 요청 구간에는 ^XAU 앵커 접두 가능. 불가 시 GLD."""
    futures_meta = "GC=F (COMEX Gold Continuous Contract)"
    ser = _yf_history_close("GC=F", start_ts, end_ts)
    meta = futures_meta

    if ser.empty:
        ser = _yf_history_close("GLD", start_ts, end_ts)
        meta = "GLD (SPDR Gold Shares, fallback)"
    else:
        ser, pre_note = _gc_series_with_xau_preface(ser, start_ts, end_ts)
        if pre_note:
            meta = pre_note

    if ser.empty:
        raise SystemExit("Yahoo Finance에서 GC=F·GLD 금 가격 시계열을 받지 못했습니다.")

    c = ser.astype(float)
    g = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c}, index=ser.index)
    return g.sort_index(), meta


def _events_between(ev: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    if ev.empty or len(ev.columns) == 0:
        return ev
    dcol = pd.to_datetime(ev["Date"])
    return ev.loc[(dcol >= t0) & (dcol <= t1)]


def pick_divergence_episodes(
    nav_qld_def: pd.Series,
    nav_gold: pd.Series,
    *,
    win: int,
    min_gap_days: int,
    k_episodes: int,
) -> list[dict]:
    """두 NAV 비율 로그변화량 |Δln(NAV금/NAVQLD방어)| 의 상위 비중첩 없는 창."""
    idx = nav_qld_def.index
    rg = nav_gold.astype(float).values
    rl = nav_qld_def.astype(float).values
    lr = np.log(np.maximum(rg, 1e-12) / np.maximum(rl, 1e-12))

    n = len(lr)
    candidates: list[tuple[float, int, float]] = []
    for s in range(0, max(0, n - win)):
        e = s + win
        dlt = float(lr[e] - lr[s])
        candidates.append((abs(dlt), s, dlt))

    candidates.sort(reverse=True, key=lambda x: x[0])

    spans: list[tuple[int, int, float]] = []
    for _abs_mag, s, dlt in candidates:
        e = s + win
        if any(not (e < es - min_gap_days or s > ee + min_gap_days) for es, ee, _ in spans):
            continue
        spans.append((s, e, dlt))
        if len(spans) >= k_episodes:
            break

    spans.sort(key=lambda x: idx[x[0]])
    episodes: list[dict] = []
    for si, ei, dlt_log in spans:
        t0, t1 = idx[si], idx[ei]
        ret_g = float(nav_gold.iloc[ei] / nav_gold.iloc[si] - 1.0)
        ret_q = float(nav_qld_def.iloc[ei] / nav_qld_def.iloc[si] - 1.0)
        episodes.append(
            {
                "start": str(t0.date()),
                "end": str(t1.date()),
                "days": win,
                "delta_ln_nav_ratio_gold_over_qld": dlt_log,
                "period_return_RM_abs_QLD_defqqq_pct": ret_q * 100.0,
                "period_return_RM_abs_GOLD_pct": ret_g * 100.0,
                "spread_gold_minus_qld_pp": (ret_g - ret_q) * 100.0,
            }
        )
    return episodes


def _events_records(ev: pd.DataFrame) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    if ev is None or ev.empty:
        return out
    for _, r in ev.iterrows():
        out.append(
            {
                "Date": str(pd.Timestamp(r["Date"]).date()),
                "type": str(r["type"]),
                "value": float(r["value"]) if pd.notna(r.get("value")) else None,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="1999-03-10")
    ap.add_argument(
        "--episode-k",
        type=int,
        default=8,
        help="252거래일 창에서 두 전략 NAV비 로그변화량 상위 에피소드 개수(비중첩)",
    )
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    q_raw = load_extended_daily("QQQ")
    ql_raw = load_extended_daily("QLD")
    t_raw = load_extended_daily("TQQQ")
    ix0 = q_raw.index.intersection(ql_raw.index).intersection(t_raw.index).sort_values()
    root_ts = pd.Timestamp(args.root.strip())
    ix0 = ix0[ix0 >= root_ts]
    Q0, L0, T0 = q_raw.loc[ix0], ql_raw.loc[ix0], t_raw.loc[ix0]

    gold_df, gold_note = load_gold_series(ix0[0], ix0[-1])
    common = ix0.intersection(gold_df.index).sort_values()
    if len(common) < 400:
        raise SystemExit(
            f"금·주식 교집합 거래일 {len(common)}일 — 너무 짧음. "
            "데이터 빈틈 또는 --root가 너무 이른 경우. 필요 시 금 원천 확인(GC=F/GLD)."
        )

    Q, L, T, G = Q0.loc[common], L0.loc[common], T0.loc[common], gold_df.loc[common]

    nav_qld_def, ev_qq = strategy_rsi_ma_based_switching(
        Q,
        L,
        T,
        CAP,
        series_name="RMABS-QLD(ref)",
        warmup_hold_cash=True,
    )
    nav_gold, ev_gold = strategy_rsi_ma_based_switching_gold(Q, G, L, T, CAP, warmup_hold_cash=True)

    mq, mg = full_metrics(nav_qld_def), full_metrics(nav_gold)

    print("=" * 88)
    print("RMABS-GOLD 전구간 1회 비교  [warmup: 현금 보유까지 첫 이벤트 대기]")
    print(f"  금 데이터: Yahoo Finance — {gold_note}")
    print(f"  백테스트 공통 교집합: {common[0].date()} ~ {common[-1].date()}  ({len(common)} 거래일)")
    print(f"  (QQQ 확장 시작 {ix0[0].date()} 대비, 금·주식 교집합으로 앞쪽 {len(ix0)-len(common)}일 제외 가능)")
    print("=" * 88)
    print(fmt_metrics_row("RMABS-QLD (방어=QQQ)", mq))
    print(fmt_metrics_row("RMABS-GOLD (방어=GC=F 노출)", mg))

    tot_q = float(nav_qld_def.iloc[-1] / nav_qld_def.iloc[0] - 1.0)
    tot_g = float(nav_gold.iloc[-1] / nav_gold.iloc[0] - 1.0)
    pp_spread = (tot_g - tot_q) * 100.0
    print("-" * 88)
    print(
        f"  추가(전량): 총수익률  QQQ방어방={tot_q*100:.2f}%  "
        f"GOLD방어방={tot_g*100:.2f}%  스프레드(GOLD−QQQ방)={pp_spread:+.2f} pp"
    )
    print(
        f"  보조 메트릭: avg_dd (% wealth 기준 분수) "
        f"QQQ방={mq['avg_dd']*100:.4f}%  GOLD={mg['avg_dd']*100:.4f}%  "
        f"| 일 VaR95 대략적 QQQ방={mq['var95']*100:.4f}% GOLD={mg['var95']*100:.4f}%"
    )

    print("-" * 88)
    print(f"체결 이벤트 수  방어=QQQ={len(ev_qq)} | 방어금(RMABSG)={len(ev_gold)}")
    if not ev_gold.empty:
        print("\n[RMABSG 이벤트 유형 빈도]")
        print(ev_gold["type"].value_counts().to_string())

    ek = max(1, args.episode_k)
    episodes = pick_divergence_episodes(
        nav_qld_def,
        nav_gold,
        win=WIN_TRADING,
        min_gap_days=MIN_GAP_TRADING,
        k_episodes=ek,
    )
    eps_json: list[dict] = []
    print(
        "\n"
        + "=" * 88
        + "\nΔ가 극적인 에피소드 (약 252거래일 창 × "
        + str(ek)
        + ", 비중첩 최소 "
        + str(MIN_GAP_TRADING)
        + "거래일 간격)"
    )
    print(
        "|Δln(NAV_GOLD/NAV_QLD방)| 상위 순이며 스프레드는 구간(GOLD%) − 구간(QQQ방%)입니다. "
        "같은 체결일은 RMABS·RMABSG를 함께 표시합니다."
    )
    for rk, ep in enumerate(episodes, start=1):
        t0 = pd.Timestamp(ep["start"])
        t1 = pd.Timestamp(ep["end"])
        e_rmabs = _events_between(ev_qq, t0, t1)
        e_g = _events_between(ev_gold, t0, t1)
        print(f"\n--- 에피소드 #{rk}: {ep['start']} → {ep['end']} (~{WIN_TRADING} 거래일) ---")
        print(f"  |Δln(NAV_GOLD/NAV_QQQ방어)|={abs(ep['delta_ln_nav_ratio_gold_over_qld']):.4f}")
        print(
            f"  구간총수익(%): QQQ방어 RM={ep['period_return_RM_abs_QLD_defqqq_pct']:+.2f}  "
            f"GOLD방어={ep['period_return_RM_abs_GOLD_pct']:+.2f}  스프레드={ep['spread_gold_minus_qld_pp']:+.2f} pp"
        )
        chronological: list[tuple[pd.Timestamp, str, object, float]] = []
        for _, xr in e_rmabs.iterrows():
            chronological.append(
                (
                    pd.Timestamp(xr["Date"]).normalize(),
                    "RMABS",
                    xr["type"],
                    float(xr["value"]),
                )
            )
        for _, xg in e_g.iterrows():
            chronological.append(
                (
                    pd.Timestamp(xg["Date"]).normalize(),
                    "RMABSG",
                    xg["type"],
                    float(xg["value"]),
                )
            )
        chronological.sort(key=lambda z: z[0])
        udays = len({z[0].date() for z in chronological})
        print(
            f"  체결 이벤트 행: RMABS {len(e_rmabs)}건 | RMABSG {len(e_g)}건 | 발생일 {udays}일"
        )
        for dts, tag, typ, val in chronological:
            print(f"    {dts.date()}  {tag:6s}  {typ!s}  value={val:,.2f}")
        eps_json.append(
            {
                **ep,
                "events_RMABS": _events_records(e_rmabs),
                "events_RMABSG": _events_records(e_g),
            }
        )

    def_metrics = {k: float(mq[k]) for k in mq}
    def_metrics["total_return"] = tot_q
    gold_metrics = {k: float(mg[k]) for k in mg}
    gold_metrics["total_return"] = tot_g

    outp = OUT_DIR / f"rmabs_gold_vs_qld_{common[0].strftime('%Y%m%d')}_{common[-1].strftime('%Y%m%d')}.json"
    blob = {
        "gold_source": gold_note,
        "root": args.root.strip(),
        "common_period": [str(common[0].date()), str(common[-1].date())],
        "n_days": int(len(common)),
        "capital": CAP,
        "warmup_hold_cash": True,
        "rma_bs_qld_defqqq_metrics": def_metrics,
        "rmabs_gold_metrics": gold_metrics,
        "events_counts": {"RMABS": int(len(ev_qq)), "RMABSG": int(len(ev_gold))},
        "divergence_episodes": {
            "window_trading_days": WIN_TRADING,
            "min_gap_trading_between": MIN_GAP_TRADING,
            "k_requested": ek,
            "episodes": eps_json,
        },
    }
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, ensure_ascii=False)
    print("\n저장:", outp)


if __name__ == "__main__":
    main()
