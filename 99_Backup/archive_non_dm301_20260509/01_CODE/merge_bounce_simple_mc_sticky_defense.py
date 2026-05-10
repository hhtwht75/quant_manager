#!/usr/bin/env python3
"""
병합 FSM 단일 파일 (sticky DEFENSE 변형).

merge_bounce_simple_mc.py 와 동일하나 **DEFENSE 상태에서는 반등으로 직접 전이하지 않는다**.
손절·트레일로 방어에 들어간 뒤에는 **규칙4(0.97×MA 하방 돌파)** 로만 SAFE 로 나간다.

나머지: SAFE·SAFE_BOUNCE 진입 반등 등 **QQQ 종가가 1.03×MA200 선을 종가 상향 돌파**할 때만(전일 선 이하) 해당 전이 허용, MA5>M120 교차, 규칙4, 트레일 로직 등.

  python3 01_CODE/merge_bounce_simple_mc_sticky_defense.py \\
    --root 1999-03-10 --mc-years 3 --mc-iters 3000 --mc-seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
_ROOT = _CODE.parent
sys.path.insert(0, str(_CODE))

import merge_bounce_simple_mc as mb  # noqa: E402 — _cross_up_103_ma 재사용

from backtest_switching import load_extended_daily  # noqa: E402
from evaluation_metrics import full_metrics  # noqa: E402
from rmabs_gold_simulation import load_gold_series  # noqa: E402

MA_WIN = 200
MULT_DN = 0.97
TRAIL_DEFAULT = 0.85
CAP0 = 100_000.0
MA_FAST, MA_SLOW = 5, 120
SIG_TICKER = "QQQ"

IDLE, BOUNCE, DEFENSE, SAFE, SAFE_BOUNCE = range(5)


def mc_min_days(years: float) -> int:
    return max(MA_WIN + 10, math.ceil(max(years, 1e-9) * 252.0))


def ma_long(close: pd.Series, win: int | None = None) -> pd.Series:
    w = MA_WIN if win is None else int(win)
    if w < 2:
        raise ValueError(w)
    return close.astype(float).rolling(w, min_periods=w).mean()


def ma200(close: pd.Series) -> pd.Series:
    return ma_long(close, MA_WIN)


def _nav(r: int, units: float, u_safe: float, u_sb: float, sfx: float, bv: float, dv: float) -> float:
    if r == IDLE:
        return units
    if r == DEFENSE:
        return units * dv
    if r == BOUNCE:
        return units * bv
    if r in (SAFE, SAFE_BOUNCE):
        return u_safe * sfx + u_sb * bv
    raise AssertionError(f"unexpected regime {r}")


def pick(t: str, sg: pd.DataFrame, ql: pd.DataFrame, tg: pd.DataFrame, gld: pd.DataFrame) -> pd.Series:
    u = t.strip().upper()
    if u == "QQQ":
        return sg["Close"].astype(float)
    if u == "QLD":
        return ql["Close"].astype(float)
    if u == "TQQQ":
        return tg["Close"].astype(float)
    if u == "GOLD":
        return gld["Close"].astype(float)
    raise ValueError(f"지원 틱: QQQ, QLD, TQQQ, Gold — 입력: {t!r}")


def _ma_dual_ok(
    m5: float, m12: float, prev_m5: float, prev_m120: float
) -> bool:
    return (
        not any(map(math.isnan, (m5, m12, prev_m5, prev_m120)))
        and prev_m5 <= prev_m120 + 1e-12
        and m5 > m12
    )


def run_merge_fsm(
    sig: pd.Series,
    safe: pd.Series,
    bounce: pd.Series,
    defense: pd.Series,
    ma: pd.Series,
    *,
    trail: float = TRAIL_DEFAULT,
    capital: float = CAP0,
    mult_dn: float | None = None,
    mult_up: float | None = None,
    ma_fast: int | None = None,
    ma_slow: int | None = None,
) -> pd.Series:
    """일봉 순회 → NAV 시계열. DEFENSE는 규칙4로만 SAFE 이탈 (Bounce 직행 없음)."""
    md = MULT_DN if mult_dn is None else float(mult_dn)
    mu = mb.MULT_UP if mult_up is None else float(mult_up)
    mf = MA_FAST if ma_fast is None else int(ma_fast)
    ms = MA_SLOW if ma_slow is None else int(ma_slow)
    if mf <= 1 or ms <= 1 or mf >= ms:
        raise ValueError(f"require 1 < ma_fast < ma_slow, got ma_fast={mf} ma_slow={ms}")
    ix = sig.index
    n = len(ix)
    nav_out = np.empty(n, dtype=np.float64)
    sgv = sig.astype(float).to_numpy()
    m5a = pd.Series(sgv, index=ix).rolling(mf, min_periods=mf).mean().to_numpy()
    m12a = pd.Series(sgv, index=ix).rolling(ms, min_periods=ms).mean().to_numpy()
    mav = ma.astype(float).to_numpy()

    r = IDLE
    units = float(capital)
    u_safe = u_sb = 0.0
    agg_i = agg_h = 0.0
    prev_sv = prev_mah = float("nan")
    prev_ma_ok = False
    prev_m5 = prev_m120 = float("nan")

    for k in range(n):
        sv = float(sgv[k])
        sfx = float(safe.iloc[k])
        bv = float(bounce.iloc[k])
        dv = float(defense.iloc[k])
        mah = float(mav[k])
        ma_ok = not math.isnan(mah)
        m5 = float(m5a[k])
        m12 = float(m12a[k])

        nav_pre = _nav(r, units, u_safe, u_sb, sfx, bv, dv)

        cross_dn = (
            ma_ok
            and prev_ma_ok
            and sv < md * mah
            and prev_sv >= md * prev_mah
        )

        if cross_dn:
            agg_i = agg_h = 0.0
            if sfx <= 0:
                raise ValueError("규칙4: safe<=0")
            r = SAFE
            u_safe = nav_pre / sfx
            u_sb = 0.0
            units = 0.0
        elif r == BOUNCE and agg_i > 1e-12:
            agg_h = max(agg_h, bv)
            if bv < agg_i - 1e-12:
                r = DEFENSE
                units = nav_pre / dv
                u_safe = u_sb = 0.0
                agg_i = agg_h = 0.0
            elif agg_h > 1e-12 and bv < agg_h * trail - 1e-12:
                r = DEFENSE
                units = nav_pre / dv
                u_safe = u_sb = 0.0
                agg_i = agg_h = 0.0
        elif r in (SAFE, SAFE_BOUNCE):
            if mb._cross_up_103_ma(sv, mah, ma_ok, prev_sv, prev_mah, prev_ma_ok, mult_up=mu):
                if bv <= 0:
                    raise ValueError("SAFE→BOUNCE: bounce<=0")
                r = BOUNCE
                units = nav_pre / bv
                agg_i = agg_h = bv
                u_safe = u_sb = 0.0
            else:
                ma_dual = _ma_dual_ok(m5, m12, prev_m5, prev_m120)
                if ma_dual and u_safe > 1e-12 and bv > 0:
                    u_sb = nav_pre / bv
                    u_safe = 0.0
                r = SAFE_BOUNCE if u_sb > 1e-12 else SAFE
        elif r == BOUNCE and mb._cross_up_103_ma(sv, mah, ma_ok, prev_sv, prev_mah, prev_ma_ok, mult_up=mu):
            if bv <= 0:
                raise ValueError("reseed: bounce<=0")
            agg_i = agg_h = bv
        elif r == IDLE:
            if ma_ok:
                if bv <= 0:
                    raise ValueError("IDLE→BOUNCE: bounce<=0")
                r = BOUNCE
                units = nav_pre / bv
                u_safe = u_sb = 0.0
                agg_i = agg_h = bv

        nav_out[k] = _nav(r, units, u_safe, u_sb, sfx, bv, dv)
        prev_sv, prev_mah, prev_ma_ok = sv, mah, ma_ok
        prev_m5, prev_m120 = m5, m12

    return pd.Series(nav_out, index=ix, name="NAV")


def sortino_window(nav: pd.Series) -> float:
    return float(full_metrics(nav)["sortino"])


def dist_block(s: np.ndarray) -> dict[str, float]:
    s = s.astype(float)
    if len(s) < 1:
        return {"mean": 0.0, "std": 0.0, "median": 0.0}
    return {
        "mean": float(np.mean(s)),
        "std": float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
        "median": float(np.median(s)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="병합 FSM (sticky DEFENSE) + 단일 조합 MC Sortino")
    ap.add_argument("--bounce", default="TQQQ", help="반등 자산 (기본 TQQQ)")
    ap.add_argument("--defense", default="QQQ", help="방어 자산 (기본 QQQ)")
    ap.add_argument("--safe", default="Gold", help="안전 자산 (기본 Gold)")
    ap.add_argument("--root", default="1999-03-10")
    ap.add_argument("--trail", type=float, default=TRAIL_DEFAULT)
    ap.add_argument("--mc-years", type=float, default=3.0)
    ap.add_argument("--mc-iters", type=int, default=3000)
    ap.add_argument("--mc-seed", type=int, default=42)
    ap.add_argument("--capital", type=float, default=CAP0)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    bt = args.bounce.strip()
    dt = args.defense.strip()
    st = args.safe.strip()

    out_dir = Path(args.out_dir).resolve() if args.out_dir.strip() else _ROOT / "03_RESULT" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    root_ts = pd.Timestamp(args.root.strip())
    sg = load_extended_daily("QQQ")
    ql = load_extended_daily("QLD")
    tg = load_extended_daily("TQQQ")
    ix0 = sg.index.intersection(ql.index).intersection(tg.index).sort_values()
    ix0 = ix0[ix0 >= root_ts]
    gold_df, gold_note = load_gold_series(ix0[0], ix0[-1])
    ix = ix0.intersection(gold_df.index).sort_values()
    if len(ix) < MA_WIN:
        raise SystemExit(f"일수 {len(ix)} < MA{MA_WIN}")

    lo = mc_min_days(args.mc_years)
    if len(ix) < lo:
        raise SystemExit(f"전체 {len(ix)}일 < 최소 창 {lo}일")

    sig_s = sg["Close"].astype(float).reindex(ix)
    mav = ma200(sig_s).reindex(ix)

    sgx, qlx, tgx, glx = sg.reindex(ix), ql.reindex(ix), tg.reindex(ix), gold_df.reindex(ix)
    bser = pick(bt, sgx, qlx, tgx, glx)
    dser = pick(dt, sgx, qlx, tgx, glx)
    sser = pick(st, sgx, qlx, tgx, glx)

    print(f"전구간 NAV (sticky DEF) · bounce={bt} defense={dt} safe={st} …")
    nav = run_merge_fsm(sig_s, sser, bser, dser, mav, trail=args.trail, capital=args.capital)

    rng = np.random.default_rng(args.mc_seed)
    n_len = len(ix)
    sortinos: list[float] = []
    print(f"MC: ≥{args.mc_years:g}년, {args.mc_iters}회, seed={args.mc_seed}")
    for _ in range(args.mc_iters):
        s0 = int(rng.integers(0, n_len - lo + 1))
        win = int(rng.integers(lo, (n_len - s0) + 1))
        sortinos.append(sortino_window(nav.iloc[s0 : s0 + win]))

    blk = dist_block(np.array(sortinos))

    tk_b, tk_d, tk_s = bt.upper(), dt.upper(), st.upper()
    fx0 = pd.Timestamp(ix[0]).strftime("%Y%m%d")
    fx1 = pd.Timestamp(ix[-1]).strftime("%Y%m%d")
    stem = (
        f"merge_bounce_mc_stickydef_{fx0}_{fx1}_{tk_b}_{tk_d}_{tk_s}_"
        f"{args.mc_years:g}yr_n{args.mc_iters}_s{args.mc_seed}"
    )
    csv_p = out_dir / f"{stem}_sortino.csv"
    json_p = out_dir / f"{stem}_summary.json"

    row = {
        "bounce": tk_b,
        "defense": tk_d,
        "safe": tk_s,
        "sortino_mean": blk["mean"],
        "sortino_median": blk["median"],
        "sortino_std": blk["std"],
    }
    pd.DataFrame([row]).to_csv(csv_p, index=False)
    json_p.write_text(
        json.dumps(
            {
                "meta": {
                    "strategy": (
                        "merge_bounce_simple_mc_sticky_defense "
                        "(no DEF→BOUNCE; SAFE/BOUNCE 103MA는 상향돌파만; DEF는 규칙4→SAFE만)"
                    ),
                    "sig": SIG_TICKER,
                    "bounce": tk_b,
                    "defense": tk_d,
                    "safe": tk_s,
                    "gold_series_note": gold_note,
                    "index_days": n_len,
                    "trail": args.trail,
                    "mc_years": args.mc_years,
                    "mc_min_days": lo,
                    "mc_iters": args.mc_iters,
                    "mc_seed": args.mc_seed,
                },
                "mc_sortino": blk,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("=== 완료 ===")
    print(csv_p.resolve())
    print(json_p.resolve())


if __name__ == "__main__":
    main()
