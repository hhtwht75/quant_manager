"""
evaluation_metrics.py
=====================
포트폴리오 평가용 강화된 메트릭 + paired bootstrap 차이 검정.

추가 메트릭
-----------
• Sortino:   downside-only volatility 사용 (upside vol 페널티 X)
• Ulcer Index: drawdown 누적 면적 — 단일 MDD가 아닌 "고통의 지속"
• Pain Ratio: CAGR / mean-DD (회복 기간을 직접 가중)
• Calmar:    CAGR / |MDD|  — 최대 손실 대비 수익
• MAR:       CAGR / Avg Drawdown
• Tail risk: 95% / 99% Daily Var, CVaR

Paired Bootstrap Difference Test
--------------------------------
같은 stationary block bootstrap sample에서 strategy와 S1을 동시 평가하여
ΔSharpe, ΔCAGR, ΔSortino 등을 paired 분포로 계산.
→ Politis-Romano (1994), Lopez de Prado (2018, ch.13) 의 권장 방법.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable

RF_ANNUAL = 0.04
RF_DAILY  = (1 + RF_ANNUAL) ** (1 / 252) - 1


def annualized_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return (returns.mean() - RF_DAILY) / returns.std() * np.sqrt(252)


def annualized_sortino(returns: np.ndarray, mar_daily: float = 0.0) -> float:
    """Downside-only volatility 사용. mar = minimum acceptable return."""
    if len(returns) < 2:
        return 0.0
    excess = returns - mar_daily
    downside = excess[excess < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0 if excess.mean() <= 0 else 10.0  # cap
    return excess.mean() / downside.std() * np.sqrt(252)


def cagr_from_returns(returns: np.ndarray, n_years: float | None = None) -> float:
    if len(returns) == 0:
        return 0.0
    if n_years is None:
        n_years = len(returns) / 252
    if n_years <= 0:
        return 0.0
    cum = (1 + returns).prod()
    return cum ** (1 / n_years) - 1


def max_drawdown(wealth: np.ndarray) -> float:
    """음수로 반환 (e.g. -0.30)."""
    if len(wealth) == 0:
        return 0.0
    peaks = np.maximum.accumulate(wealth)
    dd = (wealth - peaks) / peaks
    return float(dd.min())


def ulcer_index(wealth: np.ndarray) -> float:
    """sqrt( mean(squared drawdown%) ).  단위: %.
    낮을수록 좋음. drawdown 의 RMS — 깊고 오래갈수록 큼."""
    if len(wealth) == 0:
        return 0.0
    peaks = np.maximum.accumulate(wealth)
    dd_pct = (wealth - peaks) / peaks * 100  # 음수
    return float(np.sqrt(np.mean(dd_pct ** 2)))


def avg_drawdown(wealth: np.ndarray) -> float:
    """평균 drawdown (음수). 회복 기간을 자연스럽게 반영."""
    if len(wealth) == 0:
        return 0.0
    peaks = np.maximum.accumulate(wealth)
    dd = (wealth - peaks) / peaks
    return float(dd.mean())


def calmar(returns: np.ndarray, wealth: np.ndarray) -> float:
    """CAGR / |MDD|. 높을수록 좋음."""
    cagr = cagr_from_returns(returns)
    mdd  = abs(max_drawdown(wealth))
    if mdd == 0:
        return 0.0
    return cagr / mdd


def pain_ratio(returns: np.ndarray, wealth: np.ndarray) -> float:
    """CAGR / mean(|drawdown|). MDD 단일 시점이 아닌 'drawdown 평균 깊이'."""
    cagr = cagr_from_returns(returns)
    avg  = abs(avg_drawdown(wealth))
    if avg == 0:
        return 0.0
    return cagr / avg


def ulcer_performance_index(returns: np.ndarray, wealth: np.ndarray) -> float:
    """UPI = (CAGR - rf) / Ulcer.  Sharpe-like but uses Ulcer for risk."""
    cagr = cagr_from_returns(returns)
    ui = ulcer_index(wealth) / 100  # 비율로
    if ui == 0:
        return 0.0
    return (cagr - RF_ANNUAL) / ui


def daily_var(returns: np.ndarray, p: float = 0.05) -> float:
    """historical Value at Risk (음수)."""
    if len(returns) == 0:
        return 0.0
    return float(np.quantile(returns, p))


def daily_cvar(returns: np.ndarray, p: float = 0.05) -> float:
    """conditional VaR — VaR 이하 평균."""
    if len(returns) == 0:
        return 0.0
    var = daily_var(returns, p)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return float(var)
    return float(tail.mean())


def full_metrics(portfolio: pd.Series) -> dict:
    """포트폴리오 시계열에서 모든 메트릭 계산."""
    if len(portfolio) < 2:
        return {k: 0.0 for k in [
            "cagr", "sharpe", "sortino", "mdd", "ulcer", "calmar",
            "pain_ratio", "upi", "var95", "cvar95", "n_years",
        ]}
    rets = portfolio.pct_change().dropna().values
    wealth = portfolio.values
    n_years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
    return {
        "cagr":       cagr_from_returns(rets, n_years),
        "sharpe":     annualized_sharpe(rets),
        "sortino":    annualized_sortino(rets, RF_DAILY),
        "mdd":        max_drawdown(wealth),
        "ulcer":      ulcer_index(wealth),
        "avg_dd":     avg_drawdown(wealth),
        "calmar":     calmar(rets, wealth),
        "pain_ratio": pain_ratio(rets, wealth),
        "upi":        ulcer_performance_index(rets, wealth),
        "var95":      daily_var(rets, 0.05),
        "cvar95":     daily_cvar(rets, 0.05),
        "n_years":    n_years,
    }


def oos_metric_bundle(portfolio: pd.Series) -> dict:
    """OOS 리포트용: 누적총수익률, CAGR, MDD, Sharpe, Sortino, Ulcer (JSON 직렬화 가능)."""
    m = full_metrics(portfolio)
    tot = float(portfolio.iloc[-1] / portfolio.iloc[0] - 1.0)
    return {
        "total_return": tot,
        "cagr": float(m["cagr"]),
        "mdd": float(m["mdd"]),
        "sharpe": float(m["sharpe"]),
        "sortino": float(m["sortino"]),
        "ulcer": float(m["ulcer"]),
        "n_years": float(m["n_years"]),
    }


# ── Paired bootstrap difference test ─────────────────────────────────────────

def stationary_bootstrap_indices(n: int, block_len: int, n_iter: int,
                                  seed: int = 7) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix (n_iter × n).
    ★ Paired test 핵심: 두 strategy에 같은 idx 사용 → 시점 매칭"""
    rng = np.random.default_rng(seed)
    p = 1.0 / block_len
    idx_mat = np.empty((n_iter, n), dtype=np.int64)
    for it in range(n_iter):
        i = rng.integers(0, n)
        for t in range(n):
            idx_mat[it, t] = i
            if rng.random() < p:
                i = rng.integers(0, n)
            else:
                i = (i + 1) % n
    return idx_mat


def paired_bootstrap_compare(
    strat_returns: pd.Series,
    bench_returns: pd.Series,
    block_len: int = 60,
    n_iter: int = 500,
    seed: int = 7,
) -> dict:
    """Strategy vs Benchmark paired bootstrap.

    같은 시점 인덱스에서 두 시리즈를 함께 resample → ΔSharpe/ΔCAGR/Δ... 분포.
    """
    aligned = pd.concat([strat_returns, bench_returns], axis=1, join="inner").dropna()
    aligned.columns = ["s", "b"]
    s = aligned["s"].values
    b = aligned["b"].values
    n = len(s)
    if n < block_len * 2:
        block_len = max(5, n // 4)

    idx_mat = stationary_bootstrap_indices(n, block_len, n_iter, seed)

    diffs_sharpe   = np.empty(n_iter)
    diffs_cagr     = np.empty(n_iter)
    diffs_sortino  = np.empty(n_iter)
    diffs_calmar   = np.empty(n_iter)
    diffs_ulcer    = np.empty(n_iter)
    s_sharpes      = np.empty(n_iter)
    b_sharpes      = np.empty(n_iter)

    for it in range(n_iter):
        idx = idx_mat[it]
        ss = s[idx]; bb = b[idx]
        ws = (1 + ss).cumprod()
        wb = (1 + bb).cumprod()
        s_sh = annualized_sharpe(ss); b_sh = annualized_sharpe(bb)
        s_sharpes[it] = s_sh
        b_sharpes[it] = b_sh
        diffs_sharpe[it]  = s_sh - b_sh
        diffs_cagr[it]    = cagr_from_returns(ss) - cagr_from_returns(bb)
        diffs_sortino[it] = annualized_sortino(ss) - annualized_sortino(bb)
        diffs_calmar[it]  = calmar(ss, ws) - calmar(bb, wb)
        diffs_ulcer[it]   = ulcer_index(ws) - ulcer_index(wb)

    def summarise(arr: np.ndarray, higher_better: bool = True) -> dict:
        median = float(np.median(arr))
        p5     = float(np.percentile(arr, 5))
        p95    = float(np.percentile(arr, 95))
        prob_pos = float((arr > 0).mean())
        # one-sided p-value: H0 = no difference, H1 = strategy > bench
        if higher_better:
            p_value = 1.0 - prob_pos
        else:
            p_value = prob_pos
        return {"median": median, "p5": p5, "p95": p95,
                "prob_better": prob_pos if higher_better else 1.0 - prob_pos,
                "p_value": p_value}

    return {
        "n_iter":        n_iter,
        "block_len":     block_len,
        "n_obs":         n,
        "delta_sharpe":  summarise(diffs_sharpe,  True),
        "delta_cagr":    summarise(diffs_cagr,    True),
        "delta_sortino": summarise(diffs_sortino, True),
        "delta_calmar":  summarise(diffs_calmar,  True),
        "delta_ulcer":   summarise(diffs_ulcer,   False),  # 낮을수록 좋음
        "raw": {
            "delta_sharpe":  diffs_sharpe.tolist(),
            "delta_cagr":    diffs_cagr.tolist(),
            "delta_sortino": diffs_sortino.tolist(),
            "delta_calmar":  diffs_calmar.tolist(),
            "delta_ulcer":   diffs_ulcer.tolist(),
            "s_sharpe":      s_sharpes.tolist(),
            "b_sharpe":      b_sharpes.tolist(),
        },
    }


def stationary_bootstrap_sharpe_distribution(
    returns: np.ndarray,
    *,
    block_len: int = 60,
    n_iter: int = 500,
    seed: int = 7,
) -> dict[str, float | int | dict[str, float]]:
    """단변량 정적 블록 부트스트랩으로 Sharpe 재표본 분포 산출."""
    n = len(returns)
    if n < block_len * 2:
        block_len = max(5, n // 4)
    rng = np.random.default_rng(seed)
    p = 1.0 / block_len
    obs_sh = annualized_sharpe(returns)
    ys = np.empty(n_iter, dtype=float)
    for it in range(n_iter):
        out: list[float] = []
        idx = int(rng.integers(0, n))
        for _ in range(n):
            out.append(float(returns[idx]))
            if rng.random() < p:
                idx = int(rng.integers(0, n))
            else:
                idx = (idx + 1) % n
        r = np.array(out, dtype=float)
        ys[it] = annualized_sharpe(r)

    pct = float((ys < obs_sh).mean()) if n_iter else 0.0
    return {
        "n_obs": int(n),
        "block_len": int(block_len),
        "n_iter": int(n_iter),
        "obs_sharpe": float(obs_sh),
        "bootstrap_median_sharpe": float(np.median(ys)),
        "bootstrap_p95_sharpe": float(np.percentile(ys, 95)),
        "fraction_boot_ge_obs": float((ys >= obs_sh).mean()),
        "obs_percentile_approx": pct,
        "distribution": {
            "p05": float(np.percentile(ys, 5)),
            "p50": float(np.median(ys)),
            "p95": float(np.percentile(ys, 95)),
        },
    }


# ── 보고서용 헬퍼 ─────────────────────────────────────────────────────────────

def fmt_metrics_row(label: str, m: dict) -> str:
    return (
        f"  {label:30s}  "
        f"CAGR={m['cagr']*100:>+6.2f}%  "
        f"Sharpe={m['sharpe']:>5.2f}  "
        f"Sortino={m['sortino']:>5.2f}  "
        f"MDD={m['mdd']*100:>6.2f}%  "
        f"Ulcer={m['ulcer']:>5.2f}  "
        f"Calmar={m['calmar']:>5.2f}  "
        f"Pain={m['pain_ratio']:>5.2f}  "
        f"UPI={m['upi']:>5.2f}"
    )


def fmt_diff_row(label: str, d: dict) -> str:
    """Bootstrap diff 결과 한 줄 출력."""
    def cell(key, fmt="{:+.3f}"):
        x = d[key]
        sig = "★" if x["p_value"] < 0.05 else ("·" if x["p_value"] < 0.20 else " ")
        return f"{fmt.format(x['median'])}{sig} ({x['prob_better']*100:.0f}%)"

    return (
        f"  {label:30s}  "
        f"ΔSharpe = {cell('delta_sharpe'):20s}  "
        f"ΔCAGR  = {cell('delta_cagr', '{:+.1%}'):22s}  "
        f"ΔSortino = {cell('delta_sortino'):20s}  "
        f"ΔUlcer = {cell('delta_ulcer'):20s}"
    )
