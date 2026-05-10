"""전략 비교 결과를 한눈에 보는 표(DataFrame / 텍스트) 생성."""

from __future__ import annotations

from typing import Any

import pandas as pd

from evaluation_metrics import full_metrics


def full_period_metrics_table(
    *,
    codes: tuple[str, ...],
    series_meta: tuple[tuple[str, str], ...],
    nav: dict[str, pd.Series],
    event_counts: dict[str, int | None],
) -> pd.DataFrame:
    """행=전략, 열=전구간 절대 지표."""
    meta_map = dict(series_meta)
    rows: list[dict[str, Any]] = []
    for code in codes:
        m = full_metrics(nav[code])
        ev = event_counts.get(code)
        rows.append(
            {
                "코드": code,
                "전략": meta_map[code],
                "CAGR%": round(m["cagr"] * 100.0, 2),
                "Sharpe": round(m["sharpe"], 3),
                "Sortino": round(m["sortino"], 3),
                "MDD%": round(m["mdd"] * 100.0, 2),
                "Ulcer": round(m["ulcer"], 2),
                "Calmar": round(m["calmar"], 3),
                "Pain": round(m["pain_ratio"], 2),
                "UPI": round(m["upi"], 2),
                "총수익%": round(float(nav[code].iloc[-1] / nav[code].iloc[0] - 1.0) * 100.0, 2),
                "이벤트": ("—" if ev is None else int(ev)),
            }
        )
    return pd.DataFrame(rows)


def mc_window_median_table(
    *,
    codes: tuple[str, ...],
    series_meta: tuple[tuple[str, str], ...],
    summaries: dict[str, dict[str, dict[str, float]]],
) -> pd.DataFrame:
    """행=전략, 열=무작위 창 슬라이스 지표의 중앙값(p50).

    summaries[code][metric] 에 median 등이 들어있음 (monte_carlo 요약 형식).
    """
    meta_map = dict(series_meta)
    rows: list[dict[str, Any]] = []
    for code in codes:
        s = summaries[code]
        rows.append(
            {
                "코드": code,
                "전략": meta_map[code],
                "창CAGR%_p50": round(s["cagr"]["median"] * 100.0, 2),
                "창Sharpe_p50": round(s["sharpe"]["median"], 3),
                "창Sortino_p50": round(s["sortino"]["median"], 3),
                "창MDD%_p50": round(s["mdd"]["median"] * 100.0, 2),
                "창Ulcer_p50": round(s["ulcer"]["median"], 2),
            }
        )
    return pd.DataFrame(rows)


def mc_window_distrib_by_metric(
    df: pd.DataFrame,
    *,
    codes: tuple[str, ...],
    series_meta: tuple[tuple[str, str], ...],
    metrics: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """무작위 창에서 같은 열 이름 ``{코드}_{metric}`` 으로 집계.

    CAGR·MDD는 저장값이 소수(예 0.12)라 표시만 ×100 해 퍼센트 포인트로 맞춤.
    각 지표(metric)당 행=전략, 열=평균·중앙값·표준편차(표본, ddof=1).
    """
    meta_map = dict(series_meta)
    pct_scaled = frozenset({"cagr", "mdd"})
    out: dict[str, pd.DataFrame] = {}

    def _rnd(met: str, x: float) -> float:
        if met in pct_scaled:
            return round(float(x), 2)
        if met in ("sharpe", "sortino"):
            return round(float(x), 4)
        return round(float(x), 3)

    for met in metrics:
        rows: list[dict[str, Any]] = []
        for code in codes:
            ser = pd.to_numeric(df[f"{code}_{met}"], errors="coerce").dropna()
            if len(ser) == 0:
                mu = med = std = float("nan")
            else:
                med = float(ser.median())
                mu = float(ser.mean())
                std = float(ser.std(ddof=1)) if len(ser) > 1 else 0.0
            scl = 100.0 if met in pct_scaled else 1.0
            rows.append(
                {
                    "코드": code,
                    "전략": meta_map[code],
                    "평균": _rnd(met, mu * scl),
                    "중앙값": _rnd(met, med * scl),
                    "표준편차": _rnd(met, std * scl),
                }
            )
        out[met] = pd.DataFrame(rows)

    return out


MET_TITLE_KR = {
    "cagr": "CAGR (%·연환산, 각 창)",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "mdd": "MDD (%·각 창)",
    "ulcer": "Ulcer",
}


def delta_vs_b1_median_table(
    *,
    challengers: tuple[str, ...],
    series_meta: tuple[tuple[str, str], ...],
    delta_summary: dict[str, dict[str, dict[str, float]]],
) -> pd.DataFrame:
    """B1(QQQ) 대비 창별 Δ의 중앙값·승률. 키는 ``{코드}−B1``."""
    meta_map = dict(series_meta)
    rows: list[dict[str, Any]] = []
    for pref in challengers:
        lab = f"{pref}−B1"
        block = delta_summary[lab]
        rows.append(
            {
                "대비": lab,
                "전략": meta_map[pref],
                "ΔCAGR%_p50": round(block["cagr"]["median"] * 100.0, 2),
                "ΔSharpe_p50": round(block["sharpe"]["median"], 3),
                "ΔSortino_p50": round(block["sortino"]["median"], 3),
                "ΔMDD%_p50": round(block["mdd"]["median"] * 100.0, 2),
                "ΔUlcer_p50": round(block["ulcer"]["median"], 2),
                "ΔCAGR승%": round(block["cagr"]["win_pct"], 1),
                "ΔMDD승%": round(block["mdd"]["win_pct"], 1),
                "ΔUlcer승%": round(block["ulcer"]["win_pct"], 1),
            }
        )
    return pd.DataFrame(rows)


def df_to_github_markdown_table(df: pd.DataFrame) -> str:
    """GitHub 스타일 파이프 표(열 개수 일치 보장). tabulate 불필요."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = [_md_cell_esc(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _md_cell_esc(val: Any) -> str:
    s = "" if pd.isna(val) else str(val)
    return s.replace("|", "\\|").replace("\n", " ")


def df_to_wide_console_block(title: str, df: pd.DataFrame) -> str:
    """표 제목 + 정렬된 텍스트 (콘솔·파일 공용)."""
    body = df.to_string(index=False)
    w = max(len(line) for line in body.splitlines()) if body else 0
    sep = "=" * max(w, len(title))
    return f"{sep}\n{title}\n{sep}\n{body}\n{sep}"
