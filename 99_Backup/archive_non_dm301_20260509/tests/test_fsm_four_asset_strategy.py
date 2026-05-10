"""fsm_four_asset_strategy 전이 검증."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent.parent / "01_CODE"
sys.path.insert(0, str(_CODE))

from fsm_four_asset_strategy import CAPITAL_START, Regime, run_fsm_backtest  # noqa: E402


def _frame(
    idx: pd.DatetimeIndex,
    sig: list[float],
    agg: list[float],
    bounce_px: float | list[float] = 100.0,
    defense_px: float | list[float] = 100.0,
    safe_px: float | list[float] = 100.0,
) -> pd.DataFrame:
    n = len(idx)
    assert len(sig) == n == len(agg)

    def _col(v: float | list[float]) -> list[float]:
        if isinstance(v, (int, float)):
            return [float(v)] * n
        assert len(v) == n
        return [float(x) for x in v]

    return pd.DataFrame(
        {
            "Close_sig": sig,
            "Close_safe": _col(safe_px),
            "Close_bounce": _col(bounce_px),
            "Close_defense": _col(defense_px),
            "Close_agg": agg,
        },
        index=idx,
    )


class TestFsmTransitions(unittest.TestCase):
    def test_idle_cash_until_ma_valid(self):
        ix = pd.bdate_range("2024-01-01", periods=3)
        aligned = _frame(ix, [105.0, 106.0, 107.0], [100.0, 100.0, 100.0])
        ma = pd.Series([math.nan] * 3, index=ix)
        nav, ev = run_fsm_backtest(aligned, ma, trail_stop=0.85)
        self.assertAlmostEqual(nav.iloc[0], CAPITAL_START)
        self.assertAlmostEqual(nav.iloc[-1], CAPITAL_START)
        self.assertEqual(ev.iloc[-1]["regime_after"], Regime.IDLE.name)

    def test_idle_agg_bounce_barrier_103(self):
        ix = pd.bdate_range("2024-06-03", periods=1)
        ma = pd.Series([100.0], index=ix)
        _, ev_d = run_fsm_backtest(_frame(ix, [103.0], [110.0]), ma)
        _, ev_a = run_fsm_backtest(_frame(ix, [104.0], [110.0]), ma)
        self.assertEqual(ev_d.iloc[0]["regime_after"], Regime.BOUNCE.name)
        self.assertIn("IDLE_LE_103MA_BOUNCE", ev_d.iloc[0]["transitions"])
        self.assertEqual(ev_a.iloc[0]["regime_after"], Regime.AGG.name)
        self.assertIn("IDLE_GT_103MA_AGG", ev_a.iloc[0]["transitions"])

    def test_agg_exit_below_init_to_defense(self):
        ix = pd.bdate_range("2025-07-07", periods=5)
        ma = pd.Series([math.nan, math.nan, 100.0, 100.0, 100.0], index=ix)
        sig = [math.nan, math.nan, 104.0, 104.0, 104.0]
        agg = [110.0, 110.0, 115.0, 100.0, 100.0]
        aligned = _frame(ix, sig, agg)
        _, ev = run_fsm_backtest(aligned, ma, trail_stop=0.85)
        self.assertIn("AGG_LT_INIT_DEFENSE", "".join(ev["transitions"].tolist()))
        self.assertEqual(ev.iloc[3]["regime_after"], Regime.DEFENSE.name)

    def test_rule4_prior_over_agg_no_defense_transition_same_bar(self):
        """규칙 4 발생일에는 SAFE 직결(AGG defense 문자열 불가)."""
        ix = pd.bdate_range("2025-01-06", periods=6)
        ma = pd.Series([math.nan, math.nan, 100.0, 100.0, 100.0, 100.0], index=ix)
        sig = [math.nan, math.nan, 104.0, 104.0, 104.0, 93.0]
        agg = [110.0, 110.0, 110.0, 118.0, 117.0, 117.0]
        _, ev = run_fsm_backtest(_frame(ix, sig, agg, safe_px=50.0), ma, trail_stop=0.85)
        tail = ev.iloc[-1]
        self.assertEqual(tail["regime_after"], Regime.SAFE.name)
        self.assertIn("GLOBAL_LT_097MA_SAFE", tail["transitions"])
        self.assertFalse(
            tail["transitions"].replace(";", "").startswith("AGG"),
            msg=f"예상 불포함 defense 접두: {tail['transitions']}",
        )

    def test_rule4_mabs_returns_bounce_transition(self):
        ix = pd.bdate_range("2025-01-06", periods=6)
        ma = pd.Series([math.nan, math.nan, 100.0, 100.0, 100.0, 100.0], index=ix)
        sig = [math.nan, math.nan, 104.0, 104.0, 104.0, 93.0]
        agg = [110.0, 110.0, 110.0, 118.0, 117.0, 117.0]
        _, ev = run_fsm_backtest(
            _frame(ix, sig, agg, safe_px=50.0), ma, trail_stop=0.85, use_safe_ma_rule=False
        )
        tail = ev.iloc[-1]
        self.assertEqual(tail["regime_after"], Regime.BOUNCE.name)
        self.assertIn("GLOBAL_LT_097MA_BOUNCE", tail["transitions"])

    def test_defense_remains_until_rule4_only(self):
        """DEFENSE 이후 명시 규칙 없음: sig 깊게 이탈까지 idle 아님 bounce 아님 defense 유지."""

        ix = pd.bdate_range("2026-02-09", periods=8)
        ma = pd.Series(
            [
                math.nan,
                math.nan,
                math.nan,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
            ],
            index=ix,
        )
        sig = [math.nan, math.nan, math.nan, 104.0, 104.0, 104.0, 103.8, 103.8]
        agg = [
            100.0,
            100.0,
            100.0,
            116.0,
            117.5,
            112.8,
            112.9,
            112.95,
        ]
        _, ev = run_fsm_backtest(_frame(ix, sig, agg), ma, trail_stop=0.85)
        dr = ev[ev["regime_after"] == Regime.DEFENSE.name]
        self.assertGreaterEqual(len(dr), 2)

    def test_safe_rsi_cross_halves_then_only_on_new_cross(self):
        """SAFE에서 규칙4가 꺼진 날만 RSI 교차 검사 · 전일≥30·당일<30 에만 이전 매도."""
        ix = pd.bdate_range("2028-06-05", periods=11)
        ma = pd.Series([100.0] * 11, index=ix)
        # 전일≥0.97MA, 당일 하방 교차 후 SAFE → RSI 블리딩 테스트
        sig = [98.0, 93.0] + [98.0] * 9
        agg = [10.0] * 11
        aligned = _frame(ix, sig, agg, safe_px=2.0, bounce_px=2.0, defense_px=2.0)
        rsi_old_tail = [28.0, 29.0, 35.0, 34.0, 29.0, 29.0, 29.0, 20.0]
        rsi_ov = pd.Series([np.nan, np.nan] + [np.nan] + rsi_old_tail, index=ix)
        _, ev = run_fsm_backtest(
            aligned, ma, rsi_override=rsi_ov, rsi_cross_below=30.0, trail_stop=0.85
        )
        r7 = ev[ev["Date"] == ix[7]].iloc[0]
        self.assertIn("SAFE_RSI30_HALVE_TO_BOUNCE", str(r7["transitions"]))
        r8 = ev[ev["Date"] == ix[8]].iloc[0]
        self.assertNotIn("SAFE_RSI30", str(r8["transitions"]))
        self.assertEqual(ev.iloc[-1]["regime_after"], Regime.SAFE_BOUNCE.name)

    def test_stress_agg_breakout_before_rsi_bleed_same_bar(self):
        ix = pd.bdate_range("2029-01-02", periods=5)
        ma = pd.Series([100.0] * 5, index=ix)
        sig = [100.0, 93.0, 104.0, 104.0, 104.0]
        agg = [10.0, 10.0, 60.0, 60.0, 60.0]
        aligned = _frame(ix, sig, agg, safe_px=1.0)
        rsi_ov = pd.Series([np.nan, np.nan, 25.0, 25.0, 25.0], index=ix)
        _, ev = run_fsm_backtest(aligned, ma, rsi_override=rsi_ov, trail_stop=0.85)
        row = ev.iloc[2]
        self.assertEqual(row["regime_after"], Regime.AGG.name)
        self.assertIn("SIG_GT_MULT_MA_TO_AGG", row["transitions"])
        self.assertNotIn("SAFE_RSI30", row["transitions"])

    def test_dmabs_ma5_cross_above_ma120_full_bounce_then_rule4_flush(self):
        """DMABS: MA5>M120 교차 시 안전→반등 100%(SAFE_BOUNCE), 이후 규칙4로 SAFE 재전량.

        SAFE 분기에서 103·MA 초과 시 AGG로 선점전이 하므로, 종가 전 구간은 103 미만으로 둠.
        """
        n = 146
        ix = pd.bdate_range("2029-01-02", periods=n)
        ma = pd.Series([100.0] * n, index=ix)
        sig_a = np.full(n, 101.0)
        sig_a[-17:-2] = np.linspace(101.0, 101.98, 15)
        sig_a[-2] = 101.9
        sig_a[-1] = 93.0
        aligned = _frame(ix, list(sig_a), [10.0] * n, safe_px=1.0, bounce_px=50.0)
        bootstrap_eod = ix[119]

        _, ev_dm = run_fsm_backtest(
            aligned,
            ma,
            trail_stop=0.85,
            bootstrap_eod_date=bootstrap_eod,
            bootstrap_capital=CAPITAL_START,
            bootstrap_initial_regime="AUTO",
            stress_bleed_mode="MA5_CROSS_MA120_FULL",
        )
        tr = "".join(ev_dm["transitions"].tolist())
        self.assertIn("SAFE_MA5_GT_MA120_FULL_BOUNCE", tr)
        self.assertIn("GLOBAL_LT_097MA_SAFE_FLUSH_MIX", tr)
        tail = ev_dm.iloc[-1]
        self.assertEqual(tail["regime_after"], Regime.SAFE.name)
        self.assertAlmostEqual(float(tail["u_safe"]), CAPITAL_START, places=3)
        self.assertAlmostEqual(float(tail["u_stress_bounce"]), 0.0, places=6)

        _, ev_rsi = run_fsm_backtest(
            aligned,
            ma,
            trail_stop=0.85,
            bootstrap_eod_date=bootstrap_eod,
            bootstrap_capital=CAPITAL_START,
            bootstrap_initial_regime="AUTO",
            stress_bleed_mode="RSI_HALF",
        )
        self.assertFalse(
            any("SAFE_MA5_GT_MA120" in str(r) for r in ev_rsi["transitions"]),
            msg="RSI_HALF 경로에서는 MA5 교차 전이 없어야 함",
        )

    def test_merge_bounce_below_init_defense_matches_agg_exit(self):
        """merge_agg_into_bounce_trailing: AGG 분리 없이 반등 종가로 동일 초기가 이탈."""
        ix = pd.bdate_range("2025-07-07", periods=5)
        ma = pd.Series([math.nan, math.nan, 100.0, 100.0, 100.0], index=ix)
        sig = [math.nan, math.nan, 104.0, 104.0, 104.0]
        agg = [110.0, 110.0, 115.0, 100.0, 100.0]
        aligned = _frame(ix, sig, agg, bounce_px=agg)
        _, ev = run_fsm_backtest(aligned, ma, trail_stop=0.85, merge_agg_into_bounce_trailing=True)
        self.assertIn("BOUNCE_LT_INIT_DEFENSE", "".join(ev["transitions"].tolist()))
        self.assertEqual(ev.iloc[3]["regime_after"], Regime.DEFENSE.name)


if __name__ == "__main__":
    unittest.main()
