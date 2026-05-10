# Monte Carlo: MABS×3 · RMABS-4tier-Gold/Cash · RMABS×2 · 벤치×4

설정: 최소 거래연 **3년**(거래일 ≥ 756일), 시행 **2회**, seed=1, trail=0.85.

- 교집합: `2002-10-01` ~ `2026-04-30` (**5918일**) · RMABS-4tier-Gold 안전=금 · RMABS-4tier-Cash 안전=1.0 고정 · GC=F (COMEX Gold Continuous Contract)
- 각 전략별 **전구간 1회 NAV** 후 창별 슬라이스 (**벤치만** 각 창 첫 거래일 B&H).

## 지표 평균 (CAGR / MDD / Sharpe / Sortino / Ulcer)

| 구분 | CAGR 평균 | MDD 평균 | Sharpe 평균 | Sortino 평균 | Ulcer 평균 |
|------|----------:|----------:|------------:|-------------:|-----------:|
| MABS-QQQ | 34.72% | -48.46% | 0.890 | 1.101 | 15.7551 |
| MABS-Gold | 33.48% | -49.82% | 0.888 | 1.068 | 15.3183 |
| MABS-Cash | 28.78% | -50.21% | 0.797 | 0.948 | 17.1825 |
| RMABS-QLD | 34.93% | -50.13% | 0.884 | 1.090 | 16.3817 |
| RMABS-QQQ | 26.68% | -34.13% | 0.875 | 1.062 | 11.6407 |
| RMABS-4tier-Gold | 34.59% | -49.82% | 0.910 | 1.096 | 15.3886 |
| RMABS-4tier-Cash | 32.15% | -50.21% | 0.865 | 1.028 | 16.0250 |
| Bench · QQQ 100% B&H | 20.39% | -31.84% | 0.799 | 0.997 | 8.3433 |
| Bench · QLD 100% B&H | 34.08% | -57.70% | 0.823 | 1.022 | 17.5198 |
| Bench · QQQ50/TQQQ50 | 35.75% | -66.88% | 0.791 | 0.968 | 21.5734 |
| Bench · TQQQ 100% B&H | 43.41% | -75.79% | 0.843 | 1.048 | 26.5997 |

## 참고 분포 통계(JSON)

파일 `mc_suite_mabs3_rmabs4tierGold_rmabs4tierCash_rmabs2_bh4_navslice_safeGoldFut_20021001_20260430_3yr_n2_s1_summary.json` 에 시리즈별·지표별 mean/std/median/p05~p95가 있습니다.

## 원시 행 데이터

`mc_suite_mabs3_rmabs4tierGold_rmabs4tierCash_rmabs2_bh4_navslice_safeGoldFut_20021001_20260430_3yr_n2_s1_windows.csv` (창 시작·종료 포함, 시행별 전 컬럼)
