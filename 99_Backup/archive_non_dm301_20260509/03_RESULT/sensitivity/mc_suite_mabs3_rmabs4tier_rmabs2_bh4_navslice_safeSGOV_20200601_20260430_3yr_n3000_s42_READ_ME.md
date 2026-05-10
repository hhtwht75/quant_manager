# Monte Carlo: MABS×3 · RMABS-4tier · RMABS×2 · 벤치×4

설정: 최소 거래연 **3년**(거래일 ≥ 756일), 시행 **3000회**, seed=42, trail=0.85.

- 교집합: `2020-06-01` ~ `2026-04-30` (**1487일**) · RMABS-4tier 안전레그 `SGOV` · 금: GC=F (COMEX Gold Continuous Contract)
- 각 전략별 **전구간 1회 NAV** 후 창별 슬라이스 (**벤치만** 각 창 첫 거래일 B&H).

## 지표 평균 (CAGR / MDD / Sharpe / Sortino / Ulcer)

| 구분 | CAGR 평균 | MDD 평균 | Sharpe 평균 | Sortino 평균 | Ulcer 평균 |
|------|----------:|----------:|------------:|-------------:|-----------:|
| MABS-QQQ | 33.78% | -39.43% | 0.864 | 1.266 | 17.1199 |
| MABS-Gold | 39.79% | -30.33% | 1.039 | 1.440 | 13.0013 |
| MABS-Cash | 34.30% | -29.56% | 0.942 | 1.257 | 13.4104 |
| RMABS-QLD | 36.73% | -39.44% | 0.916 | 1.343 | 17.4979 |
| RMABS-QQQ | 30.76% | -31.56% | 0.934 | 1.301 | 13.3358 |
| RMABS-4tier | 35.06% | -28.83% | 0.960 | 1.246 | 13.1902 |
| Bench · QQQ 100% B&H | 17.83% | -30.26% | 0.682 | 0.973 | 11.4956 |
| Bench · QLD 100% B&H | 25.50% | -55.43% | 0.648 | 0.928 | 24.1744 |
| Bench · QQQ50/TQQQ50 | 23.94% | -55.98% | 0.600 | 0.842 | 25.4487 |
| Bench · TQQQ 100% B&H | 28.54% | -72.73% | 0.646 | 0.921 | 36.0229 |

## 참고 분포 통계(JSON)

파일 `mc_suite_mabs3_rmabs4tier_rmabs2_bh4_navslice_safeSGOV_20200601_20260430_3yr_n3000_s42_summary.json` 에 시리즈별·지표별 mean/std/median/p05~p95가 있습니다.

## 원시 행 데이터

`mc_suite_mabs3_rmabs4tier_rmabs2_bh4_navslice_safeSGOV_20200601_20260430_3yr_n3000_s42_windows.csv` (창 시작·종료 포함, 시행별 전 컬럼)
