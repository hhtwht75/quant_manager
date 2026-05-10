# Monte Carlo: MABS×3 · RMABS-4tier · RMABS×2 · 벤치×4

설정: 최소 거래연 **3년**(거래일 ≥ 756일), 시행 **3000회**, seed=42, trail=0.85.

- 교집합: `2002-10-01` ~ `2026-04-30` (**5918일**) · **RMABS-4tier** 안전=금선물 레그 동일 소스 · GC=F (COMEX Gold Continuous Contract)
- 각 전략별 **전구간 1회 NAV** 후 창별 슬라이스 (**벤치만** 각 창 첫 거래일 B&H).

## 지표 평균 (CAGR / MDD / Sharpe / Sortino / Ulcer)

| 구분 | CAGR 평균 | MDD 평균 | Sharpe 평균 | Sortino 평균 | Ulcer 평균 |
|------|----------:|----------:|------------:|-------------:|-----------:|
| MABS-QQQ | 29.68% | -48.40% | 0.786 | 1.033 | 17.8247 |
| MABS-Gold | 27.81% | -45.96% | 0.759 | 0.967 | 17.1491 |
| MABS-Cash | 25.21% | -44.73% | 0.717 | 0.892 | 17.7790 |
| RMABS-QLD | 28.39% | -48.96% | 0.759 | 0.986 | 17.7444 |
| RMABS-QQQ | 22.65% | -37.26% | 0.757 | 0.970 | 12.7249 |
| RMABS-4tier | 28.52% | -47.53% | 0.774 | 0.989 | 17.2520 |
| Bench · QQQ 100% B&H | 16.87% | -35.57% | 0.663 | 0.866 | 10.4003 |
| Bench · QLD 100% B&H | 26.58% | -61.16% | 0.686 | 0.892 | 21.6514 |
| Bench · QQQ50/TQQQ50 | 27.66% | -64.18% | 0.673 | 0.871 | 23.5637 |
| Bench · TQQQ 100% B&H | 33.65% | -75.78% | 0.726 | 0.952 | 30.2203 |

## 참고 분포 통계(JSON)

파일 `mc_suite_mabs3_rmabs4tier_rmabs2_bh4_navslice_safeGoldFut_20021001_20260430_3yr_n3000_s42_summary.json` 에 시리즈별·지표별 mean/std/median/p05~p95가 있습니다.

## 원시 행 데이터

`mc_suite_mabs3_rmabs4tier_rmabs2_bh4_navslice_safeGoldFut_20021001_20260430_3yr_n3000_s42_windows.csv` (창 시작·종료 포함, 시행별 전 컬럼)
