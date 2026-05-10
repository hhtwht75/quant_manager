# Monte Carlo: MABS-QQQ · RMABS-4tier-Gold · RMABS-4tier-Cash

설정: 최소 거래연 **3년**(거래일 ≥ 756일), 시행 **3000회**, seed=42, trail=0.85.

- **RMABS-4tier-Gold** 안전=금 · **RMABS-4tier-Cash** 안전=종가 1.0(무이자)
- 교집합: `2002-10-01` ~ `2026-04-30` (**5918일**) · 금 레그: GC=F (COMEX Gold Continuous Contract)

## 지표 평균 (CAGR / MDD / Sharpe / Sortino / Ulcer)

| 구분 | CAGR 평균 | MDD 평균 | Sharpe 평균 | Sortino 평균 | Ulcer 평균 |
|------|----------:|----------:|------------:|-------------:|-----------:|
| MABS-QQQ | 29.68% | -48.40% | 0.786 | 1.033 | 17.8247 |
| RMABS-4tier-Gold | 28.52% | -47.53% | 0.774 | 0.989 | 17.2520 |
| RMABS-4tier-Cash | 28.11% | -46.34% | 0.775 | 0.973 | 17.0353 |

## 참고 분포 통계(JSON)

파일 `mc_three_mabsqqq_rmabs4tierGold_rmabs4tierCash_20021001_20260430_3yr_n3000_s42_summary.json` 에 시리즈별·지표별 mean/std/median/p05~p95가 있습니다.

## 원시 행 데이터

`mc_three_mabsqqq_rmabs4tierGold_rmabs4tierCash_20021001_20260430_3yr_n3000_s42_windows.csv` (창 시작·종료 포함, 시행별 전 컬럼)
