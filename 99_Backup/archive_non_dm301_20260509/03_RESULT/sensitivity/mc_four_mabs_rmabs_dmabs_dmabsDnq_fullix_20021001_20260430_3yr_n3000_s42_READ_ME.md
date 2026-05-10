# Monte Carlo: MABS-QQQ · RMABS-4tier-Gold · DMABS-4tier-Gold · DMABS-Gold(def·nor=QLD)

설정: 최소 거래연 **3년**(거래일 ≥ 756일), 시행 **3000회**, seed=42, trail=0.85.

- **포함 전략**: MABS-QQQ, RMABS-4tier-Gold, DMABS-4tier-Gold, **DMABS-4tier-Gold (방어·노말=QLD)** (방어도 QLD 로 통일한 DMABS).
- **교집합**: ``--root`` 이후 QQQ·QLD·TQQQ·금 **모두 존재하는 거래일** 전 구간.
- 현재 실행: `2002-10-01` ~ `2026-04-30` (**5918일**) · 금 GC=F (COMEX Gold Continuous Contract)

## 무작위 창 지표 (3000회·동일 표본별 슬라이스): 평균·중앙

| 전략 | CAGR 평균 | CAGR 중앙 | MDD 평균 | MDD 중앙 | Sharpe 평균 | Sharpe 중앙 | Sortino 평균 | Sortino 중앙 | Ulcer 평균 | Ulcer 중앙 |
|------|----------:|----------:|----------:|----------:|------------:|-----------:|-------------:|------------:|----------:|-----------:|
| MABS-QQQ | 29.68% | 30.51% | -48.40% | -48.46% | 0.786 | 0.809 | 1.033 | 1.043 | 17.8247 | 17.6928 |
| RMABS-4tier-Gold | 28.52% | 29.82% | -47.53% | -49.82% | 0.774 | 0.822 | 0.989 | 1.018 | 17.2520 | 16.2664 |
| DMABS-4tier-Gold | 30.99% | 31.93% | -44.10% | -49.82% | 0.834 | 0.868 | 1.064 | 1.072 | 15.7316 | 15.0298 |
| DMABS-4tier-Gold (방어·노말=QLD) | 32.78% | 33.58% | -43.88% | -49.82% | 0.868 | 0.901 | 1.117 | 1.119 | 15.2455 | 14.7269 |

## 참고 분포 통계(JSON)

파일 `mc_four_mabs_rmabs_dmabs_dmabsDnq_fullix_20021001_20260430_3yr_n3000_s42_summary.json` 에 시리즈별·지표별 mean/std/median/p05~p95가 있습니다.

## 원시 행 데이터

`mc_four_mabs_rmabs_dmabs_dmabsDnq_fullix_20021001_20260430_3yr_n3000_s42_windows.csv` (창 시작·종료 포함, 시행별 전 컬럼)
