# Monte Carlo: MABS-QQQ · RMABS-4tier-Gold · DMABS-4tier-Gold (금 교집합 전구간 · --root 무시)

설정: 최소 거래연 **3년**(거래일 ≥ 756일), 시행 **3000회**, seed=42, trail=0.85.

- **포함 전략**: MABS-QQQ, RMABS-4tier-Gold, DMABS-4tier-Gold만.
- **교집합**: QQQ·QLD·TQQQ·금 **전 구간** (트리플 모드에서 ``--root`` 미적용). QQQ 확장 데이터는 보통 ~1999-03부터 가능하나, 금 포함 시 시작은 더 늦음.
- 현재 실행 기준 표본: `2000-08-30` ~ `2026-04-30` (**6435일**) · 금 GC=F (COMEX Gold Continuous Contract)

## 지표 평균 (CAGR / MDD / Sharpe / Sortino / Ulcer)

| 구분 | CAGR 평균 | MDD 평균 | Sharpe 평균 | Sortino 평균 | Ulcer 평균 |
|------|----------:|----------:|------------:|-------------:|-----------:|
| MABS-QQQ | 28.07% | -50.30% | 0.749 | 0.985 | 18.9313 |
| RMABS-4tier-Gold | 26.96% | -48.68% | 0.736 | 0.943 | 17.9231 |
| DMABS-4tier-Gold | 29.50% | -44.82% | 0.799 | 1.020 | 16.3315 |

## 참고 분포 통계(JSON)

파일 `mc_three_mabs_rmabs_dmabsGold_fullix_20000830_20260430_3yr_n3000_s42_summary.json` 에 시리즈별·지표별 mean/std/median/p05~p95가 있습니다.

## 원시 행 데이터

`mc_three_mabs_rmabs_dmabsGold_fullix_20000830_20260430_3yr_n3000_s42_windows.csv` (창 시작·종료 포함, 시행별 전 컬럼)
