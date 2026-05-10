# Monte Carlo: MABS-QQQ · RMABS-4tier-Gold · DMABS-4tier-Gold · DMABS-4tier-Cash

설정: 최소 거래연 **1년**(거래일 ≥ 252일), 시행 **2회**, seed=1, trail=0.85.

- **레그 요약**: 시그 QQ / 방 QQ / 노 QLD / 공 TQQQ
- **RMABS-4tier-Gold** / **DMABS-4tier-Gold**: 안전=금(GC=F (COMEX Gold Continuous Contract)) — DMABS는 RSI 대신 **MA5>MA120** 스트레스 블리드
- **DMABS-4tier-Cash**: 안전=**종가 1.0**(무이자), 스트레스 동일(MA5>M120 → 방어)
- 교집합: `2002-10-01` ~ `2026-04-30` (**5918일**)

## 지표 평균 (CAGR / MDD / Sharpe / Sortino / Ulcer)

| 구분 | CAGR 평균 | MDD 평균 | Sharpe 평균 | Sortino 평균 | Ulcer 평균 |
|------|----------:|----------:|------------:|-------------:|-----------:|
| MABS-QQQ | 37.09% | -48.46% | 0.924 | 1.140 | 16.4879 |
| RMABS-4tier-Gold | 37.29% | -49.82% | 0.953 | 1.144 | 15.9712 |
| DMABS-4tier-Gold | 38.91% | -49.82% | 0.987 | 1.181 | 14.5060 |
| DMABS-4tier-Cash | 32.97% | -50.21% | 0.877 | 1.036 | 16.6122 |

## 참고 분포 통계(JSON)

파일 `mc_four_mabs_rmabs_dmabsGold_dmabsCash_20021001_20260430_1yr_n2_s1_summary.json` 에 시리즈별·지표별 mean/std/median/p05~p95가 있습니다.

## 원시 행 데이터

`mc_four_mabs_rmabs_dmabsGold_dmabsCash_20021001_20260430_1yr_n2_s1_windows.csv` (창 시작·종료 포함, 시행별 전 컬럼)
