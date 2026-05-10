# Monte Carlo: MABS · RMABS/DMABS-Gold · RMABS/DMABS-PSQ (PSQ 상장 이후 교집합)

설정: 최소 거래연 **3년**(거래일 ≥ 756일), 시행 **3000회**, seed=42, trail=0.85.

- **레그 요약**: 시그 QQ / 방 QQ / 노 QLD / 공 TQQQ
- **RMABS-4tier-Gold** / **DMABS-4tier-Gold**: 안전=금(GC=F (COMEX Gold Continuous Contract)) — DMABS는 RSI 대신 **MA5>MA120** 스트레스 블리드
- **RMABS-4tier-PSQ** / **DMABS-4tier-PSQ**: 레그 동일 · 안전=**PSQ**(Yahoo 수정주가 파일)
- PSQ(안전)=Yahoo 수정주가 02_DATA/yahoo/PSQ/PSQ_daily.csv 등; 금·QQQ 교집합과 PSQ 종가 존재일만 사용. PSQ 시계열에서 양호 첫 거래일: 2006-06-21.
- 공통 교집합: `2006-06-21` ~ `2026-04-30` (**4991일**) · 금: GC=F (COMEX Gold Continuous Contract)

## 지표 평균 (CAGR / MDD / Sharpe / Sortino / Ulcer)

| 구분 | CAGR 평균 | MDD 평균 | Sharpe 평균 | Sortino 평균 | Ulcer 평균 |
|------|----------:|----------:|------------:|-------------:|-----------:|
| MABS-QQQ | 32.50% | -45.09% | 0.849 | 1.112 | 16.7451 |
| RMABS-4tier-Gold | 32.01% | -44.72% | 0.857 | 1.092 | 15.9952 |
| DMABS-4tier-Gold | 34.37% | -42.43% | 0.913 | 1.159 | 14.4755 |
| RMABS-4tier-PSQ | 28.08% | -51.29% | 0.755 | 0.977 | 18.0015 |
| DMABS-4tier-PSQ | 24.82% | -51.36% | 0.680 | 0.892 | 18.4853 |

## 참고 분포 통계(JSON)

파일 `mc_five_gold_dmabs_psq_20060621_20260430_3yr_n3000_s42_summary.json` 에 시리즈별·지표별 mean/std/median/p05~p95가 있습니다.

## 원시 행 데이터

`mc_five_gold_dmabs_psq_20060621_20260430_3yr_n3000_s42_windows.csv` (창 시작·종료 포함, 시행별 전 컬럼)
