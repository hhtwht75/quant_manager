# Monte Carlo: MABS-QQQ · RMABS-4tier-Gold · DMABS-4tier-Gold · DMABS-Gold(def·nor=QLD)

설정: 최소 거래연 **3년**(거래일 ≥ 756일), 시행 **3000회**, seed=42, trail=0.85.

- **포함 전략**: MABS-QQQ, RMABS-4tier-Gold, DMABS-4tier-Gold, **DMABS-4tier-Gold (방어·노말=QLD)** (방어도 QLD 로 통일한 DMABS).
- **교집합**: ``--root`` 이후 QQQ·QLD·TQQQ·금 **모두 존재하는 거래일** 전 구간.
- 현재 실행: `1999-03-10` ~ `2026-04-30` (**6809일**) · 금 GC=F (COMEX) from 2000-08-30; preceding dates use ^XAU anchored to first GC=F close (approx. pre-contract Yahoo window — not TradingView GC1!)

## 무작위 창 지표 (3000회·동일 표본별 슬라이스): 평균·중앙

| 전략 | CAGR 평균 | CAGR 중앙 | MDD 평균 | MDD 중앙 | Sharpe 평균 | Sharpe 중앙 | Sortino 평균 | Sortino 중앙 | Ulcer 평균 | Ulcer 중앙 |
|------|----------:|----------:|----------:|----------:|------------:|-----------:|-------------:|------------:|----------:|-----------:|
| MABS-QQQ | 26.39% | 28.40% | -52.69% | -48.46% | 0.713 | 0.763 | 0.938 | 0.993 | 21.7171 | 19.3669 |
| RMABS-4tier-Gold | 25.50% | 27.43% | -50.63% | -49.82% | 0.703 | 0.771 | 0.901 | 0.962 | 20.1883 | 16.7875 |
| DMABS-4tier-Gold | 28.24% | 29.89% | -46.20% | -49.82% | 0.769 | 0.825 | 0.983 | 1.027 | 17.8467 | 15.4713 |
| DMABS-4tier-Gold (방어·노말=QLD) | 29.96% | 31.65% | -45.93% | -49.82% | 0.804 | 0.857 | 1.036 | 1.077 | 17.3264 | 15.2553 |

## 참고 분포 통계(JSON)

파일 `mc_four_mabs_rmabs_dmabs_dmabsDnq_fullix_19990310_20260430_3yr_n3000_s42_summary.json` 에 시리즈별·지표별 mean/std/median/p05~p95가 있습니다.

## 원시 행 데이터

`mc_four_mabs_rmabs_dmabs_dmabsDnq_fullix_19990310_20260430_3yr_n3000_s42_windows.csv` (창 시작·종료 포함, 시행별 전 컬럼)
