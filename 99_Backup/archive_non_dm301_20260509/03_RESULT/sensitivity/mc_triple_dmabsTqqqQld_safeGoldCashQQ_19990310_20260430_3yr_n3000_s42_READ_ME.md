# Monte Carlo: DMABS-TQQQ-QLD-Gold · Cash · QQQ

설정: 최소 거래연 **3년**(거래일 ≥ 756일), 시행 **3000회**, seed=42, trail=0.85.

- 교집합: `1999-03-10` ~ `2026-04-30` (**6809일**), 금 레그 포함 시: GC=F (COMEX) from 2000-08-30; preceding dates use ^XAU anchored to first GC=F close (approx. pre-contract Yahoo window — not TradingView GC1!)

## 무작위 창 지표 (3000회): 평균·중앙

| 전략 | CAGR 평균 | CAGR 중앙 | MDD 평균 | MDD 중앙 | Sharpe 평균 | Sharpe 중앙 | Sortino 평균 | Sortino 중앙 | Ulcer 평균 | Ulcer 중앙 |
|------|----------:|----------:|----------:|----------:|------------:|-----------:|-------------:|------------:|----------:|-----------:|
| DMABS-TQQQ-QLD-Gold | 29.96% | 31.65% | -45.93% | -49.82% | 0.804 | 0.857 | 1.036 | 1.077 | 17.3264 | 15.2553 |
| DMABS-TQQQ-QLD-Cash | 26.49% | 27.49% | -44.84% | -50.21% | 0.742 | 0.779 | 0.931 | 0.959 | 18.1926 | 16.3085 |
| DMABS-TQQQ-QLD-QQQ | 28.08% | 30.11% | -52.49% | -48.46% | 0.747 | 0.797 | 0.990 | 1.047 | 20.8901 | 18.8573 |

## 참고 분포 통계(JSON)

파일 `mc_triple_dmabsTqqqQld_safeGoldCashQQ_19990310_20260430_3yr_n3000_s42_summary.json`

## 원시

`mc_triple_dmabsTqqqQld_safeGoldCashQQ_19990310_20260430_3yr_n3000_s42_windows.csv`
