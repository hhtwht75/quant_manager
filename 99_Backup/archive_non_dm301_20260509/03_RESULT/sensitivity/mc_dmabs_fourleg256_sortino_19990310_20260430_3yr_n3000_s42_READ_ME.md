# DMABS 256조합 — MC Sortino 평균 (정렬: 내림차순)

- 교집합 `1999-03-10` ~ `2026-04-30` (**6809일**), 금: GC=F (COMEX) from 2000-08-30; preceding dates use ^XAU anchored to first GC=F close (approx. pre-contract Yahoo window — not TradingView GC1!)
- 무작위 창 ≥**3년**(≥756일), **3000**회, seed=42, trail=0.85

**Close_sig 및 MA 로직은 전 조합 공통으로 QQQ.**

## 전체 표 (CSV)

`mc_dmabs_fourleg256_sortino_19990310_20260430_3yr_n3000_s42_sortino_rank.csv` · Top/Bottom 요약:

### Sortino 평균 Top 25

| 공격 | 반등 | 방어 | 안전 | Sortino 평균 | Sortino 중앙 |
|------|------|------|------|-------------:|-------------:|
| TQQQ | TQQQ | QQQ | Gold | 1.1868 | 1.2570 |
| Gold | TQQQ | QQQ | Gold | 1.1621 | 1.2046 |
| TQQQ | QLD | QQQ | Gold | 1.1472 | 1.2230 |
| TQQQ | TQQQ | Gold | Gold | 1.1369 | 1.1350 |
| Gold | QLD | QQQ | Gold | 1.1352 | 1.1724 |
| QLD | TQQQ | QQQ | Gold | 1.1232 | 1.1690 |
| TQQQ | QLD | Gold | Gold | 1.1021 | 1.1050 |
| TQQQ | QQQ | QQQ | Gold | 1.0846 | 1.1597 |
| QLD | QLD | QQQ | Gold | 1.0806 | 1.1259 |
| QLD | TQQQ | Gold | Gold | 1.0778 | 1.0557 |
| TQQQ | TQQQ | QLD | Gold | 1.0774 | 1.1175 |
| Gold | QQQ | QQQ | Gold | 1.0741 | 1.0974 |
| QQQ | TQQQ | QQQ | Gold | 1.0659 | 1.0848 |
| Gold | TQQQ | QLD | Gold | 1.0587 | 1.0956 |
| TQQQ | TQQQ | QQQ | QQQ | 1.0529 | 1.0963 |
| QLD | QLD | Gold | Gold | 1.0424 | 1.0257 |
| TQQQ | QQQ | Gold | Gold | 1.0415 | 1.0454 |
| TQQQ | QLD | QLD | Gold | 1.0359 | 1.0769 |
| TQQQ | TQQQ | QLD | QQQ | 1.0328 | 1.0867 |
| Gold | QLD | QLD | Gold | 1.0226 | 1.0528 |
| QQQ | QLD | QQQ | Gold | 1.0204 | 1.0398 |
| QLD | QQQ | QQQ | Gold | 1.0121 | 1.0600 |
| TQQQ | QLD | QQQ | QQQ | 1.0085 | 1.0535 |
| Gold | TQQQ | QLD | QQQ | 1.0027 | 1.0406 |
| QLD | TQQQ | QLD | Gold | 0.9997 | 1.0244 |

### Sortino 평균 Bottom 10

| 공격 | 반등 | 방어 | 안전 | Sortino 평균 | Sortino 중앙 |
|------|------|------|------|-------------:|-------------:|
| QQQ | Gold | QQQ | TQQQ | 0.4452 | 0.3964 |
| QQQ | QQQ | Gold | TQQQ | 0.4273 | 0.4122 |
| Gold | QLD | Gold | TQQQ | 0.4248 | 0.4197 |
| Gold | QQQ | Gold | QLD | 0.4002 | 0.3741 |
| Gold | QQQ | Gold | TQQQ | 0.3868 | 0.3768 |
| Gold | Gold | Gold | QQQ | 0.3752 | 0.3494 |
| QQQ | Gold | Gold | QLD | 0.3658 | 0.3328 |
| QQQ | Gold | Gold | TQQQ | 0.3623 | 0.3358 |
| Gold | Gold | Gold | TQQQ | 0.3211 | 0.2968 |
| Gold | Gold | Gold | QLD | 0.3063 | 0.2752 |

## JSON

`mc_dmabs_fourleg256_sortino_19990310_20260430_3yr_n3000_s42_summary.json`
