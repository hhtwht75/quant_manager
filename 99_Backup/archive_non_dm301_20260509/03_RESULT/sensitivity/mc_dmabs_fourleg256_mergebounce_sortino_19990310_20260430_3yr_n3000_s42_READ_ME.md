# DMABS 256조합 — MC Sortino 평균 (정렬: 내림차순)

- 교집합 `1999-03-10` ~ `2026-04-30` (**6809일**), 금: GC=F (COMEX) from 2000-08-30; preceding dates use ^XAU anchored to first GC=F close (approx. pre-contract Yahoo window — not TradingView GC1!)
- 무작위 창 ≥**3년**(≥756일), **3000**회, seed=42, trail=0.85

**Close_sig 및 MA 로직은 전 조합 공통으로 QQQ.**

## 전체 표 (CSV)

`mc_dmabs_fourleg256_mergebounce_sortino_19990310_20260430_3yr_n3000_s42_sortino_rank.csv` · Top/Bottom 요약:

### Sortino 평균 Top 25

| 공격 | 반등 | 방어 | 안전 | Sortino 평균 | Sortino 중앙 |
|------|------|------|------|-------------:|-------------:|
| TQQQ | TQQQ | QQQ | Gold | 1.1868 | 1.2570 |
| QLD | TQQQ | QQQ | Gold | 1.1868 | 1.2570 |
| QQQ | TQQQ | QQQ | Gold | 1.1868 | 1.2570 |
| Gold | TQQQ | QQQ | Gold | 1.1868 | 1.2570 |
| TQQQ | TQQQ | Gold | Gold | 1.1369 | 1.1350 |
| QLD | TQQQ | Gold | Gold | 1.1369 | 1.1350 |
| QQQ | TQQQ | Gold | Gold | 1.1369 | 1.1350 |
| Gold | TQQQ | Gold | Gold | 1.1369 | 1.1350 |
| TQQQ | QLD | QQQ | Gold | 1.0806 | 1.1259 |
| QLD | QLD | QQQ | Gold | 1.0806 | 1.1259 |
| QQQ | QLD | QQQ | Gold | 1.0806 | 1.1259 |
| Gold | QLD | QQQ | Gold | 1.0806 | 1.1259 |
| TQQQ | TQQQ | QLD | Gold | 1.0774 | 1.1175 |
| QLD | TQQQ | QLD | Gold | 1.0774 | 1.1175 |
| QQQ | TQQQ | QLD | Gold | 1.0774 | 1.1175 |
| Gold | TQQQ | QLD | Gold | 1.0774 | 1.1175 |
| TQQQ | TQQQ | QQQ | QQQ | 1.0529 | 1.0963 |
| QLD | TQQQ | QQQ | QQQ | 1.0529 | 1.0963 |
| QQQ | TQQQ | QQQ | QQQ | 1.0529 | 1.0963 |
| Gold | TQQQ | QQQ | QQQ | 1.0529 | 1.0963 |
| TQQQ | QLD | Gold | Gold | 1.0424 | 1.0257 |
| QLD | QLD | Gold | Gold | 1.0424 | 1.0257 |
| QQQ | QLD | Gold | Gold | 1.0424 | 1.0257 |
| Gold | QLD | Gold | Gold | 1.0424 | 1.0257 |
| TQQQ | TQQQ | QLD | QQQ | 1.0328 | 1.0867 |

### Sortino 평균 Bottom 10

| 공격 | 반등 | 방어 | 안전 | Sortino 평균 | Sortino 중앙 |
|------|------|------|------|-------------:|-------------:|
| QQQ | Gold | Gold | QQQ | 0.3752 | 0.3494 |
| Gold | Gold | Gold | QQQ | 0.3752 | 0.3494 |
| TQQQ | Gold | Gold | TQQQ | 0.3211 | 0.2968 |
| QLD | Gold | Gold | TQQQ | 0.3211 | 0.2968 |
| QQQ | Gold | Gold | TQQQ | 0.3211 | 0.2968 |
| Gold | Gold | Gold | TQQQ | 0.3211 | 0.2968 |
| TQQQ | Gold | Gold | QLD | 0.3063 | 0.2752 |
| QLD | Gold | Gold | QLD | 0.3063 | 0.2752 |
| QQQ | Gold | Gold | QLD | 0.3063 | 0.2752 |
| Gold | Gold | Gold | QLD | 0.3063 | 0.2752 |

## JSON

`mc_dmabs_fourleg256_mergebounce_sortino_19990310_20260430_3yr_n3000_s42_summary.json`
