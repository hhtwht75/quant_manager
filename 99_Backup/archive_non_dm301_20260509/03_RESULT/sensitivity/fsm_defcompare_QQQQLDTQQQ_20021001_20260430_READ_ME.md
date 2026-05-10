# FSM 비교: def=QQQ vs 금(선물 우선)·동일 교집합

고정: **sig=QQQ**, **nor=QLD**, **agg=TQQQ**, trail=0.85, 초기자본=100,000.

- 공통 평가 구간: `2002-10-01` ~ `2026-04-30` 거래일 **5918일** (`--root`=2002-10-01 이후 ∩ QQQ·QLD·TQQQ∩금 종가 존재일).
- 금 가격 소스: GC=F (COMEX Gold Continuous Contract)
- MA200 캐시: `/Users/vincent/Library/CloudStorage/Dropbox/Do_Money/quant_manager/02_DATA/cache/ma200/QQQ_200_ce592c835bfc2ba1_ma.csv`

## FSM 전략만 (def 체인지 비교)

| 이름 | CAGR | MDD | Sharpe | Sortino | Ulcer | 총수익률 |
|------|-----:|----:|-------:|--------:|------:|-----------:|
| FSM · def_asset=QQQ | 27.39% | -65.09% | 0.745 | 0.997 | 19.8688 | 30018.54% |
| FSM · def_asset=금(위 소스) | 23.62% | -57.43% | 0.677 | 0.883 | 20.6417 | 14730.57% |

## 벤치마크 (항상 동일 4종·동일 일자)

**벤치마크 정의(전략과 동일 달력·동일 평가지표):** 구간 첫날 종가 기준 전액 매수 후 보유(QQQ · QLD · TQQQ). 혼합은 자본을 반으로 나눠 QQQ·TQQQ 각각 매수 후 중간 리밸런스 없음.

| 이름 | CAGR | MDD | Sharpe | Sortino | Ulcer | 총수익률 |
|------|-----:|----:|-------:|--------:|------:|-----------:|
| 벤치 · QQQ 100% buy & hold | 16.42% | -53.40% | 0.629 | 0.836 | 12.1735 | 3501.33% |
| 벤치 · QLD 100% buy & hold | 25.06% | -83.13% | 0.645 | 0.853 | 25.8304 | 19377.42% |
| 벤치 · 초기 QQQ 50% + TQQQ 50% 매수 후 보유(무리밸런스) | 28.64% | -81.01% | 0.669 | 0.886 | 30.0124 | 37819.19% |
| 벤치 · TQQQ 100% buy & hold | 32.20% | -91.64% | 0.697 | 0.943 | 34.3327 | 72137.05% |

### def=금 − def=QQQ (전략만, 동일 차원)

- **Δcagr**: -3.7707 pct pt (비율 차이 ×100)
- **Δmdd**: +7.6571 pct pt (비율 차이 ×100)
- **Δsharpe**: -0.067996
- **Δsortino**: -0.114378
- **Δulcer**: +0.772924
- **Δtotal_return**: -152.879747 (총수익 배수‑1 차이; 표 위 총수익열은 ×100 표기)

## 산출 파일

| 파일 | 설명 |
|------|------|
| `fsm_defcompare_QQQQLDTQQQ_20021001_20260430_READ_ME.md` | 본 표 (에이전트 창에서 읽기 좋음) |
| `fsm_defcompare_QQQQLDTQQQ_20021001_20260430_combined.json` | 메타 · 전략 · 벤치(JSON) |
| `fsm_defcompare_QQQQLDTQQQ_20021001_20260430_strategy_only.json` | 전략 두 케이스 및 차이만 |
| `fsm_defcompare_QQQQLDTQQQ_20021001_20260430_nav_paired_defQQQ_vs_defGold.csv` | 일자별 NAV 두 시나리오 (paired CSV) |
