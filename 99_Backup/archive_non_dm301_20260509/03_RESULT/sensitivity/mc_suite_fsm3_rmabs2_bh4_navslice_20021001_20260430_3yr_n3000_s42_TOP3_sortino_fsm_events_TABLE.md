# MC 창별 FSM-defQQQ Sortino > Gold 상위 분석

- 사용한 시행표: `03_RESULT/sensitivity/mc_suite_fsm3_rmabs2_bh4_navslice_20021001_20260430_3yr_n3000_s42_windows.csv`
- 금 시계열: GC=F (COMEX Gold Continuous Contract)
- trail_stop: 0.85
- 상위 선택: (Sortino 초과폭 Δ>0 인 행 중 상위)

## 시행 순위 #1: `2009-02-27` ~ `2013-03-01`

| 항목 | FSM-defQQQ | FSM-defGold |
|---|---:|---:|
| 이 창에서 CSV 기록 Sortino | 1.389814 | 0.696529 |
| Δ_sortino(QQ−Gold) | **0.693285** | — |

- 구간 거래일(시행표): **1009**, 전구간 이벤트로 펼친 일수: QQ **1009**, Gold **1009**
- **상태/체이닝 문자열이 빈 날 제외한 ‘이벤트 발생일’ 행수: 17**

### 이벤트 발생일만 (둘 중 하나라도 `transitions` 비어있지 않음)

| Date | regime_after_QQ_defQQQ | transitions_QQ_defQQQ | regime_after_defGoldPX | transitions_defGoldPX | nav_eod_QQ_defQQQ | nav_eod_defGoldPX | sig_QQ_defQQQ | sig_defGoldPX |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2009-05-04 | AGG | DEF_GT_103MA_AGG | AGG | DEF_GT_103MA_AGG | 159992.32 | 118096.21 | 30.3047 | 30.3047 |
| 2009-05-05 | NOR | AGG_LT_INIT_NOR | NOR | AGG_LT_INIT_NOR | 159475.0 | 117714.36 | 30.2788 | 30.2788 |
| 2010-06-29 | DEF | GLOBAL_LT_097MA_DEF | DEF | GLOBAL_LT_097MA_DEF | 233529.83 | 172376.95 | 37.8293 | 37.8293 |
| 2010-08-04 | AGG | DEF_GT_103MA_AGG | AGG | DEF_GT_103MA_AGG | 252752.86 | 165673.4 | 40.9433 | 40.9433 |
| 2010-08-05 | NOR | AGG_LT_INIT_NOR | NOR | AGG_LT_INIT_NOR | 251260.7 | 164695.32 | 40.8473 | 40.8473 |
| 2010-08-24 | DEF | GLOBAL_LT_097MA_DEF | DEF | GLOBAL_LT_097MA_DEF | 217834.1 | 142784.99 | 38.0736 | 38.0736 |
| 2010-09-13 | AGG | DEF_GT_103MA_AGG | AGG | DEF_GT_103MA_AGG | 235799.85 | 144326.66 | 41.2137 | 41.2137 |
| 2011-03-15 | NOR | AGG_TRAIL_NOR | NOR | AGG_TRAIL_NOR | 373646.52 | 228698.85 | 48.6105 | 48.6105 |
| 2011-08-04 | DEF | GLOBAL_LT_097MA_DEF | DEF | GLOBAL_LT_097MA_DEF | 352656.2 | 215851.24 | 47.6277 | 47.6277 |
| 2011-10-14 | AGG | DEF_GT_103MA_AGG | AGG | DEF_GT_103MA_AGG | 379464.46 | 219187.68 | 51.2482 | 51.2482 |
| 2011-10-17 | NOR | AGG_LT_INIT_NOR | NOR | AGG_LT_INIT_NOR | 361898.96 | 209041.43 | 50.4555 | 50.4555 |
| 2011-11-21 | DEF | GLOBAL_LT_097MA_DEF | DEF | GLOBAL_LT_097MA_DEF | 322886.68 | 186507.02 | 47.8658 | 47.8658 |
| 2012-01-05 | AGG | DEF_GT_103MA_AGG | AGG | DEF_GT_103MA_AGG | 343626.9 | 179961.54 | 50.9404 | 50.9404 |
| 2012-04-24 | NOR | AGG_TRAIL_NOR | NOR | AGG_TRAIL_NOR | 482488.89 | 252685.24 | 57.3332 | 57.3332 |
| 2012-11-13 | DEF | GLOBAL_LT_097MA_DEF | DEF | GLOBAL_LT_097MA_DEF | 452300.2 | 236875.06 | 56.0709 | 56.0709 |
| 2013-01-02 | AGG | DEF_GT_103MA_AGG | AGG | DEF_GT_103MA_AGG | 485294.59 | 231888.08 | 60.1611 | 60.1611 |
| 2013-01-03 | NOR | AGG_LT_INIT_NOR | NOR | AGG_LT_INIT_NOR | 478264.44 | 228528.87 | 59.8478 | 59.8478 |

## 시행 순위 #2: `2004-08-10` ~ `2007-10-22`

| 항목 | FSM-defQQQ | FSM-defGold |
|---|---:|---:|
| 이 창에서 CSV 기록 Sortino | 0.860831 | 0.221637 |
| Δ_sortino(QQ−Gold) | **0.639194** | — |

- 구간 거래일(시행표): **799**, 전구간 이벤트로 펼친 일수: QQ **799**, Gold **799**
- **상태/체이닝 문자열이 빈 날 제외한 ‘이벤트 발생일’ 행수: 8**

### 이벤트 발생일만 (둘 중 하나라도 `transitions` 비어있지 않음)

| Date | regime_after_QQ_defQQQ | transitions_QQ_defQQQ | regime_after_defGoldPX | transitions_defGoldPX | nav_eod_QQ_defQQQ | nav_eod_defGoldPX | sig_QQ_defQQQ | sig_defGoldPX |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2004-10-28 | AGG | DEF_GT_103MA_AGG | AGG | DEF_GT_103MA_AGG | 200360.43 | 176459.89 | 31.1902 | 31.1902 |
| 2004-10-29 | NOR | AGG_LT_INIT_NOR | NOR | AGG_LT_INIT_NOR | 199256.02 | 175487.23 | 31.1396 | 31.1396 |
| 2005-04-15 | DEF | GLOBAL_LT_097MA_DEF | DEF | GLOBAL_LT_097MA_DEF | 173749.43 | 153023.26 | 29.5976 | 29.5976 |
| 2005-05-23 | AGG | DEF_GT_103MA_AGG | AGG | DEF_GT_103MA_AGG | 189303.81 | 150034.11 | 32.2472 | 32.2472 |
| 2005-05-25 | NOR | AGG_LT_INIT_NOR | NOR | AGG_LT_INIT_NOR | 188449.03 | 149356.65 | 32.2131 | 32.2131 |
| 2006-05-17 | DEF | GLOBAL_LT_097MA_DEF | DEF | GLOBAL_LT_097MA_DEF | 190143.31 | 150699.46 | 33.635 | 33.635 |
| 2006-10-04 | AGG | DEF_GT_103MA_AGG | AGG | DEF_GT_103MA_AGG | 199967.57 | 122654.74 | 35.3729 | 35.3729 |
| 2007-02-27 | NOR | AGG_TRAIL_NOR | NOR | AGG_TRAIL_NOR | 214320.29 | 131458.31 | 37.0367 | 37.0367 |

## 시행 순위 #3: `2009-01-20` ~ `2012-04-27`

| 항목 | FSM-defQQQ | FSM-defGold |
|---|---:|---:|
| 이 창에서 CSV 기록 Sortino | 1.686187 | 1.106724 |
| Δ_sortino(QQ−Gold) | **0.579462** | — |

- 구간 거래일(시행표): **826**, 전구간 이벤트로 펼친 일수: QQ **826**, Gold **826**
- **상태/체이닝 문자열이 빈 날 제외한 ‘이벤트 발생일’ 행수: 14**

### 이벤트 발생일만 (둘 중 하나라도 `transitions` 비어있지 않음)

| Date | regime_after_QQ_defQQQ | transitions_QQ_defQQQ | regime_after_defGoldPX | transitions_defGoldPX | nav_eod_QQ_defQQQ | nav_eod_defGoldPX | sig_QQ_defQQQ | sig_defGoldPX |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2009-05-04 | AGG | DEF_GT_103MA_AGG | AGG | DEF_GT_103MA_AGG | 159992.32 | 118096.21 | 30.3047 | 30.3047 |
| 2009-05-05 | NOR | AGG_LT_INIT_NOR | NOR | AGG_LT_INIT_NOR | 159475.0 | 117714.36 | 30.2788 | 30.2788 |
| 2010-06-29 | DEF | GLOBAL_LT_097MA_DEF | DEF | GLOBAL_LT_097MA_DEF | 233529.83 | 172376.95 | 37.8293 | 37.8293 |
| 2010-08-04 | AGG | DEF_GT_103MA_AGG | AGG | DEF_GT_103MA_AGG | 252752.86 | 165673.4 | 40.9433 | 40.9433 |
| 2010-08-05 | NOR | AGG_LT_INIT_NOR | NOR | AGG_LT_INIT_NOR | 251260.7 | 164695.32 | 40.8473 | 40.8473 |
| 2010-08-24 | DEF | GLOBAL_LT_097MA_DEF | DEF | GLOBAL_LT_097MA_DEF | 217834.1 | 142784.99 | 38.0736 | 38.0736 |
| 2010-09-13 | AGG | DEF_GT_103MA_AGG | AGG | DEF_GT_103MA_AGG | 235799.85 | 144326.66 | 41.2137 | 41.2137 |
| 2011-03-15 | NOR | AGG_TRAIL_NOR | NOR | AGG_TRAIL_NOR | 373646.52 | 228698.85 | 48.6105 | 48.6105 |
| 2011-08-04 | DEF | GLOBAL_LT_097MA_DEF | DEF | GLOBAL_LT_097MA_DEF | 352656.2 | 215851.24 | 47.6277 | 47.6277 |
| 2011-10-14 | AGG | DEF_GT_103MA_AGG | AGG | DEF_GT_103MA_AGG | 379464.46 | 219187.68 | 51.2482 | 51.2482 |
| 2011-10-17 | NOR | AGG_LT_INIT_NOR | NOR | AGG_LT_INIT_NOR | 361898.96 | 209041.43 | 50.4555 | 50.4555 |
| 2011-11-21 | DEF | GLOBAL_LT_097MA_DEF | DEF | GLOBAL_LT_097MA_DEF | 322886.68 | 186507.02 | 47.8658 | 47.8658 |
| 2012-01-05 | AGG | DEF_GT_103MA_AGG | AGG | DEF_GT_103MA_AGG | 343626.9 | 179961.54 | 50.9404 | 50.9404 |
| 2012-04-24 | NOR | AGG_TRAIL_NOR | NOR | AGG_TRAIL_NOR | 482488.89 | 252685.24 | 57.3332 | 57.3332 |
