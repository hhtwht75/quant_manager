# Monte Carlo 4전략 비교: `--root` 1999-03-10 vs 2002-10-01

공통 설정: 최소 무작위 창 **3년**(거래일 ≥756), 시행 **3000회**, `seed=42`, `trail=0.85`.  
무작위 창은 시행마다 동일한 (시작·종료)를 **MABS · RMABS-4tier-Gold · DMABS-4tier-Gold(방어=QQQ)** · **DMABS(방어·노말=QLD)** 네 NAV에 동시에 슬라이스합니다.

| `--root` | 전략 | CAGR 평균 | CAGR 중앙 | MDD 평균 | MDD 중앙 | Sharpe 평균 | Sharpe 중앙 | Sortino 평균 | Sortino 중앙 | Ulcer 평균 | Ulcer 중앙 |
|----------|------|----------:|----------:|----------:|----------:|------------:|-----------:|-------------:|------------:|----------:|-----------:|
| 1999-03-10 | MABS-QQQ | 26.39% | 28.40% | -52.69% | -48.46% | 0.713 | 0.763 | 0.938 | 0.993 | 21.7171 | 19.3669 |
| 1999-03-10 | RMABS-4tier-Gold | 25.50% | 27.43% | -50.63% | -49.82% | 0.703 | 0.771 | 0.901 | 0.962 | 20.1883 | 16.7875 |
| 1999-03-10 | DMABS-4tier-Gold | 28.24% | 29.89% | -46.20% | -49.82% | 0.769 | 0.825 | 0.983 | 1.027 | 17.8467 | 15.4713 |
| 1999-03-10 | DMABS-4tier-Gold (방어·노말=QLD) | 29.96% | 31.65% | -45.93% | -49.82% | 0.804 | 0.857 | 1.036 | 1.077 | 17.3264 | 15.2553 |
| 2002-10-01 | MABS-QQQ | 29.68% | 30.51% | -48.40% | -48.46% | 0.786 | 0.809 | 1.033 | 1.043 | 17.8247 | 17.6928 |
| 2002-10-01 | RMABS-4tier-Gold | 28.52% | 29.82% | -47.53% | -49.82% | 0.774 | 0.822 | 0.989 | 1.018 | 17.2520 | 16.2664 |
| 2002-10-01 | DMABS-4tier-Gold | 30.99% | 31.93% | -44.10% | -49.82% | 0.834 | 0.868 | 1.064 | 1.072 | 15.7316 | 15.0298 |
| 2002-10-01 | DMABS-4tier-Gold (방어·노말=QLD) | 32.78% | 33.58% | -43.88% | -49.82% | 0.868 | 0.901 | 1.117 | 1.119 | 15.2455 | 14.7269 |

**DMABS-4tier-Gold (방어·노말=QLD)** 레그: 신호·MA200·규칙4·MA5/MA120 스트레스 동일, `_align_five(…)` 에서 **Close_def=Close_nor=QLD**, **Close_sig=QQQ**, **Close_safe=금(`GC=F` 등)**, **Close_agg=TQQQ**.

원본 산출물(4전략 동시 MC, 동일 창):

- `03_RESULT/sensitivity/mc_four_mabs_rmabs_dmabs_dmabsDnq_fullix_19990310_20260430_3yr_n3000_s42_*` (교집합 6809일)
- `03_RESULT/sensitivity/mc_four_mabs_rmabs_dmabs_dmabsDnq_fullix_20021001_20260430_3yr_n3000_s42_*` (교집합 5918일)

3전략만 필요할 때(변형 제외): `--triple-mabs-rmabs-dmagold` 에서 `--include-dmabs-gold-defnor-qld` 를 빼면 기존 파일 스템(`mc_three_mabs_rmabs_dmabsGold_fullix_…`)이 유지된다.

재실행 예:

```bash
python3 01_CODE/fsm_mc_suite_fsm3_rmabs2_bench4.py \
  --triple-mabs-rmabs-dmagold --include-dmabs-gold-defnor-qld \
  --root 1999-03-10 --mc-years 3 --mc-iters 3000 --mc-seed 42

python3 01_CODE/fsm_mc_suite_fsm3_rmabs2_bench4.py \
  --triple-mabs-rmabs-dmagold --include-dmabs-gold-defnor-qld \
  --root 2002-10-01 --mc-years 3 --mc-iters 3000 --mc-seed 42
```
