import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 종목과 기간 정의
tickers = ["SOXS", "LABD"]
start_date = "2023-01-01"
end_date = "2023-12-31"

# 주식 데이터 가져오기
data = yf.download(tickers, start=start_date, end=end_date)

# 개장가와 종가 분리
soxl_open = data['Open']['SOXS']
soxl_close = data['Close']['SOXS']
labu_open = data['Open']['LABD']
labu_close = data['Close']['LABD']

# 일일 수익률 계산
soxl_returns = (soxl_close - soxl_open) / soxl_open * 100
labu_returns = (labu_close - labu_open) / labu_open * 100

# SOXL 수익률이 양수인 날만 필터링
positive_soxl_days = soxl_returns > 0

# 필터링된 날에 대해 SOXL과 LABU 수익률의 합 계산
combined_positive_returns = soxl_returns[positive_soxl_days] + labu_returns[positive_soxl_days]

# 그래프 그리기
plt.figure(figsize=(15, 7))

# SOXL과 LABU 수익률의 합
plt.bar(combined_positive_returns.index, combined_positive_returns, label='SOXL + LABU 수익률 합', color='blue', alpha=0.6)

# SOXL의 수익률만
plt.bar(soxl_returns[positive_soxl_days].index, soxl_returns[positive_soxl_days], label='SOXL 수익률', color='green', alpha=0.6)

# 그래프 제목 및 레이블 설정
plt.title('SOXL 수익률이 양수인 날의 SOXL과 LABU 수익률 비교')
plt.xlabel('날짜')
plt.ylabel('수익률 (%)')
plt.legend()

# 그래프 표시
plt.show()