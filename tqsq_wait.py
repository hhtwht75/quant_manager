import yfinance as yf
import pandas as pd
import random

# 데이터 다운로드
start_date = '2022-09-01'
end_date = '2023-09-01'
long_ticker = 'TSLA'
short_ticker = 'TSLS'
full_long_data = yf.download(long_ticker, start=start_date, end=end_date)
full_short_data = yf.download(short_ticker, start=start_date, end=end_date)

# 초기 설정
gain_threshold = 0.02  # 상승
transaction_fee_rate = 0.002  # 수수료
iterations = 40

profits = []
max_losses = []


for _ in range(iterations):
    # 랜덤한 시작 날짜와 랜덤한 구간 길이 선택
    # random_length = random.randint(30, 200)
    random_length = 30
    random_start_index = random.randint(0, len(full_long_data) - random_length - 1)
    random_end_index = random_start_index + random_length

    # 랜덤한 데이터 구간 추출
    long = full_long_data.iloc[random_start_index:random_end_index]
    short = full_short_data.iloc[random_start_index:random_end_index]

    # 초기 포트폴리오 가치
    cash = portfolio_value = 100000  

    # 날짜별로 전략 실행
    for date in long.index:
        # long position에 대한 로직
        purchase_price_long = long.loc[date, 'Open'] * (1 + gain_threshold)
        if long.loc[date, 'High'] >= purchase_price_long:
            buy_amount = (0.5 * cash) / (1 + transaction_fee_rate)
            purchased_shares = buy_amount // purchase_price_long
            cash -= purchased_shares * purchase_price_long * (1 + transaction_fee_rate)
            cash += purchased_shares * long.loc[date, 'Close'] * (1 - transaction_fee_rate)

        # # short position에 대한 로직
        # purchase_price_short = short.loc[date, 'Open'] * (1 + gain_threshold)
        # if short.loc[date, 'High'] >= purchase_price_short:
        #     buy_amount = (0.5 * cash) / (1 + transaction_fee_rate)
        #     purchased_shares = buy_amount // purchase_price_short
        #     cash -= purchased_shares * purchase_price_short * (1 + transaction_fee_rate)
        #     cash += purchased_shares * short.loc[date, 'Close'] * (1 - transaction_fee_rate)

    # 각 랜덤 데이터 구간의 마지막 날짜에서 수익률과 최대 손실 계산
    profits.append((cash - portfolio_value) / portfolio_value)
    max_losses.append(min(0, cash - portfolio_value))  # 손실만 저장

# 결과 출력
for i, (profit, max_loss) in enumerate(zip(profits, max_losses)):
    print(f"Iteration {i+1}: Profit: {profit*100:.2f}")
