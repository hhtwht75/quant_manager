import yfinance as yf
import pandas as pd

# 데이터 다운로드
start_date = '2023-01-01'
end_date = '2023-09-01'
long_ticker = 'TSLA'
short_ticker = 'TSLS'
long = yf.download(long_ticker, start=start_date, end=end_date)
short = yf.download(short_ticker, start=start_date, end=end_date)

# 전략 실행
portfolio_value = 100000  # 초기 포트폴리오 가치
cash = portfolio_value
positions = {long_ticker: 0, short_ticker: 0}
loss_threshold = 0.01  # 1% 손실
transaction_fee_rate = 0.0002  # 0.1% 수수료

# 날짜별 cash 값을 저장할 딕셔너리
cash_history = {}

for date in long.index:
    # 주식 구매
    if cash > 0:
        buy_amount_tqqq = (0.5 * cash) / (1 + transaction_fee_rate)
        buy_amount_sqqq = (0.5 * cash) / (1 + transaction_fee_rate)

        purchased_shares_tqqq = buy_amount_tqqq // long.loc[date, 'Open']
        purchased_shares_sqqq = buy_amount_sqqq // short.loc[date, 'Open']

        positions[long_ticker] += purchased_shares_tqqq
        positions[short_ticker] += purchased_shares_sqqq

        actual_cost_tqqq = purchased_shares_tqqq * long.loc[date, 'Open'] * (1 + transaction_fee_rate)
        actual_cost_sqqq = purchased_shares_sqqq * short.loc[date, 'Open'] * (1 + transaction_fee_rate)
        
        cash -= (actual_cost_tqqq + actual_cost_sqqq)

    # 손실 체크
    for ticker in positions:
        if positions[ticker] > 0:
            if ticker == long_ticker:
                open_price = long.loc[date, 'Open']
                threshold_price = open_price * (1 - loss_threshold)
                low_price = long.loc[date, 'Low']
            else:
                open_price = short.loc[date, 'Open']
                threshold_price = open_price * (1 - loss_threshold)
                low_price = short.loc[date, 'Low']

            if low_price <= threshold_price:
                cash += positions[ticker] * threshold_price * (1 - transaction_fee_rate)
                positions[ticker] = 0

    # 장 마감 시 청산
    cash += positions[long_ticker] * long.loc[date, 'Close'] * (1 - transaction_fee_rate)
    cash += positions[short_ticker] * short.loc[date, 'Close'] * (1 - transaction_fee_rate)
    positions = {long_ticker: 0, short_ticker: 0}
    
    # 날짜별 cash 값을 저장
    cash_history[date] = cash

# 날짜별 cash 값 및 주식 가격 출력
for date, value in cash_history.items():
    tqqq_info = long.loc[date, ['Open', 'Low', 'Close']]
    sqqq_info = short.loc[date, ['Open', 'Low', 'Close']]
    
    print(f"{date}:")
    print(f"Cash Value: ${value:.2f}")
    # print(f"TQQQ - Open: ${tqqq_info['Open']:.2f}, Low: ${tqqq_info['Low']:.2f}, Close: ${tqqq_info['Close']:.2f}")
    # print(f"SQQQ - Open: ${sqqq_info['Open']:.2f}, Low: ${sqqq_info['Low']:.2f}, Close: ${sqqq_info['Close']:.2f}")
    # print("-" * 50)  # 출력 구분선
