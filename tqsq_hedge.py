import yfinance as yf
import pandas as pd

# 데이터 다운로드
start_date = '2023-01-01'
end_date = '2023-09-01'
tqqq = yf.download('TQQQ', start=start_date, end=end_date)
sqqq = yf.download('SQQQ', start=start_date, end=end_date)

# 전략 실행
portfolio_value = 100000  # 초기 포트폴리오 가치
cash = portfolio_value
positions = {'TQQQ': 0, 'SQQQ': 0}
loss_threshold = 0.01  # 1% 손실
transaction_fee_rate = 0.0002  # 0.1% 수수료

# 날짜별 cash 값을 저장할 딕셔너리
cash_history = {}

for date in tqqq.index:
    # 주식 구매
    if cash > 0:
        buy_amount_tqqq = (0.5 * cash) / (1 + transaction_fee_rate)
        buy_amount_sqqq = (0.5 * cash) / (1 + transaction_fee_rate)

        purchased_shares_tqqq = buy_amount_tqqq // tqqq.loc[date, 'Open']
        purchased_shares_sqqq = buy_amount_sqqq // sqqq.loc[date, 'Open']

        positions['TQQQ'] += purchased_shares_tqqq
        positions['SQQQ'] += purchased_shares_sqqq

        actual_cost_tqqq = purchased_shares_tqqq * tqqq.loc[date, 'Open'] * (1 + transaction_fee_rate)
        actual_cost_sqqq = purchased_shares_sqqq * sqqq.loc[date, 'Open'] * (1 + transaction_fee_rate)
        
        cash -= (actual_cost_tqqq + actual_cost_sqqq)

    # 손실 체크
    for ticker in positions:
        if positions[ticker] > 0:
            if ticker == 'TQQQ':
                open_price = tqqq.loc[date, 'Open']
                threshold_price = open_price * (1 - loss_threshold)
                low_price = tqqq.loc[date, 'Low']
            else:
                open_price = sqqq.loc[date, 'Open']
                threshold_price = open_price * (1 - loss_threshold)
                low_price = sqqq.loc[date, 'Low']

            if low_price <= threshold_price:
                cash += positions[ticker] * threshold_price * (1 - transaction_fee_rate)
                positions[ticker] = 0

    # 장 마감 시 청산
    cash += positions['TQQQ'] * tqqq.loc[date, 'Close'] * (1 - transaction_fee_rate)
    cash += positions['SQQQ'] * sqqq.loc[date, 'Close'] * (1 - transaction_fee_rate)
    positions = {'TQQQ': 0, 'SQQQ': 0}
    
    # 날짜별 cash 값을 저장
    cash_history[date] = cash

# 날짜별 cash 값 및 주식 가격 출력
for date, value in cash_history.items():
    tqqq_info = tqqq.loc[date, ['Open', 'Low', 'Close']]
    sqqq_info = sqqq.loc[date, ['Open', 'Low', 'Close']]
    
    print(f"{date}:")
    print(f"Cash Value: ${value:.2f}")
    # print(f"TQQQ - Open: ${tqqq_info['Open']:.2f}, Low: ${tqqq_info['Low']:.2f}, Close: ${tqqq_info['Close']:.2f}")
    # print(f"SQQQ - Open: ${sqqq_info['Open']:.2f}, Low: ${sqqq_info['Low']:.2f}, Close: ${sqqq_info['Close']:.2f}")
    # print("-" * 50)  # 출력 구분선
