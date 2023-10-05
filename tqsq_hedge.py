import yfinance as yf
import pandas as pd

# 데이터 다운로드
start_date = '2020-01-01'
end_date = '2023-01-01'
tqqq = yf.download('TQQQ', start=start_date, end=end_date)
sqqq = yf.download('SQQQ', start=start_date, end=end_date)

# 전략 실행
portfolio_value = 100000  # 초기 포트폴리오 가치
cash = portfolio_value
positions = {'TQQQ': 0, 'SQQQ': 0}
loss_threshold = 0.02  # 2% 손실
transaction_fee_rate = 0.001  # 0.1% 수수료

print(tqqq)

for date in tqqq.index:
    # 주식 구매
    if cash > 0:
        buy_amount_tqqq = (0.5 * cash) / (1 + transaction_fee_rate)
        buy_amount_sqqq = (0.5 * cash) / (1 + transaction_fee_rate)

        positions['TQQQ'] += buy_amount_tqqq / tqqq.loc[date, 'Open']
        positions['SQQQ'] += buy_amount_sqqq / sqqq.loc[date, 'Open']
        
        cash -= (buy_amount_tqqq + buy_amount_sqqq) * (1 + transaction_fee_rate)

    # 손실 체크
    for ticker in positions:
        if positions[ticker] > 0:
            if ticker == 'TQQQ':
                current_price = tqqq.loc[date, 'Low']
            else:
                current_price = sqqq.loc[date, 'Low']

            if (current_price / tqqq.loc[date, 'Open']) - 1 <= -loss_threshold:
                cash += positions[ticker] * current_price * (1 - transaction_fee_rate)
                positions[ticker] = 0

    # 장 마감 시 청산
    if date == tqqq.index[-1] or date.day != tqqq.index[tqqq.index.get_loc(date) + 1].day:
        cash += positions['TQQQ'] * tqqq.loc[date, 'Close'] * (1 - transaction_fee_rate)
        cash += positions['SQQQ'] * sqqq.loc[date, 'Close'] * (1 - transaction_fee_rate)
        positions = {'TQQQ': 0, 'SQQQ': 0}

print(f"Final Portfolio Value: ${cash:.2f}")