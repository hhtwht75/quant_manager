import yfinance as yf
import pandas as pd

# 종목 설정 및 기간 설정
tickers = ["JEPI", "SCHD", "O", "IFN"]
start_date = '2021-01-01'
end_date = '2023-09-30'
investment_per_stock = 1000

# 데이터 다운로드
price_data = {}
dividend_data = {}
for ticker in tickers:
    stock_data = yf.Ticker(ticker)
    price_data[ticker] = stock_data.history(start=start_date, end=end_date)
    dividend_data[ticker] = stock_data.dividends
print(f"Start Date: {start_date}")
print(f"End Date: {end_date}")

# 전략 실행
total_dividends = 0
positions = {ticker: 0 for ticker in tickers}

for date in pd.date_range(start_date, end_date, freq='D'):
    localized_date = date.tz_localize('America/New_York')  # 시간대 정보를 포함하는 datetime 객체로 변환
    for ticker in tickers:
        # 해당 날짜의 배당 데이터만 필터링
        if date in dividend_data[ticker].index:
            daily_dividend = dividend_data[ticker][date]
            total_dividends += daily_dividend * positions[ticker]
            print(f"{date}: {ticker} Dividend = {daily_dividend}, Total Dividends = {total_dividends}")

        # 매월 첫 영업일에만 투자
        if date.is_month_start and date in price_data[ticker].index:
            positions[ticker] += investment_per_stock / price_data[ticker].loc[date, 'Open']
            print(f"{date}: {ticker} Invested, Position = {positions[ticker]}")

# 최종 자산 계산
final_portfolio_value = sum(positions[ticker] * price_data[ticker].iloc[-1]['Close'] for ticker in tickers) + total_dividends
initial_investment = len(tickers) * investment_per_stock * len(pd.date_range(start_date, end_date, freq='M'))
return_rate = (final_portfolio_value - initial_investment) / initial_investment

print(f"Total Dividend Income: ${total_dividends:.2f}")
print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
print(f"Return Rate: {return_rate*100:.2f}%")
