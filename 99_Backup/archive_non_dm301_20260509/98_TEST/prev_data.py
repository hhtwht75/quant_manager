import yfinance as yf
import datetime

def get_last_trading_day_data(ticker_symbol):
    # 오늘 날짜와 어제 날짜 계산
    today = datetime.datetime.now()
    
    # Yahoo Finance에서 오늘 날짜까지의 데이터 다운로드
    data = yf.download(ticker_symbol, start=(today - datetime.timedelta(30)).strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
    
    # 데이터의 마지막 행의 날짜 가져오기
    last_data_date = data.index[-1].date()
    
    # 마지막 행의 날짜가 오늘이라면, 마지막 거래일은 그 전 행
    if last_data_date == today.date():
        last_trading_day_data = data.iloc[-2]
    else:
        # 마지막 행이 마지막 거래일
        last_trading_day_data = data.iloc[-1]
    
    return last_trading_day_data

# 여러 티커
tickers = ["SOXL", "SOXS", "TQQQ", "SQQQ"]

# 각 티커의 마지막 거래일 시가와 종가 저장하기
ticker_data = {}

for ticker in tickers:
    data = get_last_trading_day_data(ticker)
    ticker_data[ticker] = {"Open": data['Open'], "Close": data['Close']}

print(ticker_data["SQQQ"]["Close"])