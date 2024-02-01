# coding: utf-8
# API: UKEEHPB4Z00E8C7Y
import requests
import datetime
import pandas as pd
import yfinance as yf

def fetch_alpha(tickers, month):
    df = {}
    for ticker in tickers:
        try:
            data = alpha_request(ticker, month)
            df[ticker] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return df

def alpha_request(ticker='SOXL', month='2024-01'):

    # url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=UKEEHPB4Z00E8C7Y'
    url = 'https://www.alphavantage.co/query'
    params = {
        'function' : 'TIME_SERIES_INTRADAY',
        'symbol': ticker,
        'interval': '5min', # 1min, 5min, 15min, 30min, 60min
        'extended_hours' : 'false', 
        'month' : month,
        'outputsize': 'full',
        'apikey': 'UKEEHPB4Z00E8C7Y'
    }

    r = requests.get(url, params=params)
    data = r.json()

    # print(data)

    time_series_data = data['Time Series (5min)']
    df = pd.DataFrame.from_dict(time_series_data, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'})
    df = df.sort_index() 
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(int)

    return df

def fetch_stock_data(tickers=['SOXL'], interval="15m", start_date="2024-01-01", end_date="2024-01-31"):
    df = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, interval=interval, start=start_date, end=end_date)
            df[ticker] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return df

# pd.set_option('display.max_columns', 500)
# print(fetch_stock_data())
print(alpha_request('SOXL','2024-01'))
