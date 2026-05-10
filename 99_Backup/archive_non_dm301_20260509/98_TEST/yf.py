import yfinance as yf

def fetch_stock_data(tickers, interval="15m", start_date=None, end_date=None):
    df = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, interval=interval, start=start_date, end=end_date)
            df[ticker] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return df

def print_stock_data(data):
    for ticker, df in data.items():
        print(f"Data for {ticker}:")
        for index, row in df.iterrows():
            print(index, row.to_dict())

data = fetch_stock_data(('SOXS',), '1m', '2024-02-29', '2024-03-01')
print_stock_data(data)
