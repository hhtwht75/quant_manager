import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers, interval="1d", start_date=None, end_date=None):
    df = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, interval=interval, start=start_date, end=end_date)
            df[ticker] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return df

def backtest_strategy(tickers, stock_data, initial_capital=100000, commission_rate=0.001):
    capital = initial_capital
    holdings = 0  # Number of stocks held

    for date in stock_data[tickers[0]].index:
        
        for ticker in tickers:   
            sell_price = stock_data[ticker].loc[date, "Open"]
            capital += (holdings * sell_price) * (1 - commission_rate)
            accumulated_return = ((capital - initial_capital) / initial_capital) * 100
            holdings = 0

            # change_rate = ((capital - previous_capital) / previous_capital) * 100
            print(f"Sold {ticker} at ${sell_price:.2f} on {date}")
            # print(f"Change rate since last sale: {change_rate:.2f}%")
            print(f"Accumulated return after sale: {accumulated_return:.2f}%")
            # print(f"     ")
            # previous_capital = capital
    
            buy_price = stock_data[ticker].loc[date, "Close"]
            num_stocks = capital // (buy_price * (1 + commission_rate))
            capital -= num_stocks * buy_price * (1 + commission_rate)
            holdings += num_stocks
            print(f"Bought {num_stocks} of {ticker} at ${buy_price:.2f} on {date}")
       
    # profit_or_loss = capital - initial_capital
    return accumulated_return

whole_tickers = ["SPY"]
average_rate = []

for i in range(0,1):
    
    interval = "1d"
    # period = "20d"
    start_date = "2023-12-01"
    end_date = "2023-12-08"
    initial_capital = 10000
    commission_rate = 0.001

    tickers = whole_tickers
    stock_data = fetch_stock_data(tickers, interval, start_date, end_date)
  
    accumulated_return = backtest_strategy(tickers, stock_data, initial_capital, commission_rate)
    # print(f"Initial Capital: ${initial_capital:.2f}")
    # print(f"Final Capital: ${final_capital:.2f}")
    # print(f"Profit or Loss: ${profit_or_loss:.2f}")

    print(tickers)
    print(f"Accumulated Return: {accumulated_return:.2f}%")
    print(f"      ")

    average_rate.append(accumulated_return)

average_rate_value = sum(average_rate)/len(average_rate)
print(f"Average Return: {average_rate_value:.2f}%")
