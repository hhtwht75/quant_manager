import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers, interval="5m", period="10d"):
    df = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, interval=interval, period=period)
            df[ticker] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return df

# tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
tickers = ["TQQQ", "SQQQ"]
# tickers = ["TSLA", "TSLQ"]


stock_data = fetch_stock_data(tickers)

def backtest_strategy(stock_data, initial_capital=100000):
    capital = initial_capital
    previous_capital = initial_capital  # Capital after the previous sell
    holdings = 0  # Number of stocks held
    commission_rate = 0.002  # Commission rate
    closing_time = stock_data[tickers[0]].index[-1].time()
    bought_today = False  # Flag to check if a stock was bought on the current date

    for date in stock_data[tickers[0]].index:

        if holdings > 0:
            if stock_data[bought_ticker].loc[date, "Low"] <= buy_price * 0.985:
                sell_price = buy_price * 0.985
                capital += (holdings * sell_price) * (1 - commission_rate)
                holdings = 0
                accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                change_rate = ((capital - previous_capital) / previous_capital) * 100
                print(f"Sold {bought_ticker} at ${sell_price:.2f} due to 1.5% loss rule on {date}")
                print(f"Change rate since last sale: {change_rate:.2f}%")
                print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                print(f"     ")
                previous_capital = capital

            elif date.time() == closing_time:
                sell_price = stock_data[bought_ticker].loc[date, "Close"]
                capital += (holdings * sell_price) * (1 - commission_rate)
                holdings = 0
                accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                change_rate = ((capital - previous_capital) / previous_capital) * 100
                print(f"Sold {bought_ticker} at ${sell_price:.2f} at end of day on {date}")
                print(f"Change rate since last sale: {change_rate:.2f}%")
                print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                print(f"     ")

                previous_capital = capital

            
        elif not bought_today:
            for ticker in tickers:
                opening_price = stock_data[ticker].loc[date, "Open"]
                if stock_data[ticker].loc[date, "High"] >= 1.01 * opening_price:
                    bought_ticker = ticker
                    buy_price = 1.01 * opening_price
                    num_stocks = capital // (buy_price * (1 + commission_rate))
                    capital -= num_stocks * buy_price * (1 + commission_rate)
                    holdings += num_stocks
                    print(f"Bought {num_stocks} of {ticker} at ${buy_price:.2f} on {date}")
                    bought_today = True
                    break

        if date.time() == closing_time:
            bought_today = False  # Reset the flag for the next day
        
    profit_or_loss = capital - initial_capital
    return capital, profit_or_loss

initial_capital = 100000

final_capital, profit_or_loss = backtest_strategy(stock_data, initial_capital)
print(f"Initial Capital: ${initial_capital:.2f}")
print(f"Final Capital: ${final_capital:.2f}")
print(f"Profit or Loss: ${profit_or_loss:.2f}")