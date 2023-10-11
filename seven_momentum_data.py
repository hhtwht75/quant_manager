import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers, interval="15m", period="60d"):
    df = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, interval=interval, period=period)
            df[ticker] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return df

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
stock_data = fetch_stock_data(tickers)


def backtest_strategy(stock_data, initial_capital=100000):
    capital = initial_capital
    holdings = 0  # 보유 주식 수
    commission_rate = 0.002  # 거래 수수료 0.2%
    closing_time = stock_data[tickers[0]].index[-1].time()

    for date in stock_data[tickers[0]].index:
        if holdings > 0:
            if stock_data[bought_ticker].loc[date, "Low"] <= buy_price * 0.985:
                sell_price = buy_price * 0.985
                capital += (holdings * sell_price) * (1 - commission_rate)
                holdings = 0
                print(f"Sold {bought_ticker} at ${sell_price:.2f} due to 1.5% loss rule on {date}")
                print(f"Capital on {date.strftime('%Y-%m-%d %H:%M:%S')}: ${capital:.2f}")
            elif date.time() == closing_time:
                sell_price = stock_data[bought_ticker].loc[date, "Close"]
                capital += (holdings * sell_price) * (1 - commission_rate)
                holdings = 0
                print(f"Sold {bought_ticker} at ${sell_price:.2f} at end of day on {date}")
                print(f"Capital on {date.strftime('%Y-%m-%d %H:%M:%S')}: ${capital:.2f}")

        else:
            for ticker in tickers:
                opening_price = stock_data[ticker].loc[date, "Open"]
                if stock_data[ticker].loc[date, "High"] >= 1.01 * opening_price and holdings == 0:
                    bought_ticker = ticker
                    buy_price = 1.01 * opening_price
                    num_stocks = capital // (buy_price * (1 + commission_rate))
                    capital -= num_stocks * buy_price * (1 + commission_rate)
                    holdings += num_stocks
                    print(f"Bought {num_stocks} of {ticker} at ${buy_price:.2f} on {date}")
                    break
        
    profit_or_loss = capital - initial_capital
    return capital, profit_or_loss

initial_capital = 10000

final_capital, profit_or_loss = backtest_strategy(stock_data, initial_capital)
print(f"Initial Capital: ${initial_capital:.2f}")
print(f"Final Capital: ${final_capital:.2f}")
print(f"Profit or Loss: ${profit_or_loss:.2f}")
