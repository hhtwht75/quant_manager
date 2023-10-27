import yfinance as yf
import pandas as pd


# def fetch_stock_data(tickers, interval="15m", period="60d"):
#     df = {}
#     for ticker in tickers:
#         try:
#             data = yf.download(ticker, interval=interval, period=period)
#             df[ticker] = data
#         except Exception as e:
#             print(f"Error fetching data for {ticker}: {e}")
#     return df

def fetch_stock_data(tickers, interval="15m", start_date=None, end_date=None):
    df = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, interval=interval, start=start_date, end=end_date)
            df[ticker] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return df

def backtest_strategy(tickers, stock_data, initial_capital=100000, stop_loss=0.015, commission_rate=0.001):
    capital = initial_capital
    previous_capital = initial_capital  # Capital after the previous sell
    holdings = 0  # Number of stocks held
    closing_time = stock_data[tickers[0]].index[-1].time()
    bought_today = False  # Flag to check if a stock was bought on the current date

    for date in stock_data[tickers[0]].index:

        if holdings > 0:
            if stock_data[bought_ticker].loc[date, "Low"] <= buy_price * 0.985:
                sell_price = buy_price * (1 - stop_loss)
                capital += (holdings * sell_price) * (1 - commission_rate)
                holdings = 0
                accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                change_rate = ((capital - previous_capital) / previous_capital) * 100
                # print(f"Sold {bought_ticker} at ${sell_price:.2f} due to 1.5% loss rule on {date}")
                # print(f"Change rate since last sale: {change_rate:.2f}%")
                # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                # print(f"     ")
                previous_capital = capital

            elif date.time() == closing_time:
                sell_price = stock_data[bought_ticker].loc[date, "Close"]
                capital += (holdings * sell_price) * (1 - commission_rate)
                holdings = 0
                accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                change_rate = ((capital - previous_capital) / previous_capital) * 100
                # print(f"Sold {bought_ticker} at ${sell_price:.2f} at end of day on {date}")
                # print(f"Change rate since last sale: {change_rate:.2f}%")
                # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                # print(f"     ")

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
                    # print(f"Bought {num_stocks} of {ticker} at ${buy_price:.2f} on {date}")
                    bought_today = True
                    break

        if date.time() == closing_time:
            bought_today = False  # Reset the flag for the next day
        
    profit_or_loss = capital - initial_capital
    return accumulated_return

# whole_tickers = ["TQQQ", "SQQQ", "TMV", "TMf", "TYO", "TYD", "YANG", "YINN", "EDZ", "EDC", "TZA", "TNA", "WEBL", "WEBS", "FAZ", "FAS", "DRV", "DRN", "HIBS", "HIBL", "LABD", "LABU", "SOXL", "SOXS", "TECL", "TECS"]
whole_tickers = ["TQQQ", "SQQQ", "LABD", "LABU", "SOXL", "SOXS", "TECL", "TECS", "FNGU", "FNGD"]
# whole_tickers = ["TYD", "TYO"]


average_rate = []

for i in range(0, len(whole_tickers) // 2):
    
    interval = "1h"
    # period = "20d"
    start_date = "2023-08-01"
    end_date = "2023-09-01"
    initial_capital = 10000
    stop_loss = 0.01
    commission_rate = 0.000

    tickers = whole_tickers[i*2 : i*2 + 2]
    stock_data = fetch_stock_data(tickers, interval, start_date, end_date)

    accumulated_return = backtest_strategy(tickers, stock_data, initial_capital, stop_loss, commission_rate)
    # print(f"Initial Capital: ${initial_capital:.2f}")
    # print(f"Final Capital: ${final_capital:.2f}")
    # print(f"Profit or Loss: ${profit_or_loss:.2f}")

    print(tickers)
    print(f"Accumulated Return: {accumulated_return:.2f}%")
    print(f"      ")

    average_rate.append(accumulated_return)

average_rate_value = sum(average_rate)/len(average_rate)
print(f"Average Return: {average_rate_value:.2f}%")
