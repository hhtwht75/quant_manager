import yfinance as yf
import pandas as pd
from alpha_vantage_data import *
# import time

def fetch_alpha(tickers, month):
    df = {}
    for ticker in tickers:
        try:
            data = alpha_request(ticker, month)
            df[ticker] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return df

def backtest_strategy(tickers, stock_data, initial_capital=100000, margin = 0.01, stop_loss=0.015, commission_rate=0.001, count_hold=1):
    capital = initial_capital
    previous_capital = initial_capital  # Capital after the previous sell
    holdings = 0  # Number of stocks held
    opening_time = stock_data[tickers[0]].index[0].time()
    closing_time = stock_data[tickers[0]].index[-1].time()
    bought_today = False  # Flag to check if a stock was bought on the current date

    date_save = None

    for date in stock_data[tickers[0]].index:

        if date_save == date.date():
            count += 1
        else:
            count = 0
            date_save = date.date()

        # print(count)
        # print(date.date())
        
        # if date.time() == opening_time:
        if count == 0:

            opening_price = []
            bought_today = False

            for ticker in tickers:   
                opening_price.append(stock_data[ticker].loc[date, "Open"])

        for ticker in tickers:

            if holdings == 0 and bought_today == False:

                if stock_data[ticker].loc[date, "High"] >= (1 + margin) * opening_price[tickers.index(ticker)]:
                    bought_ticker = ticker
                    buy_price = (1 + margin) * opening_price[tickers.index(ticker)]
                    num_stocks = capital // (buy_price * (1 + commission_rate))
                    capital -= num_stocks * buy_price * (1 + commission_rate)
                    holdings += num_stocks
                    bought_today = True
                    # print(f"Bought {num_stocks} of {ticker} at ${buy_price:.2f} on {date}")
                    break

        if holdings > 0:

            # if stock_data[bought_ticker].loc[date, "High"] >= buy_price * (1+margin2):
            #     sell_price = buy_price * (1+margin2)
            #     capital += (holdings * sell_price) * (1 - commission_rate)
            #     accumulated_return = ((capital - initial_capital) / initial_capital) * 100
            #     change_rate = ((capital - previous_capital) / previous_capital) * 100
            #     print(f"Sold {bought_ticker} at ${sell_price:.2f} at Margin GET on {date}")
            #     print(f"Change rate since last sale: {change_rate:.2f}%")
            #     # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
            #     # print(f"     ")
            #     previous_capital = capital
            #     holdings = 0

            if stock_data[bought_ticker].loc[date, "Low"] <= buy_price * (1-stop_loss):
                # if True:
                # if date.time() != opening_time:
                if count >= count_hold:
                    sell_price = min(buy_price * (1 - stop_loss), stock_data[bought_ticker].loc[date, "Open"])
                    capital += (holdings * sell_price) * (1 - commission_rate)
                    accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                    change_rate = ((capital - previous_capital) / previous_capital) * 100
                    # print(f"Sold {bought_ticker} at ${sell_price:.2f} due to 1.5% STOP LOSS rule on {date}")
                    # print(f"Change rate since last sale: {change_rate:.2f}%")
                    # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                    # print(f"     ")
                    previous_capital = capital
                    holdings = 0
                elif stock_data[bought_ticker].loc[date, "Low"] <= buy_price * 0.97:
                    sell_price = buy_price * 0.97
                    capital += (holdings * sell_price) * (1 - commission_rate)
                    accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                    change_rate = ((capital - previous_capital) / previous_capital) * 100
                    # print(f"Sold {bought_ticker} at ${sell_price:.2f} due to 3% STOP LOSS rule on {date}")
                    # print(f"Change rate since last sale: {change_rate:.2f}%")
                    # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                    # print(f"     ")
                    # print("3% STOP LOSSSSSSSSSSSS")
                    previous_capital = capital
                    holdings = 0

                else: #Stoploss even if during hold time
                    if stock_data[bought_ticker].loc[date, "Low"] <= buy_price * (1-stop_loss2):
                        sell_price = buy_price * (1 - stop_loss2)
                        capital += (holdings * sell_price) * (1 - commission_rate)
                        accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                        change_rate = ((capital - previous_capital) / previous_capital) * 100
                        # print(f"Sold {bought_ticker} at ${sell_price:.2f} due to 5% STOP LOSS rule on {date}")
                        # print(f"Change rate since last sale: {change_rate:.2f}%")
                        # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                        # print(f"     ")
                        previous_capital = capital
                        holdings = 0

            elif date.time() == closing_time:
                sell_price = stock_data[bought_ticker].loc[date, "Close"]
                capital += (holdings * sell_price) * (1 - commission_rate)
                accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                change_rate = ((capital - previous_capital) / previous_capital) * 100
                # print(f"Sold {bought_ticker} at ${sell_price:.2f} at END OF DAY on {date}")
                # print(f"Change rate since last sale: {change_rate:.2f}%")
                # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                # print(f"     ")
                previous_capital = capital
                holdings = 0

        if date.time() == closing_time:
            bought_today = False  # Reset the flag for the next day
        
    profit_or_loss = capital - initial_capital
    return accumulated_return

# whole_tickers = ["TQQQ", "SQQQ", "TMV", "TMf", "TYO", "TYD", "YANG", "YINN", "EDZ", "EDC", "TZA", "TNA", "WEBL", "WEBS", "FAZ", "FAS", "DRV", "DRN", "HIBS", "HIBL", "LABD", "LABU", "SOXL", "SOXS", "TECL", "TECS"]
# whole_tickers = ["TQQQ", "SQQQ", "LABD", "LABU", "SOXL", "SOXS", "FNGU", "FNGD"]
# whole_tickers = ["TQQQ", "SQQQ","SOXL", "SOXS", "LABD", "LABU", "FNGU", "FNGD"]
# whole_tickers = ["TQQQ", "SQQQ", "SOXL", "SOXS"]
# whole_tickers = ["LABD", "LABU", "SOXL", "SOXS"]
# whole_tickers = ["SOXL", "SOXS", "LABD", "LABU"]

# whole_tickers = ["TQQQ", "SQQQ"]
# whole_tickers = ["LABD", "LABU"]
whole_tickers = ["SOXS", "SOXL"]
# whole_tickers = ["SOXL", "SOXS"]
# whole_tickers = ["TYD", "TYO"]

start_month = "2023-02"
end_month = "2024-01"
start_month_pd = pd.to_datetime(start_month)
end_month_pd = pd.to_datetime(end_month)
sim_result = {}

current = start_month_pd

while current <= end_month_pd:

    input_month = current.strftime("%Y-%m")
    print(input_month)

    month = input_month
    initial_capital = 10000
    margin = 0.010
    margin2 = 0.05
    stop_loss = 0.015
    stop_loss2 = 0.05
    commission_rate = 0.001
    count_hold = 4

    tickers = whole_tickers
    stock_data = fetch_alpha(tickers, month)
    accumulated_return = backtest_strategy(tickers, stock_data, initial_capital, margin, stop_loss, commission_rate, count_hold)

    sim_result[input_month] = accumulated_return
    print(input_month, accumulated_return)

    current += pd.DateOffset(month=1)

for date, result in sim_result.items():
    print(date, result)