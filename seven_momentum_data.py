import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import yfinance as yf
import pandas as pd
from pandas.tseries.offsets import MonthBegin, MonthEnd

import alpha_vantage_data

def fetch_stock_data(tickers, interval="15m", start_date=None, end_date=None):
    df = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, interval=interval, start=start_date, end=end_date)
            df[ticker] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return df

def backtest_strategy(tickers, stock_data, initial_capital=100000, margin = 0.01, stop_loss=0.015, commission_rate=0.001, count_hold=1, count_end=25):
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
                # if count >= count_hold and stock_data[ticker].loc[date, "Open"] < (1 + margin) * opening_price[tickers.index(ticker)]:
                if stock_data[ticker].loc[date, "High"] >= (1 + margin) * opening_price[tickers.index(ticker)]:
                    bought_ticker = ticker
                    # buy_price = max(stock_data[ticker].loc[date, "Open"],(1 + margin) * opening_price[tickers.index(ticker)])
                    buy_price = (1 + margin) * opening_price[tickers.index(ticker)]
                    num_stocks = capital // (buy_price * (1 + commission_rate))
                    capital -= num_stocks * buy_price * (1 + commission_rate)
                    holdings += num_stocks
                    bought_today = True
                    # print(f"Bought {num_stocks} of {ticker} at ${buy_price:.2f} on {date}")
                    break

            # elif holdings == 0 and bought_today == True:
            #     if count < count_hold:
            #         if stock_data[ticker].loc[date, "High"] >= (1 + margin) * opening_price[tickers.index(ticker)]:
            #             bought_ticker = ticker
            #             # buy_price = max(stock_data[ticker].loc[date, "Open"],(1 + margin) * opening_price[tickers.index(ticker)])
            #             buy_price = (1 + margin) * opening_price[tickers.index(ticker)]
            #             num_stocks = (capital*2) // (3*(buy_price * (1 + commission_rate)))
            #             capital -= num_stocks * buy_price * (1 + commission_rate)
            #             holdings += num_stocks
            #             bought_today = True
            #             print(f"Bought {num_stocks} of {ticker} at ${buy_price:.2f} on {date}")
            #             break

        if holdings > 0:

            if stock_data[bought_ticker].loc[date, "Low"] <= buy_price * (1-stop_loss):
                # if True:
                # if date.time() != opening_time:
                if count >= count_hold:
                    sell_price = min(buy_price * (1 - stop_loss), stock_data[bought_ticker].loc[date, "Open"])
                    capital += (holdings * sell_price) * (1 - commission_rate)
                    accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                    change_rate = ((capital - previous_capital) / previous_capital) * 100
                    print(f"Sold {bought_ticker} at ${sell_price:.2f} due to 1.5% STOP LOSS rule on {date}")
                    # print(f"Change rate since last sale: {change_rate:.2f}%")
                    # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                    # print(f"     ")
                    previous_capital = capital
                    holdings = 0

                else: #Stoploss even if during hold time
                    if stock_data[bought_ticker].loc[date, "Low"] <= buy_price * (1-stop_loss2):
                        sell_price = buy_price * (1 - stop_loss2)
                        capital += (holdings * sell_price) * (1 - commission_rate)
                        accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                        change_rate = ((capital - previous_capital) / previous_capital) * 100
                        print(f"Sold {bought_ticker} at ${sell_price:.2f} due to 5% STOP LOSS rule on {date}")
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
                print(f"Sold {bought_ticker} at ${sell_price:.2f} at END OF DAY on {date}")
                # print(f"Change rate since last sale: {change_rate:.2f}%")
                # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                # print(f"     ")
                previous_capital = capital
                holdings = 0

        if count == count_end:
            bought_today = False  # Reset the flag for the next day
        
    profit_or_loss = capital - initial_capital
    return accumulated_return

# whole_tickers = ["SOXS", "SOXL"]
whole_tickers = ["SOXL", "SOXS"]
# whole_tickers = ["LABU", "LABD"]
# whole_tickers = ["LABD", "LABU"]
# whole_tickers = ["SOXL"]

interval = "1h"
start_month = "2023-12"
end_month = "2023-12"
start_month_pd = pd.to_datetime(start_month)
end_month_pd = pd.to_datetime(end_month)
sim_result = {}

current = start_month_pd

while current <= end_month_pd:

    current_start_date = current - MonthBegin(1)  # 현재 달의 시작일
    current_end_date = current + MonthEnd(0)  # 현재 달의 말일

    current_start_date = "2023-12-05"
    current_end_date = "2024-01-01"
    interval = "5m"

    initial_capital = 10000
    margin = 0.010
    margin2 = 0.05
    stop_loss = 0.015
    stop_loss2 = 0.5
    commission_rate = 0.001
    count_hold = 12
    count_end = (count_hold * 6.5) - 1
    # count_end = 20

    tickers = whole_tickers
    stock_data = fetch_stock_data(tickers, interval, current_start_date, current_end_date)
    print(stock_data["SOXL"].loc["2023-12-11 15:55:00", "Close"])
    # print("STOCK")
    # print(stock_data)
    accumulated_return = backtest_strategy(tickers, stock_data, initial_capital, margin, stop_loss, commission_rate, count_hold, count_end)

    sim_result[current] = accumulated_return
    # print(current, accumulated_return)

    current += pd.DateOffset(months=1)

total_return = 1
for date, result in sim_result.items():
    total_return = total_return * (1+(result/100))
    print(date, result, total_return)
