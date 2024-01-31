import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers, interval="15m", start_date=None, end_date=None):
    df = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, interval=interval, start=start_date, end=end_date)
            df[ticker] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return df

def backtest_strategy(tickers, stock_data, initial_capital=100000, margin = 0.01, stop_loss=0.015, commission_rate=0.001, count_hold=1, close_count=25):
    capital = initial_capital
    previous_capital = initial_capital  # Capital after the previous sell
    holdings = 0  # Number of stocks held
    opening_time = stock_data[tickers[0]].index[0].time()
    # closing_time = stock_data[tickers[0]].index[-1].time()
    closing_time = stock_data[tickers[0]].index[close_count].time()
    # print("Closing: ", stock_data[tickers[0]].index[-1].time())
    bought_today = False  # Flag to check if a stock was bought on the current date

    date_save = None

    for date in stock_data[tickers[0]].index:

        if date_save == date.date():
            count += 1
        else:
            count = 0
            date_save = date.date()

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
            #     # print(f"Sold {bought_ticker} at ${sell_price:.2f} at Margin GET on {date}")
            #     # print(f"Change rate since last sale: {change_rate:.2f}%")
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
                    print(f"Sold {bought_ticker} at ${sell_price:.2f} due to 1.5% STOP LOSS rule on {date}")
                    print(f"Change rate since last sale: {change_rate:.2f}%")
                    # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                    # print(f"     ")
                    previous_capital = capital
                    holdings = 0
                elif stock_data[bought_ticker].loc[date, "Low"] <= buy_price * (1-stop_loss2):
                    sell_price = buy_price * (1-stop_loss2)
                    capital += (holdings * sell_price) * (1 - commission_rate)
                    accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                    change_rate = ((capital - previous_capital) / previous_capital) * 100
                    print(f"Sold {bought_ticker} at ${sell_price:.2f} due to 5% STOP LOSS rule on {date}")
                    print(f"Change rate since last sale: {change_rate:.2f}%")
                    # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                    # print(f"     ")
                    # print("3% STOP LOSSSSSSSSSSSS")
                    previous_capital = capital
                    holdings = 0

            elif date.time() == closing_time:
            # elif count == 7:
                sell_price = stock_data[bought_ticker].loc[date, "Close"]
                capital += (holdings * sell_price) * (1 - commission_rate)
                accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                change_rate = ((capital - previous_capital) / previous_capital) * 100
                print(f"Sold {bought_ticker} at ${sell_price:.2f} at END OF DAY on {date}")
                print(f"Change rate since last sale: {change_rate:.2f}%")
                # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                # print(f"     ")
                previous_capital = capital
                holdings = 0

        if count == 25:
            bought_today = False  # Reset the flag for the next day
        
    profit_or_loss = capital - initial_capital
    return accumulated_return

# whole_tickers = ["TQQQ", "SQQQ"]
# whole_tickers = ["LABD", "LABU"]
whole_tickers = ["SOXL", "SOXS"]
# whole_tickers = ["LABU", "LABD"]
# whole_tickers = ["LABU", "LABU", "LABD", "LABD"]


for i in range(0, len(whole_tickers) // 2):
# for i in range(0,2):

    interval = "15m"
    # period = "20d"
    start_date = "2023-12-01"
    end_date = "2024-01-23"
    initial_capital = 10000
    # margin = 0.01
    margin = 0.01
    margin2 = 0.05
    # stop_loss = 0.015
    stop_loss = 0.015
    stop_loss2 = 0.05
    commission_rate = 0.001
    count_hold = 4
    close_count= 22


    # tickers = whole_tickers
    tickers = whole_tickers[i*2 : i*2 + 2]
    stock_data = fetch_stock_data(tickers, interval, start_date, end_date)

    accumulated_return = backtest_strategy(tickers, stock_data, initial_capital, margin, stop_loss, commission_rate, count_hold, close_count)

    print(tickers)
    print(f"Accumulated Return at {close_count}: {accumulated_return:.2f}%")
    print(f"      ")
