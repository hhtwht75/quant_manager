import yfinance as yf
import pandas as pd
from alpha_vantage_data import *
from stock_data_alpha import *
import time

def fetch_alpha(input_filename,tickers, month):
    df = {}
    for ticker in tickers:
        try:
            data = filter_ticker_month_data(input_filename=input_filename, ticker=ticker, year_month=month)
            df[ticker] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return df

def backtest_strategy(tickers, stock_data, initial_capital=100000, margin = 0.01, stop_loss=0.015, commission_rate=0.001, count_hold=1):
    capital = initial_capital
    previous_capital = initial_capital  # Capital after the previous sell
    # opening_time = stock_data[tickers[0]].index[0].time()
    # closing_time = stock_data[tickers[0]].index[-1].time()
    # opening_time = "09:30:00"
    holding_time = datetime.time(10, 30, 00)
    closing_time = datetime.time(15, 25, 00)
    
    bought_today = False  # Flag to check if a stock was bought on the current date

    date_save = None

    for idx in stock_data[tickers[0]].index:
        try:
            if date_save == idx.date():
                count += 1
            else:
                count = 0
                date_save = idx.date()
            
            # Opening Price
            if count == 0:

                opening_price = {}
                bought_ticker = {}
                bought_today = False

                for ticker in tickers:   
                    # opening_price.append(stock_data[ticker].loc[idx, "Open"])
                    opening_price[ticker] = stock_data[ticker].loc[idx, "Open"]
                    # bought_ticker[ticker] = {}

            
                    
            if idx.time() > datetime.time(15, 30, 00):
                bought_today = False  # Reset the flag for the next day

            else:

                for ticker in tickers:

                    if bought_today == False:

                        if stock_data[ticker].loc[idx, "High"] >= (1 + margin) * opening_price[ticker]:
                            buy_price = (1 + margin) * opening_price[ticker]
                            num_stocks = capital // (buy_price * (1 + commission_rate))
                            capital -= num_stocks * buy_price * (1 + commission_rate)
                            bought_ticker[ticker] = {}
                            bought_ticker[ticker]["num_stocks"] = num_stocks
                            bought_ticker[ticker]["buy_price"] = buy_price
                            # print(bought_ticker)
                            bought_today = True
                            # print(f"Bought {num_stocks} of {ticker} at ${buy_price:.2f} on {idx}")
                            break

                    # elif idx.time() < holding_time and not (ticker in bought_ticker):

                    #     if stock_data[ticker].loc[idx, "High"] >= (1 + margin) * opening_price[ticker]:
                    #         buy_price = (1 + margin) * opening_price[ticker]
                    #         num_stocks = 2*previous_capital // (buy_price * (1 + commission_rate))
                    #         capital -= num_stocks * buy_price * (1 + commission_rate)
                    #         bought_ticker[ticker] = {}
                    #         bought_ticker[ticker]["num_stocks"] = num_stocks
                    #         bought_ticker[ticker]["buy_price"] = buy_price
                    #         bought_today = True
                    #         # print(f"Additionally Bought {num_stocks} of {ticker} at ${buy_price:.2f} on {idx}")
                    #         break

                if bought_today:

                    for ticker in bought_ticker:

                        if stock_data[ticker].loc[idx, "Low"] <= bought_ticker[ticker]["buy_price"] * (1-stop_loss):
                            
                            if bought_ticker[ticker]["num_stocks"] != 0:
                                
                                if count >= count_hold:
                                    sell_price = min(bought_ticker[ticker]["buy_price"] * (1 - stop_loss), stock_data[ticker].loc[idx, "Open"])
                                    capital += (bought_ticker[ticker]["num_stocks"] * sell_price) * (1 - commission_rate)
                                    accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                                    change_rate = ((capital - previous_capital) / previous_capital) * 100
                                    # print(f"Sold {ticker} at ${sell_price:.2f} due to 1.5% STOP LOSS rule on {idx}")
                                    # print(f"Change rate since last sale: {change_rate:.2f}%")
                                    # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                                    # print(f"     ")
                                    previous_capital = capital
                                    bought_ticker[ticker]["num_stocks"] = 0

                                else: #Stoploss even if during hold time
                                    if stock_data[ticker].loc[idx, "Low"] <= bought_ticker[ticker]["buy_price"] * (1-stop_loss2):
                                        sell_price = bought_ticker[ticker]["buy_price"] * (1 - stop_loss2)
                                        capital += (bought_ticker[ticker]["num_stocks"] * sell_price) * (1 - commission_rate)
                                        accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                                        change_rate = ((capital - previous_capital) / previous_capital) * 100
                                        # print(f"Sold {ticker} at ${sell_price:.2f} due to 5% STOP LOSS rule on {idx}")
                                        # print(f"Change rate since last sale: {change_rate:.2f}%")
                                        # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                                        # print(f"     ")
                                        previous_capital = capital
                                        bought_ticker[ticker]["num_stocks"] = 0

                        elif idx.time() > closing_time and bought_ticker[ticker]["num_stocks"] != 0:
                            sell_price = stock_data[ticker].loc[idx, "Close"]
                            capital += (bought_ticker[ticker]["num_stocks"] * sell_price) * (1 - commission_rate)
                            accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                            change_rate = ((capital - previous_capital) / previous_capital) * 100
                            # print(f"Sold {ticker} at ${sell_price:.2f} at END OF DAY on {idx}")
                            # print(f"Change rate since last sale: {change_rate:.2f}%")
                            # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                            # print(f"     ")
                            previous_capital = capital
                            bought_ticker[ticker]["num_stocks"] = 0
        except KeyError:
            # print(f"KeyError for timestamp: {idx}. Skipping...")
            continue
        
        
    # profit_or_loss = capital - initial_capital
    return accumulated_return

# whole_tickers = ["TQQQ", "SQQQ", "TMV", "TMf", "TYO", "TYD", "YANG", "YINN", "EDZ", "EDC", "TZA", "TNA", "WEBL", "WEBS", "FAZ", "FAS", "DRV", "DRN", "HIBS", "HIBL", "LABD", "LABU", "SOXL", "SOXS", "TECL", "TECS"]

whole_tickers = ["LABD", "LABU"]
# whole_tickers = ["LABU", "LABD"]
# whole_tickers = ["SOXL","SOXS"]
# whole_tickers = ["SOXS","SOXL"]



start_month = "2023-02"
end_month = "2023-12"
start_month_pd = pd.to_datetime(start_month)
end_month_pd = pd.to_datetime(end_month)
sim_result = {}

current = start_month_pd

while current <= end_month_pd:

    input_month = current.strftime("%Y-%m")

    month = input_month
    initial_capital = 10000
    margin = 0.010
    margin2 = 0.05
    stop_loss = 0.015
    stop_loss2 = 0.05
    commission_rate = 0.001
    count_hold = 12

    tickers = whole_tickers
    # stock_data = fetch_alpha("SOXLSOXS.csv",tickers, month)
    # stock_data = fetch_alpha("LABULABD.csv",tickers, month)
    stock_data = fetch_alpha("LABULABD_1min.csv",tickers, month)
    accumulated_return = backtest_strategy(tickers, stock_data, initial_capital, margin, stop_loss, commission_rate, count_hold)

    sim_result[input_month] = accumulated_return
    # print(input_month, accumulated_return)

    current += pd.DateOffset(months=1)

total_return = 1

for date, result in sim_result.items():
    total_return = total_return * (1+(result/100))
    print(date, result, total_return)
