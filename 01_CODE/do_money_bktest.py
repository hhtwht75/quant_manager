import yfinance as yf
import pandas as pd
from alpha_vantage_data import *
from stock_data_alpha import *
import time, os

def backtest_strategy(tickers, stock_data, initial_capital=100000, margin = 0.01, stop_loss=0.015, commission_rate=0.001, holding=datetime.time(10, 30, 00), closing=datetime.time(15, 25, 00)):
    capital = initial_capital
    previous_capital = initial_capital  # Capital after the previous sell
    holding_time=holding
    closing_time=closing    
    bought_today = False  # Flag to check if a stock was bought on the current date
    date_save = None

    prev1_open = {}
    prev1_close = {}
    prev2_open = {}
    prev2_close = {}
    update = False

    for idx in stock_data[tickers[0]].index:
        try:
            if date_save == idx.date():
                count += 1
            else:
                count = 0
                date_save = idx.date()
                # change_rate = ((capital - previous_capital) / previous_capital) * 100
                # print(f"Change rate since last sale: {change_rate:.2f}%")
                previous_capital = capital
            
            # Opening Price
            if count == 0:

                opening_price = {}
                bought_ticker = {}
                bought_today = False
                update = False

                for ticker in tickers:   
                    # opening_price.append(stock_data[ticker].loc[idx, "Open"])
                    opening_price[ticker] = stock_data[ticker].loc[idx, "Open"]
                    # bought_ticker[ticker] = {}

            
                    
            if idx.time() > datetime.time(15, 58, 00) and update == False:
                bought_today = False  # Reset the flag for the next day
                for ticker in tickers:
                    if prev1_open:
                        prev2_open[ticker] = prev1_open[ticker]
                    if prev1_close:
                        prev2_close[ticker] = prev1_close[ticker]
                    prev1_open[ticker] = opening_price[ticker]
                    prev1_close[ticker] = stock_data[ticker].loc[idx, "Close"]
                update = True

            else:

                for ticker in tickers:

                    if bought_today == False:
                            
                        if not (prev1_open and prev1_close):

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

                        else:
                            
                            ### Condition
                            # if prev1_close[ticker] < opening_price[ticker]:
                            # if prev1_open[ticker] > prev1_close[ticker]:
                            if prev1_open[ticker] > prev1_close[ticker] and prev1_close[ticker] < opening_price[ticker]:
                            # if prev1_open[ticker] > prev1_close[ticker] or prev1_close[ticker] < opening_price[ticker]:
                            # if True:
                                
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

                    # elif ticker == "SOXL" and idx.time() < holding_time and not (ticker in bought_ticker):

                    #     if stock_data[ticker].loc[idx, "High"] >= (1 + margin) * opening_price[ticker]:
                    #         buy_price = max((1 + margin) * opening_price[ticker], stock_data[ticker].loc[idx, "Open"])
                    #         num_stocks = previous_capital // (buy_price * (1 + commission_rate))
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
                                
                                if idx.time() > holding_time:
                                    sell_price = min(bought_ticker[ticker]["buy_price"] * (1 - stop_loss), stock_data[ticker].loc[idx, "Open"])
                                    capital += (bought_ticker[ticker]["num_stocks"] * sell_price) * (1 - commission_rate)
                                    accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                                    change_rate = ((capital - previous_capital) / previous_capital) * 100
                                    # print(f"Sold {ticker} at ${sell_price:.2f} due to 1.5% STOP LOSS rule on {idx}")
                                    # print(f"Change rate since last sale: {change_rate:.2f}%")
                                    # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                                    # print(f"     ")
                                    # previous_capital = capital
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
                                        # previous_capital = capital
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
                            # previous_capital = capital
                            bought_ticker[ticker]["num_stocks"] = 0
        except KeyError:
            # print(f"KeyError for timestamp: {idx}. Skipping...")
            continue
        
        
    # profit_or_loss = capital - initial_capital
    return accumulated_return


# whole_tickers = ["TQQQ", "SQQQ", "TMV", "TMf", "TYO", "TYD", "YANG", "YINN", "EDZ", "EDC", "TZA", "TNA", "WEBL", "WEBS", "FAZ", "FAS", "DRV", "DRN", "HIBS", "HIBL", "LABD", "LABU", "SOXL", "SOXS", "TECL", "TECS"]

# whole_tickers = ["TQQQ", "SQQQ"]
# whole_tickers = ["LABD", "LABU"]
# whole_tickers = ["LABU", "LABD"]
# whole_tickers = ["SOXL","SOXS"]
# whole_tickers = ["SOXS","SOXL"]

# tickers = ["SOXL", "SOXS"]
tickers = ["LABU", "LABD"]
# tickers = ["TQQQ", "SQQQ"]

years = []
# for i in range (2020,2024,1):
    # years.append(f"{i}")
years = ["2020", "2021", "2023"]
# years = ["2023"]

        
for year in years:
    start_month = f"{year}-01"
    end_month = f"{year}-12"
    start_month_pd = pd.to_datetime(start_month)
    end_month_pd = pd.to_datetime(end_month)
    sim_result = {}

    current = start_month_pd
    while current <= end_month_pd:

        input_month = current.strftime("%Y-%m")

        month = input_month
        initial_capital = 10000
        margin = 0.01
        margin2 = 0.05
        stop_loss = 0.035
        stop_loss2 = 0.05
        commission_rate = 0.001
        # commission_rate = 0.001*rate/5
        holding_time = datetime.time(10, 30, 00)
        closing_time = datetime.time(15, 00, 00)


        
        stock_data = fetch_alpha(f"./02_DATA/{tickers[0]}_{year}.csv",tickers, month)
        accumulated_return = backtest_strategy(tickers, stock_data, initial_capital, margin, stop_loss, commission_rate, holding_time, closing_time)

        sim_result[input_month] = accumulated_return
        # print(input_month, accumulated_return)

        current += pd.DateOffset(months=1)

    total_return = 1

    for date, result in sim_result.items():
        total_return = total_return * (1+(result/100))
        # print(date, result, total_return)

    print(f"Total Return of {year}: ", total_return)
