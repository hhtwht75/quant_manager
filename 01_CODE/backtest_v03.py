import pandas as pd
import exchange_calendars as ecals
import datetime, time
from fetch import fetch_stock_data

def backtest_strategy(tickers, stock_data, initial_capital=10000000000000, margin = 0.01, stop_loss=0.035, stop_loss2=0.05, commission_rate=0.001, holding=datetime.time(10, 30, 00), closing=datetime.time(15, 25, 00)):

    capital = initial_capital
    previous_capital = initial_capital  # Capital after the previous sell
    holding_time=holding
    closing_time=closing    
    bought_today = False  # Flag to check if a stock was bought on the current date
    sold_today  = False
    date_save = None
    change_rate = 0
    accumulated_return = 0
    target_cnt = []

    prev1_open = {}
    prev1_close = {}
    prev2_open = {}
    prev2_close = {}
    update = False
    bought_ticker = {}
    candidate = []

    nyse_calendar = ecals.get_calendar("XNYS")

    for idx in stock_data[tickers[0]].index:
        try:
            
            today = idx.date()
                            
            if date_save == idx.date():
                count += 1
            else:
                count = 0
                date_save = idx.date()
                change_rate = ((capital - previous_capital) / previous_capital) * 100
                # print(f"Change rate since last sale: {change_rate:.2f}%")
                previous_capital = capital
            
            # Opening Price
            if count == 0:

                opening_price = {}
                bought_today = False
                sold_today = False
                update = False
                candidate = []

                if bought_ticker:
                    for ticker in bought_ticker:
                        if bought_ticker[ticker]:
                            if bought_ticker[ticker]["num_stocks"] != 0:
                                sell_price = opening_price[ticker]
                                capital += (bought_ticker[ticker]["num_stocks"] * sell_price) * (1 - commission_rate)
                                print(f"Sold {ticker} at ${sell_price:.2f} at OPENING OF DAY on {idx}")
                                bought_ticker[ticker]["num_stocks"] = 0

                for ticker in tickers:   
                    opening_price[ticker] = stock_data[ticker].loc[idx, "Open"]
                    bought_ticker[ticker] = {}

                bought_ticker = {}
                target_cnt = []

                for ticker in tickers:   
                    if prev1_open and prev1_close:
                        if prev1_open[ticker] > prev1_close[ticker] and 1.01*prev1_close[ticker] < opening_price[ticker]:
                        # if prev1_close[ticker] < opening_price[ticker]:
                            candidate.append(ticker)
                # print(f"     ")
                    
            elif idx.time() > datetime.time(15, 57, 00) and update == False:
                bought_today = False  # Reset the flag for the next day
                for ticker in tickers:
                    if prev1_open:
                        prev2_open[ticker] = prev1_open[ticker]
                    if prev1_close:
                        prev2_close[ticker] = prev1_close[ticker]
                for ticker in tickers:
                    prev1_open[ticker] = opening_price[ticker]
                    prev1_close[ticker] = stock_data[ticker].loc[idx, "Close"]

                update = True
                # print(f"Change rate since last sale: {change_rate:.2f}%")
                # print(f"Accumulated return after sale: {accumulated_return:.2f}%")

            else:

                for ticker in tickers:

                    # if bought_today == False and idx.time() < closing_time:
                    if idx.time() < closing_time:
                            
                        if not (prev1_open and prev1_close):

                            continue

                        else:
                            if candidate:
                                if ticker in candidate and not bought_today:
                                    # if stock_data[ticker].loc[idx, "High"] >= opening_price[ticker]:
                                    if stock_data[ticker].loc[idx, "High"] >= (1 + margin) * opening_price[ticker]:
                                        buy_price = opening_price[ticker]
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
                                if not sold_today:
                                    if not target_cnt:
                                        if stock_data[ticker].loc[idx, "High"] >= (1 + margin) * opening_price[ticker]:
                                            target_cnt.append(ticker)

                                            buy_price = (1 + margin) * opening_price[ticker]
                                            temp_capital = capital / 2
                                            num_stocks = temp_capital // (buy_price * (1 + commission_rate))
                                            capital -= num_stocks * buy_price * (1 + commission_rate)
                                            bought_ticker[ticker] = {}
                                            bought_ticker[ticker]["num_stocks"] = num_stocks
                                            bought_ticker[ticker]["buy_price"] = buy_price
                                            bought_today = True

                                            # print(f"Bought {num_stocks} of {ticker} at ${buy_price:.2f} on {idx}")
                                            
                                            # print(f"{ticker}: Target Approach at {idx}")
                                
                                    elif ticker != target_cnt[-1]:
                                        
                                        if stock_data[ticker].loc[idx, "High"] >= (1 + margin) * opening_price[ticker] and ticker == target_cnt[0]:
                                            
                                            # print(f"{ticker}: Target Approach at {idx}")
                                            target_cnt.append(ticker)
                                            buy_price = (1 + margin) * opening_price[ticker]
                                            temp_capital = capital / 2
                                            num_stocks = temp_capital // (buy_price * (1 + commission_rate))
                                            capital -= num_stocks * buy_price * (1 + commission_rate)
                                            if not bought_ticker:
                                                bought_ticker[ticker] = {}
                                                bought_ticker[ticker]["num_stocks"] = num_stocks
                                                bought_ticker[ticker]["buy_price"] = buy_price
                                            else:
                                                bought_ticker[ticker]["num_stocks"] += num_stocks
                                            bought_today = True
                                            # print(f"Bought {num_stocks} of {ticker} at ${buy_price:.2f} on {idx}")
                                        
                                        elif stock_data[ticker].loc[idx, "High"] >= (1 + margin2) * opening_price[ticker] and ticker != target_cnt[0]:
                                            target_cnt.append(ticker)
                                            
                                                    
                                                    # break

                if bought_today:

                    for ticker in bought_ticker:

                        if stock_data[ticker].loc[idx, "Low"] <= bought_ticker[ticker]["buy_price"] * (1-stop_loss):
                            
                            if bought_ticker[ticker]["num_stocks"] != 0:
                                
                                if idx.time() > holding_time:
                                    sell_price = min(bought_ticker[ticker]["buy_price"] * (1 - stop_loss), stock_data[ticker].loc[idx, "Open"])
                                    capital += (bought_ticker[ticker]["num_stocks"] * sell_price) * (1 - commission_rate)
                                    accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                                    change_rate = ((capital - previous_capital) / previous_capital) * 100
                                    # print(f"Sold {ticker} at ${sell_price:.2f} due to {round(100*stop_loss,2)}% STOP LOSS rule on {idx}")
                                    # print(f"Change rate since last sale: {change_rate:.2f}%")
                                    # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                                    # print(f"     ")
                                    # previous_capital = capital
                                    bought_ticker[ticker]["num_stocks"] = 0
                                    sold_today = True

                                else: #Stoploss even if during hold time
                                    if stock_data[ticker].loc[idx, "Low"] <= bought_ticker[ticker]["buy_price"] * (1-stop_loss2):
                                        sell_price = bought_ticker[ticker]["buy_price"] * (1 - stop_loss2)
                                        capital += (bought_ticker[ticker]["num_stocks"] * sell_price) * (1 - commission_rate)
                                        accumulated_return = ((capital - initial_capital) / initial_capital) * 100
                                        change_rate = ((capital - previous_capital) / previous_capital) * 100
                                        # print(f"Sold {ticker} at ${sell_price:.2f} due to {round(100*stop_loss2,2)}% STOP LOSS rule on {idx}")
                                        # print(f"Change rate since last sale: {change_rate:.2f}%")
                                        # print(f"Accumulated return after sale: {accumulated_return:.2f}%")
                                        # print(f"     ")
                                        # previous_capital = capital
                                        bought_ticker[ticker]["num_stocks"] = 0
                                        sold_today = True

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
                            sold_today = True

        except KeyError as e:
            # print(f"KeyError: {KeyError}")
            continue
        
        
    # profit_or_loss = capital - initial_capital
    return accumulated_return


# tickers = ('SOXL', 'SOXS')
# tickers = ('TECL', 'TECS')

# tickers = ('LABU', 'LABD')
# tickers = ('TQQQ',)
# tickers = ('UMDD', 'SMDD')

# tickers = ('UMDD', 'SMDD')

# dir_path = f"./02_DATA/direxion_3x/{tickers[0]}"



years = []
for i in range (2019,2024,1):
    years.append(f"{i}")
# years = ["2020", "2021", "2023"]
years = ["2024"]
# whole_tickers = [("SOXL","SOXS"),("LABU","LABD"),("HIBL","HIBS"),("TECL","TECS"),("TQQQ","SQQQ"),("WEBL","WEBS"),("UMDD","SMDD"),("DRN","DRV"),("FAS","FAZ"),("SPXL","SPXS"),("TMF","TMV"),("YINN","YANG")]
# whole_tickers = [("DRN","DRV"),("FAS","FAZ"),("SPXL","SPXS"),("TMF","TMV"),("YINN","YANG")]
whole_tickers = [("SOXL","SOXS"),("LABU","LABD"),]
# whole_tickers = [("SOXS","SOXL"),("LABD","LABU"),]

for tickers in whole_tickers:
    ticker_order = tickers[0]
    dir_path = f"./02_DATA/3x/{ticker_order}"
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
            margin = 0.001
            margin2 = 0.005
            stop_loss = 0.05
            stop_loss2 = 0.05
            commission_rate = 0.000
            # commission_rate = 0.001*rate/5
            holding_time = datetime.time(10, 30, 00)
            closing_time = datetime.time(15, 55, 00)
            
            # stock_data = fetch_stock_data(f"./02_DATA/stock_data_{year}.csv",tickers, month)
            stock_data = fetch_stock_data(f"{dir_path}/{ticker_order}_{year}.csv",tickers, month)
            sim_result[input_month] = backtest_strategy(tickers, stock_data, initial_capital, margin, stop_loss, stop_loss2, commission_rate, holding_time, closing_time)
            # print(input_month, accumulated_return)

            current += pd.DateOffset(months=1)

        total_return = 1

        for date, result in sim_result.items():
            total_return = total_return * (1+(result/100))
            print(date, result, total_return)

        if total_return > 1.025:
            total_return = ((total_return*10000 - 10250)*0.78 + 10250)/10000 # TAX
        print(f"Total Return of {tickers} at {year}: ", total_return)

    print("===========================")

# for year in years:

#     initial_capital = 10000
#     margin = 0.01
#     margin2 = 0.05
#     stop_loss = 0.035
#     stop_loss2 = 0.05
#     commission_rate = 0.001
#     # commission_rate = 0.001*rate/5
#     holding_time = datetime.time(10, 30, 00)
#     closing_time = datetime.time(15, 00, 00)
    
#     stock_data = fetch_alpha_year(f"{dir_path}/{tickers[0]}_{year}.csv",tickers)
#     accumulated_return = backtest_strategy(tickers, stock_data, initial_capital, margin, stop_loss, commission_rate, holding_time, closing_time)

#     total_return = 1 + accumulated_return/100

#     if total_return > 1.025:
#         total_return = ((total_return*10000 - 10250)*0.78 + 10250)/10000 # TAX

#     print(f"Total Return of {tickers} at {year}: ", total_return)
