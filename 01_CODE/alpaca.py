import requests
import pandas as pd
import exchange_calendars as ecals
import datetime, time
import pytz


def get_trading_hour(date, calendar):

    trading_hour = {}
    
    if date in calendar.schedule.index:
        
        schedule = calendar.schedule.loc[date]

        open_time = schedule.open
        close_time = schedule.close
        close_time_ny = close_time.tz_convert('America/New_York')
        
        trading_hour['Open'] = open_time.tz_convert(pytz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        trading_hour['Close'] = close_time.tz_convert(pytz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        trading_hour['Check'] = close_time_ny.time() == datetime.time(16, 0, 0)
    
    else:

        trading_hour['Check'] = False

    return trading_hour

def get_daily_stock_data(symbols, start, end, timeframe="1Min", limit=10000):

    url = "https://data.alpaca.markets/v2/stocks/bars"

    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": "PKL8JM1FQX6TEVEOJOMT",
        "APCA-API-SECRET-KEY": "DCrnOi6u5fDQwWtVbu5LnXRvOweq7a6hRLFUVbkp"
    }

    params = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "limit": limit,
    }

    r =  requests.get(url, headers=headers, params=params)
    data = r.json()

    all_dfs = []
    for symbol in symbols:
        if symbol in data['bars']:
            df = pd.DataFrame(data['bars'][symbol])
            df = df.rename(columns={'t': 'index', 'o': 'Open', 'c': 'Close', 'h': 'High', 'l': 'Low'})
            df['index'] = pd.to_datetime(df['index'])
            df['index'] = df['index'].dt.tz_convert('America/New_York')
            df.set_index('index', inplace=True)
            df['symbol'] = symbol

            all_dfs.append(df)

    if all_dfs:
        combined_df = pd.concat(all_dfs)
    else:
        combined_df = pd.DataFrame()

    return combined_df

def get_stock_data(symbols, year):

    nyse_calendar = ecals.get_calendar("XNYS")
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    all_data = []
        
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    while start <= end:
        try:
            trading_hour = get_trading_hour(start, nyse_calendar)

            if trading_hour['Check']:

                daily_data = get_daily_stock_data(symbols, trading_hour['Open'], trading_hour['Close'])
                all_data.append(daily_data)

            start += datetime.timedelta(days=1)
            time.sleep(0.5)
        except TimeoutError as te:
            print(f"Timeout Error: {te}")
            time.sleep(60)
        except ValueError as ve:
            # print(f"Value Error: {ve} at {start}")
            time.sleep(0.5)
            start += datetime.timedelta(days=1)
            continue

    final_df = pd.concat(all_data)
    final_df.to_csv(f"./02_DATA/direxion_3x/{symbols[0]}/{symbols[0]}_{year}.csv")
    print(f"{symbols[0]}_{year}.csv is saved")

# whole_tickers = [("TMF","TMV"),("TNA","TZA"),("YINN","YANG")]
# for tickers in whole_tickers:
#     for year in range(2019,2024):
#         get_stock_data(tickers,year)

for year in range(2016,2024):
    get_stock_data(("QQQ","PSQ"),year)


