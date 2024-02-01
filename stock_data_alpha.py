import pandas as pd
from alpha_vantage_data import *

# def make_csv(tickers, start_month, end_month):

#     start = pd.to_datetime(start_month)
#     end = pd.to_datetime(end_month)

#     results = {}

#     current = start
#     while current <= end:
#         input_month = current.strftime("%Y-%m")
#         result_temp = fetch_alpha(tickers=tickers, month=input_month)
#         ### RESULT_TEMP를 CSV에 누적하는 코드 ###
#         current += pd.DateOffset(months=1)

#     return results

def make_csv(tickers, start_month, end_month, output_filename="combined_data.csv"):
    start = pd.to_datetime(start_month)
    end = pd.to_datetime(end_month)

    combined_data = pd.DataFrame()

    current = start
    while current <= end:
        input_month = current.strftime("%Y-%m")
        result_temp = fetch_alpha(tickers=tickers, month=input_month)
        
        for ticker in tickers:
            if ticker in result_temp:
                result_temp[ticker]['Ticker'] = ticker
                combined_data = pd.concat([combined_data, result_temp[ticker]])
        
        current += pd.DateOffset(months=1)
    
    combined_data.reset_index(inplace=True)
    combined_data.to_csv(output_filename, index=False)
    print(f"Data saved to {output_filename}")

tickers = ["LABU", "LABD"]
make_csv(tickers=tickers, start_month="2023-12", end_month="2024-01", output_filename="combined_data.csv")