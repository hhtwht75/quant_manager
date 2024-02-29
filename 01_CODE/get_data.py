import pandas as pd
from alpha_vantage_data import *
from stock_data_alpha import *
import time
    
# make_csv(["SOXL","SOXS"],"2014-01","2014-12","./02_DATA/SOXL_2014.csv")
# make_csv(["LABU","LABD"],"2017-01","2017-12","./02_DATA/LABU_2017.csv")
# make_csv(["TQQQ", "SQQQ"],"2010-01","2010-12","./02_DATA/TQQQ_2010.csv")

# tickers = ["TSLL", "TSLS"]
ticker_dict = {
    ('UDOW', 'SDOW'): 2010,
    ('UMDD', 'SMDD'): 2010,
    ('URTY', 'SRTY'): 2010,
    ('UPRO', 'SPXU'): 2006
}

for tickers, start_year in ticker_dict.items():
    for year in range(start_year,2025,1):
        try:
            time.sleep(60)  
            make_csv(tickers,f"{year}-01",f"{year}-12",f"./02_DATA/proshares_3x/{tickers[0]}/{tickers[0]}_{year}.csv")
        except BaseException as e:
            print("Error:", e)
        

    # make_csv(["SOXL","SOXS"],"2024-02","2024-02","./02_DATA/temp.csv")
