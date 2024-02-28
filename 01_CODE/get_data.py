import pandas as pd
from alpha_vantage_data import *
from stock_data_alpha import *
import time
    
# make_csv(["SOXL","SOXS"],"2014-01","2014-12","./02_DATA/SOXL_2014.csv")
# make_csv(["LABU","LABD"],"2017-01","2017-12","./02_DATA/LABU_2017.csv")
# make_csv(["TQQQ", "SQQQ"],"2010-01","2010-12","./02_DATA/TQQQ_2010.csv")

# tickers = ["TSLL", "TSLS"]
ticker_dict = {
    ('KORU',): 2013,
    ('SPXL', 'SPXS'): 2008,
    ('TNA','TZA'): 2008,
    ('DFEN',): 2017,
    ('WEBL','WEBS'): 2019,
    ('FAS','FAZ'): 2008,
    ('NAIL',): 2015,
    ('DRN','DRV'): 2009,
    ('DPST',): 2015,
    ('HIBL','HIBS'): 2019,
    ('TECL','TECS'): 2008,
    ('RETL.IV',): 2010
}

for tickers, start_year in ticker_dict.items():
    for year in range(start_year,2025,1):
        try:
            time.sleep(60)  
            make_csv(tickers,f"{year}-01",f"{year}-12",f"./02_DATA/direxion_3x/{tickers[0]}/{tickers[0]}_{year}.csv")
        except BaseException as e:
            print("Error:", e)
        

    # make_csv(["SOXL","SOXS"],"2024-02","2024-02","./02_DATA/temp.csv")
