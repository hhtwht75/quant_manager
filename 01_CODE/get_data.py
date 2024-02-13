import pandas as pd
from alpha_vantage_data import *
from stock_data_alpha import *
    
make_csv(["SOXL","SOXS"],"2020-01","2020-12","./02_DATA/SOXL_2020.csv")