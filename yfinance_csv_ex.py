import yfinance as yf
import pandas as pd

# yfinance와 pandas 설치
# pip install yfinance pandas openpyxl

# TSLA의 15분 봉 데이터를 가져옵니다.
# interval 옵션에 "15m"을 지정하면 15분 봉 데이터를 가져올 수 있습니다.
tsla_15min = yf.download('TSLA', interval='15m', period='60d')
tsls_15min = yf.download('TSLS', interval='15m', period='60d')


# 가져온 데이터를 Excel 파일로 저장합니다.
# 저장할 때, Excel 작성 엔진으로 'openpyxl'을 사용해야 합니다.
tsla_15min.to_excel('tsla_15min_data.xlsx', engine='openpyxl')
tsls_15min.to_excel('tsls_15min_data.xlsx', engine='openpyxl')

print("Data has been successfully saved")