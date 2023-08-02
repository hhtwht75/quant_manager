import yfinance as yf
import datetime
import pandas as pd
import os

def get_data(tickers, start_date, end_date):
    data = pd.DataFrame()
    while data.empty:
        data = yf.download(tickers, start=start_date, end=end_date)
        if data.empty:
            # 데이터가 비어 있으면 날짜를 하루 앞으로 옮김
            start_date -= datetime.timedelta(days=1)
            end_date -= datetime.timedelta(days=1)
    return data['Close']

def calculate_moving_average(data, window=252):
    moving_average = data.rolling(window).mean()
    return moving_average

def calculate_momentum_score(data):
    one_month_return = data.pct_change(21)  # 1개월 수익률
    three_month_return = data.pct_change(63)  # 3개월 수익률
    six_month_return = data.pct_change(126)  # 6개월 수익률
    twelve_month_return = data.pct_change(252)  # 12개월 수익률

    # 모멘텀 스코어 계산
    momentum_score = (one_month_return * 12) + (three_month_return * 4) + \
                     (six_month_return * 2) + twelve_month_return

    return momentum_score

def calculate_price_ratio(data, moving_average):
    price_ratio = data / moving_average
    return price_ratio

def stock_data_download(tickers, sdpath):
    # 파일이 이미 존재하는지 확인
    if not os.path.exists(sdpath):
        # 파일이 없으면, 1년치 데이터를 다운로드
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=400)
        data = get_data(tickers, start_date, end_date)
    else:
        # 파일이 있으면, 파일을 로드
        data = pd.read_csv(sdpath, index_col=0, parse_dates=True)

    # 오늘 날짜의 데이터를 다운로드
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=400)
    new_data = get_data(tickers, start_date, end_date)

    # 새로운 데이터의 인덱스(날짜)가 기존 데이터에 없는 경우만 추가
    new_data = new_data[~new_data.index.isin(data.index)]

    # 새로운 데이터를 기존 데이터에 추가
    data = pd.concat([data, new_data])

    # 날짜순으로 정렬
    data = data.sort_index()

    # 데이터를 CSV 파일로 다시 저장
    data.to_csv(sdpath)

    return data

# def stock_data_download(tickers):
#     # 파일이 이미 존재하는지 확인
#     if not os.path.exists('C:/Users/jhpark94/Documents/stock_data.csv'):
#         # 파일이 없으면, 1년치 데이터를 다운로드
#         end_date = datetime.datetime.now()
#         start_date = end_date - datetime.timedelta(days=365)
#         data = get_data(tickers, start_date, end_date)
#     else:
#         # 파일이 있으면, 파일을 로드
#         data = pd.read_csv('C:/Users/jhpark94/Documents/stock_data.csv', index_col=0, parse_dates=True)

#     # 오늘 날짜의 데이터를 다운로드
#     end_date = datetime.datetime.now()
#     start_date = end_date - datetime.timedelta(days=365)
#     new_data = get_data(tickers, start_date, end_date)

#     # 새로운 데이터의 인덱스(날짜)가 기존 데이터에 없는 경우만 추가
#     new_data = new_data[~new_data.index.isin(data.index)]

#     # 새로운 데이터를 기존 데이터에 추가
#     data = pd.concat([data, new_data])

#     # 날짜순으로 정렬
#     data = data.sort_index()

#     # 데이터를 CSV 파일로 다시 저장
#     data.to_csv('C:/Users/jhpark94/Documents/stock_data.csv')

#     return data