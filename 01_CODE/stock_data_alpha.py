import pandas as pd
from alpha_vantage_data import *

def fetch_alpha_month(input_filename,tickers, month):
    df = {}
    for ticker in tickers:
        try:
            data = filter_ticker_month_data(input_filename=input_filename, ticker=ticker, year_month=month)
            df[ticker] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return df

def fetch_alpha_year(input_filename,tickers):
    df = {}
    for ticker in tickers:
        try:
            data = filter_ticker_year_data(input_filename=input_filename, ticker=ticker)
            df[ticker] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return df

def make_csv(tickers, start_month, end_month, output_filename="combined_data.csv"):
    start = pd.to_datetime(start_month)
    end = pd.to_datetime(end_month)

    combined_data = pd.DataFrame()

    current = start
    while current <= end:
        input_month = current.strftime("%Y-%m")
        result_temp = get_alpha(tickers=tickers, month=input_month)
        
        for ticker in tickers:
            if ticker in result_temp:
                result_temp[ticker]['Ticker'] = ticker
                combined_data = pd.concat([combined_data, result_temp[ticker]])
        
        current += pd.DateOffset(months=1)
    
    combined_data.reset_index(inplace=True)
    combined_data.to_csv(output_filename, index=False)
    print(f"Data saved to {output_filename}")

def filter_ticker_month_data(input_filename, ticker, year_month, output_filename=None):
    # CSV 파일에서 데이터 읽기
    data = pd.read_csv(input_filename)
    
    # 'index' 칼럼을 날짜형으로 변환
    data['index'] = pd.to_datetime(data['index'])
    
    # 원하는 티커와 특정 월의 데이터만 필터링하고 복사본 생성
    filtered_data = data[(data['Ticker'] == ticker) & (data['index'].dt.year == int(year_month.split('-')[0])) & (data['index'].dt.month == int(year_month.split('-')[1]))].copy()

    # 이제 'index' 칼럼을 날짜형으로 변환하는 동작은 복사본에 직접 적용되므로 경고가 발생하지 않음
    filtered_data['index'] = pd.to_datetime(filtered_data['index'])
    filtered_data.set_index('index', inplace=True)


    if not filtered_data.empty:
        if output_filename:
            # 필터링된 데이터를 새로운 CSV 파일로 저장
            filtered_data.to_csv(output_filename, index=False)
            print(f"Filtered data for {ticker} in {year_month} saved to {output_filename}")
        else:
            # 파일로 저장하지 않고 필터링된 데이터 반환
            return filtered_data
    else:
        print(f"No data found for ticker {ticker} in {year_month}")
        return None
    
def filter_ticker_year_data(input_filename, ticker, output_filename=None):
    # CSV 파일에서 데이터 읽기
    data = pd.read_csv(input_filename)
    
    # 'index' 칼럼을 날짜형으로 변환
    data['index'] = pd.to_datetime(data['index'])
    
    # 원하는 티커와 특정 월의 데이터만 필터링하고 복사본 생성
    filtered_data = data[(data['Ticker'] == ticker)].copy()


    # 이제 'index' 칼럼을 날짜형으로 변환하는 동작은 복사본에 직접 적용되므로 경고가 발생하지 않음
    filtered_data['index'] = pd.to_datetime(filtered_data['index'])
    filtered_data.set_index('index', inplace=True)


    if not filtered_data.empty:
        if output_filename:
            # 필터링된 데이터를 새로운 CSV 파일로 저장
            filtered_data.to_csv(output_filename, index=False)
            # print(f"Filtered data for {ticker} in {year_month} saved to {output_filename}")
        else:
            # 파일로 저장하지 않고 필터링된 데이터 반환
            return filtered_data
    else:
        # print(f"No data found for ticker {ticker} in {year_month}")
        return None
    