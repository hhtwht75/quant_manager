# API: UKEEHPB4Z00E8C7Y
import requests

def alpha_request(ticker='SOXL', month='2020-01'):

    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=UKEEHPB4Z00E8C7Y'
    params = {
        'function' : 'TIME_SERIES_INTRADAY',
        'symbol': ticker,
        'interval': month,
        'extended_hours' : False,
        'month' : '2020-01',
        'outputsize': 'full',
    }

    r = requests.get(url, params=params)
    data = r.json()
    return data