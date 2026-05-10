import requests

url = 'https://www.alphavantage.co/query'
params = {
    'function' : 'TIME_SERIES_INTRADAY',
    'symbol': 'SOXS',
    'interval': '1min', # 1min, 5min, 15min, 30min, 60min
    'extended_hours' : 'false', 
    'month' : '2024-02',
    'outputsize': 'full',
    'apikey': '69Y1EGO0TWH4NCSZ'
}

r = requests.get(url, params=params)
data = r.json()

print(data)
