import exchange_calendars as ecals
from pytz import timezone

import datetime

t_now = datetime.datetime.now(timezone('America/New_York')) 
nyse_calendar = ecals.get_calendar("XNYS")
today = t_now.date()

# 오늘 날짜에 대한 거래 일정을 조회합니다.
schedule = nyse_calendar.schedule(start_date=today, end_date=today)

# 오늘의 폐장 시간을 가져옵니다.
close_time = schedule.loc[today.strftime('%Y-%m-%d'), 'market_close']

# 폐장 시간을 뉴욕 시간대로 변환합니다.
close_time_ny = close_time.tz_localize('UTC').tz_convert('America/New_York')

print(close_time_ny)