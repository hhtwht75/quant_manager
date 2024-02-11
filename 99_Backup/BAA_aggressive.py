from stock_data import *

tickers = ['QQQ', 'EEM', 'EFA', 'AGG', 'TIP', 'PDBC', 'BIL', 'IEF', 'TLT', 'LQD', 'BND', 'SPY', 'VEA', 'VWO']
sdpath = './stock_data.csv'
data = stock_data_download(tickers, sdpath)

def select_assets(momentum_scores, price_ratios):
    warning_assets = ['SPY', 'VEA', 'VWO', 'BND']
    safe_assets = ['TIP', 'PDBC', 'BIL', 'IEF', 'TLT', 'LQD', 'BND']
    aggressive_assets = ['QQQ', 'EEM', 'EFA', 'AGG']

    # 경고자산 중 모멘텀 스코어가 0보다 낮은 자산이 있는지 확인
    if any(momentum_scores[warning_assets].iloc[-1] < 0):
        # 안전자산 중 가격비율이 가장 높은 3개의 자산을 선택
        selected_assets = price_ratios[safe_assets].iloc[-1].nlargest(3).index.tolist()
        
        # 가격비율이 1보다 낮은 자산이 있다면 BIL로 변경
        selected_assets = ['BIL' if price_ratios[asset].iloc[-1] < 1 else asset for asset in selected_assets]
    else:
        # 공격자산 중에서 모멘텀 스코어가 가장 높은 자산을 선택
        selected_assets = [momentum_scores[aggressive_assets].iloc[-1].idxmax()]

    return selected_assets

# 이동 평균을 계산
moving_average = calculate_moving_average(data)

# 모멘텀 스코어와 가격비율 계산
momentum_scores = calculate_momentum_score(data)
price_ratios = calculate_price_ratio(data, moving_average)

# 자산 선택
selected_assets = select_assets(momentum_scores, price_ratios)

#print(moving_average)
print(selected_assets)