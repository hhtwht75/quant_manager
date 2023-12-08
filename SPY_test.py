import yfinance as yf
import matplotlib.pyplot as plt

# Define the ticker and time period
ticker = "SPY"
start = "1993-01-01"
end = "2018-12-31"

# Download data
data = yf.download(ticker, start=start, end=end)

# Calculate daily and cumulative returns
data['Buy_Price'] = data['Close'].shift(-1)
data['Sell_Price'] = data['Open']
data['Daily_Return'] = (data['Sell_Price'] - data['Buy_Price']) / data['Buy_Price']
data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1

# Plot
plt.figure(figsize=(12,6))
plt.plot(data['Cumulative_Return'])
plt.title('Cumulative Return of SPY (Buying at Close, Selling at Next Day Open)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.show()
