import yfinance as yf
import matplotlib.pyplot as plt
import sys 

TICKERS = ['INFY.NS', 'WIPRO.NS']
START_DATE = '2020-01-01'
END_DATE = '2025-01-01'

try:
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE)
    print("Data downloaded successfully.")
except Exception as e:
    print(f"Failed to download data: {e}")
    sys.exit() 

if data.empty:
    print("No data downloaded. Check your tickers, date range, or internet connection.")
    sys.exit() 

# dataframe of pair prices, adjusted close prices
pair_prices = data['Close'].copy()
pair_prices = pair_prices.rename(columns={'INFY.NS': 'INFY', 'WIPRO.NS': 'WIPRO'})

pair_prices = pair_prices.dropna()

print("\nCombined DataFrame:")
print(pair_prices.head())

# PLOTTING
# normalized prices to date graph
normalized_prices = pair_prices / pair_prices.iloc[0] * 100
normalized_prices.plot(figsize=(12, 6))

plt.title('Normalized Price History: INFOSYS vs WIPRO')
plt.ylabel('Normalized Price (Starting from 100)')
plt.xlabel('Date')
plt.grid(True)
plt.show()

# spread to date graph
pair_prices['spread'] = pair_prices['INFY'] / pair_prices ['WIPRO']
pair_prices['spread'].plot(figsize=(12, 6))

plt.title('Price Spread: INFOSYS vs WIPRO')
plt.ylabel('Spread Ratio')
plt.xlabel('Date')
plt.grid(True)
plt.legend()
plt.axhline(pair_prices['spread'].mean(), color='red', linestyle='--', label='Mean Spread')
plt.show()
