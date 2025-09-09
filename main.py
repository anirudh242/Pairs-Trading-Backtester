import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import sys

from nifty50_tickers import NIFTY50_TICKERS

START_DATE = '2020-01-01'
END_DATE = '2025-01-01'
CORRELATION_THRESHOLD = 0.95 # threshold for fast filter

# DOWNLOADING DATA
print(f"Downloading data for {len(NIFTY50_TICKERS)} stocks...")
try:
    data = yf.download(NIFTY50_TICKERS, start=START_DATE, end=END_DATE, progress=True)
except Exception as e:
    print(f"Failed to download data: {e}")
    sys.exit()

if data.empty:
    print("No data downloaded.")
    sys.exit()

prices = data['Close'].copy().dropna(axis=1)
print("\nData download and cleaning complete.")


# faster filtering by taking correlation first then searching if it meets correlation thresholder 
# this avoids running the ADF test on pairs which are obviously unrelated
print(f"\nStep 1: Finding pairs with correlation > {CORRELATION_THRESHOLD}...")
corr_matrix = prices.corr()
# getting upper triangle of the correlation matrix
# we do this step as the diagonal will always be 1 (the correlation with itself is 1) 
# and the matrix is symmetric (STOCK1, STOCK2) = (STOCK2, STOCK1) so we can remove the lower triangle
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# find index pairs with correlation above the threshold
highly_correlated_pairs = [column for column in upper_tri.columns if any(upper_tri[column] > CORRELATION_THRESHOLD)]
candidate_pairs = []
for col in highly_correlated_pairs:
    for row in upper_tri.index:
        if upper_tri.loc[row, col] > CORRELATION_THRESHOLD:
            candidate_pairs.append((row, col))
            
print(f"Found {len(candidate_pairs)} highly correlated candidate pairs.")

def find_cointegrated_pairs(candidate_pairs, dataframe):
    pairs = []
    
    for stock1_name, stock2_name in candidate_pairs:
        stock1_prices = dataframe[stock1_name]
        stock2_prices = dataframe[stock2_name]
        
        spread = stock1_prices / stock2_prices
        result = adfuller(spread)
        pvalue = result[1]
        
        if pvalue < 0.05:
            pairs.append((stock1_name, stock2_name, pvalue))
            
    return pairs

print("\nStep 2: Running cointegration test on candidate pairs...")
cointegrated_pairs = find_cointegrated_pairs(candidate_pairs, prices)


if not cointegrated_pairs:
    print("No statistically significant cointegrated pairs found.")
else:
    sorted_pairs = sorted(cointegrated_pairs, key=lambda x: x[2])
    
    print("\n--- Top Cointegrated Pairs Found ---")
    for pair in sorted_pairs:
        print(f"  - Pair: {pair[0]} & {pair[1]}, P-value: {pair[2]:.4f}")

    best_pair = sorted_pairs[0]
    stock1_name = best_pair[0]
    stock2_name = best_pair[1]
    
    print(f"\nVisualizing the spread of the best pair: {stock1_name} & {stock2_name}")
    best_pair_spread = prices[stock1_name] / prices[stock2_name]
    best_pair_spread.plot(figsize=(12, 6))
    plt.title(f'Price Spread (Ratio): {stock1_name} / {stock2_name}')
    plt.ylabel('Spread Ratio')
    plt.axhline(best_pair_spread.mean(), color='red', linestyle='--', label='Mean Spread')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Visualizing the normalized prices for {stock1_name} & {stock2_name}")
    best_pair_prices = prices[[stock1_name, stock2_name]]
    normalized_best_pair = (best_pair_prices / best_pair_prices.iloc[0] * 100)
    normalized_best_pair.plot(figsize=(12, 6))
    plt.title(f'Normalized Price History: {stock1_name} vs. {stock2_name}')
    plt.ylabel('Normalized Price (starts at 100)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.show()

    # BOLLINGER BANDS
    moving_average = best_pair_spread.rolling(window=20).mean()
    moving_std_dev = best_pair_spread.rolling(window=20).std()
    upper_band = moving_average + 2 * moving_std_dev
    lower_band = moving_average - 2 * moving_std_dev

    plt.figure(figsize=(12,6))
    best_pair_spread.plot(label='Spread', color='blue')
    moving_average.plot(label='Moving Average', color='black', linestyle='--')
    upper_band.plot(label='Upper Band', color='red', linestyle=':')
    lower_band.plot(label='Lower Band', color='green', linestyle=':')
    plt.title(f'Bollinger Bands on {stock1_name} / {stock2_name} Spread')
    plt.ylabel('Spread Ratio')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.show() 

    print("\nStep 4: Running the vectorized backtest...")

    backtest_df = pd.DataFrame({
        'spread': best_pair_spread,
        'moving_average': moving_average,
        'upper_band': upper_band,
        'lower_band': lower_band
    }).dropna()

    backtest_df['position'] = 0
    in_position = 0

    for index, row in backtest_df.iterrows():
        if in_position == 1 and row['spread'] >= row['moving_average']:
            in_position = 0
        elif in_position == -1 and row['spread'] <= row['moving_average']:
            in_position = 0
        
        if in_position == 0:
            if row['spread'] > row['upper_band']:
                in_position = -1
            elif row['spread'] < row['lower_band']:
                in_position = 1
        
        backtest_df.loc[index, 'position'] = in_position

    
    # CALCULATING RETURNS
    backtest_df['spread_return'] = backtest_df['spread'].pct_change()
    backtest_df['strategy_return'] = backtest_df['spread_return'] * backtest_df['position'].shift(1)
    
    # EQUITY CURVE
    backtest_df['equity_curve'] = (1 + backtest_df['strategy_return']).cumprod()

    # PLOT THE EQUITY CURVE
    plt.figure(figsize=(12, 6))
    backtest_df['equity_curve'].plot(label='Pairs Trading Strategy')
    plt.title(f'Equity Curve for {stock1_name} & {stock2_name}')
    plt.ylabel('Growth of ₹1 Investment')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.show()

    # CALCULATE FINAL METRICS
    annualized_return = backtest_df['equity_curve'].iloc[-1]**(252/len(backtest_df)) - 1
    annualized_volatility = backtest_df['strategy_return'].std() * np.sqrt(252)
    if annualized_volatility == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = annualized_return / annualized_volatility

    print("\n--- Backtest Performance Metrics ---")
    print(f"Total Return: {(backtest_df['equity_curve'].iloc[-1] - 1) * 100:.2f}%")
    print(f"Annualized Return: {annualized_return * 100:.2f}%")
    print(f"Annualized Volatility: {annualized_volatility * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")