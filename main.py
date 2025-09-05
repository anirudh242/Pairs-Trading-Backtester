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


# --- 4. Test Cointegration Only on Candidate Pairs (Precise Test) ---
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