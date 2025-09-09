import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import sys
from itertools import product
import quantstats as qs

from tickers import NIFTY50_TICKERS, NIFTY100_TICKERS, NIFTY_IT_TICKERS, US_BLUECHIP_TICKERS, SP100_TICKERS

IN_SAMPLE_START = '2015-01-01'
IN_SAMPLE_END = '2019-12-31'
OUT_OF_SAMPLE_START = '2020-01-01'
OUT_OF_SAMPLE_END = '2025-01-01'
CORRELATION_THRESHOLD = 0.95 

TICKERS = SP100_TICKERS 

# DOWNLOADING DATA
print(f"Downloading data for {len(TICKERS)} stocks...")
try:
    data = yf.download(TICKERS, start=IN_SAMPLE_START, end=OUT_OF_SAMPLE_END, progress=True)
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
        
        X = sm.add_constant(stock2_prices)
        model = sm.OLS(stock1_prices, X).fit()
        hedge_ratio = model.params.iloc[1]
        spread = stock1_prices - hedge_ratio * stock2_prices
        
        result = adfuller(spread)
        pvalue = result[1]
        
        if pvalue < 0.05:
            pairs.append((stock1_name, stock2_name, pvalue, hedge_ratio))
            
    return pairs

print("\nStep 2: Running cointegration test on candidate pairs using hedge ratios...")
cointegrated_pairs = find_cointegrated_pairs(candidate_pairs, prices)

if not cointegrated_pairs:
    print("No statistically significant cointegrated pairs found.")
    sys.exit()
else:
    sorted_pairs = sorted(cointegrated_pairs, key=lambda x: x[2])
    print("\n--- Top Cointegrated Pairs Found ---")
    for pair in sorted_pairs:
        print(f"  - Pair: {pair[0]} & {pair[1]}, P-value: {pair[2]:.4f}, Hedge Ratio: {pair[3]:.2f}")

    best_pair = sorted_pairs[0]
    stock1_name = best_pair[0]
    stock2_name = best_pair[1]
    hedge_ratio = best_pair[3]
    
    in_sample_prices = prices.loc[IN_SAMPLE_START:IN_SAMPLE_END]
    in_sample_spread = in_sample_prices[stock1_name] - hedge_ratio * in_sample_prices[stock2_name]

    print("\nStep 3: Optimizing strategy parameters on IN-SAMPLE data (2015-2019)...")
    windows = range(10, 61, 10)
    std_devs = np.arange(1.5, 3.1, 0.5)

    def run_backtest_for_optimizing(spread, window, std_dev):
        moving_average = spread.rolling(window=window).mean()
        moving_std = spread.rolling(window=window).std()
        upper_band = moving_average + (std_dev * moving_std)
        lower_band = moving_average - (std_dev * moving_std)
        df = pd.DataFrame({'spread': spread, 'moving_average': moving_average, 'upper_band': upper_band, 'lower_band': lower_band}).dropna()
        df['position'] = 0
        in_position = 0
        for index, row in df.iterrows():
            if in_position == 1 and row['spread'] >= row['moving_average']: in_position = 0
            elif in_position == -1 and row['spread'] <= row['moving_average']: in_position = 0
            if in_position == 0:
                if row['spread'] > row['upper_band']: in_position = -1
                elif row['spread'] < row['lower_band']: in_position = 1
            df.loc[index, 'position'] = in_position
        df['spread_return'] = df['spread'].diff()
        df['strategy_return'] = df['spread_return'] * df['position'].shift(1)
        if df['strategy_return'].isnull().all() or df['strategy_return'].eq(0).all(): return 0.0
        daily_returns = df['strategy_return'].dropna()
        if daily_returns.empty: return 0.0
        annualized_return = daily_returns.mean() * 252
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        if annualized_volatility == 0: return 0.0
        return annualized_return / annualized_volatility

    results = []
    for window, std_dev in product(windows, std_devs):
        sharpe = run_backtest_for_optimizing(in_sample_spread, window, std_dev)
        results.append({'window': window, 'std_dev': std_dev, 'sharpe': sharpe})

    optimization_results = pd.DataFrame(results)
    best_params = optimization_results.loc[optimization_results['sharpe'].idxmax()]
    print("\n--- Optimization Complete ---")
    print("Best Parameters Found from In-Sample Data:")
    print(best_params)

    print("\nStep 4: Running final portfolio backtest on OUT-OF-SAMPLE data (2020 onwards)...")
    
    optimal_window = int(best_params['window'])
    optimal_std_dev = best_params['std_dev']
    
    full_spread = prices[stock1_name] - hedge_ratio * prices[stock2_name]
    moving_average = full_spread.rolling(window=optimal_window).mean()
    moving_std_dev = full_spread.rolling(window=optimal_window).std()
    upper_band = moving_average + optimal_std_dev * moving_std_dev
    lower_band = moving_average - optimal_std_dev * moving_std_dev

    backtest_df = pd.DataFrame({
        'spread': full_spread,
        'stock1_price': prices[stock1_name],
        'stock2_price': prices[stock2_name],
        'moving_average': moving_average,
        'upper_band': upper_band,
        'lower_band': lower_band
    }).loc[OUT_OF_SAMPLE_START:OUT_OF_SAMPLE_END].dropna()

    INITIAL_CAPITAL = 100000
    CAPITAL_PER_TRADE = 20000 
    
    portfolio = pd.DataFrame(index=backtest_df.index)
    portfolio['cash'] = INITIAL_CAPITAL
    portfolio['stock1_value'] = 0
    portfolio['stock2_value'] = 0
    portfolio['total_value'] = INITIAL_CAPITAL
    
    position = 0
    stock1_shares = 0
    stock2_shares = 0

    for i in range(1, len(backtest_df)):
        yesterday = backtest_df.index[i-1]
        today = backtest_df.index[i]
        
        portfolio.loc[today] = portfolio.loc[yesterday]
        
        portfolio.loc[today, 'stock1_value'] = stock1_shares * backtest_df.loc[today, 'stock1_price']
        portfolio.loc[today, 'stock2_value'] = stock2_shares * backtest_df.loc[today, 'stock2_price']
        portfolio.loc[today, 'total_value'] = portfolio.loc[today, 'cash'] + portfolio.loc[today, 'stock1_value'] - portfolio.loc[today, 'stock2_value']

        if position == 1 and backtest_df.loc[today, 'spread'] >= backtest_df.loc[today, 'moving_average']:
            portfolio.loc[today, 'cash'] += (stock1_shares * backtest_df.loc[today, 'stock1_price']) - (stock2_shares * backtest_df.loc[today, 'stock2_price'])
            stock1_shares, stock2_shares, position = 0, 0, 0
        elif position == -1 and backtest_df.loc[today, 'spread'] <= backtest_df.loc[today, 'moving_average']:
            portfolio.loc[today, 'cash'] -= (stock1_shares * backtest_df.loc[today, 'stock1_price']) - (stock2_shares * backtest_df.loc[today, 'stock2_price'])
            stock1_shares, stock2_shares, position = 0, 0, 0
        
        if position == 0:
            if backtest_df.loc[today, 'spread'] > backtest_df.loc[today, 'upper_band']:
                position = -1 
                dollar_investment = portfolio.loc[today, 'total_value'] / 2 
                stock1_shares = dollar_investment / backtest_df.loc[today, 'stock1_price']
                stock2_shares = (dollar_investment * hedge_ratio) / backtest_df.loc[today, 'stock2_price'] 
                portfolio.loc[today, 'cash'] += (stock1_shares * backtest_df.loc[today, 'stock1_price']) - (stock2_shares * backtest_df.loc[today, 'stock2_price'])
            elif backtest_df.loc[today, 'spread'] < backtest_df.loc[today, 'lower_band']:
                position = 1 
                dollar_investment = portfolio.loc[today, 'total_value'] / 2
                stock1_shares = dollar_investment / backtest_df.loc[today, 'stock1_price']
                stock2_shares = (dollar_investment * hedge_ratio) / backtest_df.loc[today, 'stock2_price']
                portfolio.loc[today, 'cash'] -= (stock1_shares * backtest_df.loc[today, 'stock1_price']) - (stock2_shares * backtest_df.loc[today, 'stock2_price'])

    portfolio['daily_return'] = portfolio['total_value'].pct_change()

    plt.figure(figsize=(12, 6))
    portfolio['total_value'].plot(label='Portfolio Value')
    plt.title(f'Portfolio Equity Curve for {stock1_name} & {stock2_name}')
    plt.ylabel(f'Portfolio Value ($)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.show()

    daily_returns = portfolio['daily_return'].dropna()
    annualized_return = daily_returns.mean() * 252
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    if annualized_volatility == 0: sharpe_ratio = 0.0
    else: sharpe_ratio = annualized_return / annualized_volatility
        
    print("\n--- Final Portfolio Backtest Performance Metrics ---")
    print(f"Initial Investment: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Portfolio Value: ${portfolio['total_value'].iloc[-1]:,.2f}")
    print(f"Total Return: {(portfolio['total_value'].iloc[-1] / INITIAL_CAPITAL - 1) * 100:.2f}%")
    print(f"Annualized Return: {annualized_return * 100:.2f}%")
    print(f"Annualized Volatility: {annualized_volatility * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# GENERATING TEARSHEET
portfolio['daily_return'].name = f'{stock1_name} & {stock2_name} Strategy'

print("\nStep 5: Generating professional performance tearsheet...")

qs.reports.html(portfolio['daily_return'], 
                title=f'Pairs Trading Performance: {stock1_name} & {stock2_name}',
                output='strategy_tearsheet.html')

print("\n--- Project Complete ---")
print("A full performance report has been saved as 'strategy_tearsheet.html'")