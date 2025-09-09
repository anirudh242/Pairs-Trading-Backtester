# Pairs Trading Strategy Backtester 📈

A professional-grade quantitative research framework for discovering, optimizing, and backtesting market-neutral statistical arbitrage strategies. This project moves beyond simple backtesting by implementing a rigorous, multi-step workflow that prevents look-ahead bias, a critical flaw in naive financial modeling.

The framework systematically scans a universe of stocks, identifies cointegrated pairs using a regression-based hedge ratio, scientifically optimizes trading parameters on historical data, and then validates the strategy's performance on unseen future data, culminating in a full portfolio simulation and a professional performance report.

---

### **Technology Used**

- **Python:** The core language for the entire framework.
- **Pandas & NumPy:** For high-performance data manipulation and numerical analysis.
- **Statsmodels:** For rigorous statistical testing (ADF test) and linear regression (OLS).
- **yfinance:** For downloading historical financial market data.
- **QuantStats:** For generating professional-grade performance and risk analysis reports.
- **Matplotlib:** For data visualization of spreads and equity curves.

---

### **Key Innovations & Features**

This project demonstrates a complete, end-to-end quantitative research process:

- **Systematic Pair Discovery:** Implements an efficient two-step process to find tradable pairs from a large universe of stocks:

  1.  **Fast Filter:** Quickly identifies highly correlated candidates.
  2.  **Rigorous Test:** Runs an Augmented Dickey-Fuller (ADF) test to confirm cointegration.

- **Regression-Based Hedge Ratio:** Moves beyond a simple price ratio by using an OLS regression to calculate a dynamic hedge ratio (beta) for each pair. This creates a more statistically pure "spread" (the regression residual) based on the true equilibrium relationship.

- **Scientific Parameter Optimization:** Avoids guesswork by running a **Grid Search** on historical "in-sample" data to find the optimal Bollinger Band parameters (`window` and `std dev`) that maximize the Sharpe Ratio.

- **Look-ahead Bias Prevention:** Employs a professional **In-Sample / Out-of-Sample** testing methodology. Strategy optimization is performed _only_ on the training period (2015-2019), and the final, honest performance is evaluated on completely unseen future data (2020-2025).

- **Realistic Portfolio Simulation:** The backtester simulates a real-world portfolio, managing an initial capital, and sizing each trade to be **dollar-neutral** based on the calculated hedge ratio.

- **Professional Performance Reporting:** Generates a comprehensive HTML **tearsheet** using `quantstats`, detailing dozens of key performance and risk metrics like Sharpe Ratio, Calmar Ratio, Max Drawdown, and monthly returns.

---

### **Performance & Results**

The final strategy is validated on the out-of-sample period (Jan 2020 - Jan 2025) using the optimal parameters discovered during the in-sample training period.

- **Out-of-Sample Sharpe Ratio:** 0.98
- **Out-of-Sample Total Return:** 54.03%
- **Out-of-Sample Annualized Return:** 9.09%

---

### **Installation & Usage**

#### **Prerequisites**

- Python 3.10+
- `pip` and `venv`

#### **Setup**

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/anirudh242/pairs-trading-backtester.git](https://github.com/anirudh242/pairs-trading-backtester.git)
    cd pairs-trading-backtester
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare the `tickers.py` file:**
    - Ensure the `tickers.py` file is in the same directory and contains the lists of stocks you want to test (e.g., `SP100_TICKERS`).

#### **Running the Analysis**

- **Execute the main script:**
  ```bash
  python main.py
  ```
- The script will run the full pipeline: download data, find the best pair, optimize parameters, run the final backtest, display the equity curve, print the final metrics to the console, and generate the `strategy_tearsheet.html` report file to view
