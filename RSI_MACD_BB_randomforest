
from matplotlib import pyplot as plt
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

  
  

def fetch_data(ticker, start_date, end_date):

    """Fetch historical stock data."""
    data = yf.download(ticker, start=start_date, end=end_date)
    close_data = data[['Close']]
    return close_data

  

def calculate_indicators(data,array):

    """Calculate RSI, MACD, and Bollinger Bands."""

    for tick in array:

        data['RSI'] = RSIIndicator(data.xs(tick,level='Ticker',axis=1).squeeze()).rsi()
        macd = MACD(data.xs(tick,level='Ticker',axis=1).squeeze())
        data['MACD'] = macd.macd()
        data['Signal_Line'] = macd.macd_signal()
        bb = BollingerBands((data.xs(tick,level='Ticker',axis=1)).squeeze())
        data['Upper_Band'] = bb.bollinger_hband()
        data['Lower_Band'] = bb.bollinger_lband()

        #data['LogRSI'] = calculate_log_rsi(data)

    return data

  

def calculate_log_rsi(data, length=14):

    """"
    Calculate the custom Log RSI indicator.
    Parameters:
    data (pd.DataFrame): DataFrame containing a 'Close' column with closing prices.
    length (int): Lookback period for calculating the indicator.
    Returns:
    pd.Series: Log RSI values.
    """
    # Calculate price changes
    close_prices = data['Close'].squeeze() # Ensure 'Close' is a Series
    # Calculate price changes
    change = close_prices.diff()
    gain = np.where(change > 0, change, 0)
    loss = np.where(change < 0, -change, 0)
    # Calculate average gain and loss using a rolling mean
    avg_gain = pd.Series(gain).rolling(window=length, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=length, min_periods=1).mean()
    # Calculate Relative Strength (RS
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    # Calculate 1 - RS
    one_minus_rs = 1 - rs
    # Calculate custom RSI-like value
    custom_rsi = np.where(one_minus_rs != 0, close_prices - (close_prices / one_minus_rs), close_prices)
    # Apply logarithmic transformation
    log_rsi = np.where(custom_rsi != 0, np.sign(custom_rsi) * np.log(np.abs(custom_rsi) + 1), 0)
    return pd.Series(log_rsi, index=data.index, name='LogRSI')

    

def create_labels(data, n_days=7):

    """Create target labels for price movement."""

    close_prices = data.xs('Close', level='Price', axis=1)
    # Initialize a DataFrame for targets
    targets = pd.DataFrame(index=close_prices.index)
    # Calculate targets for each ticker

    for ticker in close_prices.columns:
        future_prices = close_prices[ticker].shift(-n_days) # Shift for future prices
        targets[ticker] = (future_prices > close_prices[ticker]).astype(int) # Binary target
        # Add 'Target' as a new column level in the original DataFrame
        targets = targets.add_prefix('Target_') # Prefix for clarity
        for col in targets.columns:
            data[('Target', col.split('_')[1])] = targets[col]
    return data.dropna(subset=[('Target', ticker) for ticker in close_prices.columns])

  

def stratified_split(data, features, target, test_size=0.2):

    """Perform stratified split to ensure ETF representation in train-test split."""

    X = data.loc(axis=1)[features].stack(level='Ticker')
    y = data.xs(target, level='Price', axis=1).stack()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    return X_train, X_test, y_train, y_test

  

def create_splits_per_ticker(data, features, target, model=None, test_size=0.2):
    """
    Train a model per ticker and display accuracy.
    Parameters:
    - data: MultiIndex DataFrame with Date and Ticker indices.
    - features: List of feature column names.
    - target: Name of the target column.
    - model: Machine learning model (default: RandomForestClassifier).
    - test_size: Proportion of data to use for testing.
    Returns:
    - results: Dictionary with tickers as keys and accuracies as values.
    """
    valid_tickers = [
    ticker for ticker in data.columns.get_level_values("Ticker").unique() if ticker != ''

    ]
    X = data.loc(axis=1)[features].dropna()
    splits = {}
    # Iterate over tickers
    for ticker in valid_tickers:
        print(f"Processing ticker: {ticker}")
    # Extract ticker-specific target
    y = data.xs(target, level="Price", axis=1)[ticker]
    # Align features and target
    common_index = X.index.intersection(y.index)
    X_ticker = X.loc[common_index]
    y_ticker = y.loc[common_index]
    # Prepare train/test splits for this ticker
    # Stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_idx, test_idx in splitter.split(X_ticker, y_ticker):
        X_train, X_test = X_ticker.iloc[train_idx], X_ticker.iloc[test_idx]
        y_train, y_test = y_ticker.iloc[train_idx], y_ticker.iloc[test_idx]
        # Store splits for this ticker
        splits[ticker] = [(X_train, y_train), (X_test, y_test)]
    return splits

  

def train_and_evaluate(data, splits, model=None):

    """

    Train and evaluate a model for each ticker.

    

    Parameters:

    - splits: Dictionary where keys are tickers and values are tuples:

    ( (X_train, y_train), (X_test, y_test) ).

    - model: Machine learning model (default: RandomForestClassifier).

    

    Returns:

    - results: Dictionary with tickers as keys and accuracy as values.

    """

    model = model or RandomForestClassifier(

    random_state=42,

    n_estimators=300,

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    max_features=None,

    )

    results = {}

    # Iterate over tickers

    # Iterate over tickers

    for ticker, ((X_train, y_train), (X_test, y_test)) in splits.items():

        print(f"Processing ticker: {ticker}")
        # Train the model
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Store results
        results[ticker] = accuracy
        add_predictions(data, ticker, y_pred)
        print(f"Accuracy for {ticker}: {accuracy:.2%}")
        importances = model.feature_importances_
        print("Variable importance", importances)
    return results

  

def train_and_evaluate_oob(data, features, model=None):
    """
    Train a Random Forest model with OOB scoring for each ticker and add predictions to the DataFrame.
    Prameters:
    - data: MultiIndex DataFrame with `Close`, `Target`, and feature columns.
    - features: List of feature column names.
    - model: RandomForestClassifier with oob_score=True (default if not provided).
    Returns:
    - Updated DataFrame with predictions added under the `Predictions` hierarchy.
    """

    # Initialize the model

    model = model or RandomForestClassifier(

    random_state=42,

    n_estimators=300,

    oob_score=True,

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    max_features=None,

    )
    oob_scores = {}
    predictions_dict = {} # Dictionary to store predictions for each ticker
    # Extract valid tickers
    valid_tickers = [
    ticker for ticker in data.columns.get_level_values("Ticker").unique() if ticker != ""

    ]
    # Prepare feature matrix
    X = data[features].fillna(data[features].mean())
    # Iterate over tickers
    for ticker in valid_tickers:
        print(f"Processing ticker: {ticker}")
    # Extract target values for the current ticker
        ticker_data = data.xs("Target", level="Price", axis=1)
        if ticker not in ticker_data.columns:
            print(f"Ticker {ticker} not found under Target. Skipping.")
            continue
        y = ticker_data[ticker].values
            # Train the model
        model.fit(X, y)
            # Get OOB accuracy
        oob_accuracy = model.oob_score_
        oob_scores[ticker] = oob_accuracy
        print(f"OOB Accuracy for {ticker}: {oob_accuracy:.2%}")
        # Generate predictions for all rows
        predictions_dict[ticker] = model.predict(X)
        # Add predictions to the DataFrame
        predictions_df = pd.DataFrame(index=data.index)
        for ticker, predictions in predictions_dict.items():
            predictions_df[ticker] = predictions    
        # Add 'Predictions' as a new level in the original DataFrame
        for ticker in predictions_df.columns:
            data[('Predictions', ticker)] = predictions_df[ticker]
    return data

  

def add_predictions(data, ticker, preds):

    """Add model predictions to the DataFrame."""

    data[('Predictions', ticker)] = preds

    return data

    

def optimize_model(X_train, y_train):
    """Optimize the model using GridSearchCV and return the best model."""

    param_grid = {

    'n_estimators': [ 200,300,400],

    'max_depth': [ None],

    'min_samples_split': [None,2],

    'min_samples_leaf': [1],

    'max_features': [None]

    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
    cv=5, scoring='f1', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    # Log the best parameters
    print("Best Parameters:", grid_search.best_params_)
    # Return the best model
    return grid_search.best_estimator_

  

def plot_feature_importance(model, features):

    """Plot feature importances."""

    importances = model.feature_importances_

    plt.bar(features, importances)

    plt.title("Feature Importances")

    plt.xlabel("Features")

    plt.ylabel("Importance")

    plt.show()

def backtest_portfolio(data, tickers, initial_balance=10000, confidence_threshold=0.7, contract_multiplier=1):
    """
    Backtest a trading strategy and return the portfolio as a DataFrame.
    Parameters:
    - data: DataFrame containing Close prices, Predictions, and other features.
    - tickers: List of tickers to include in the portfolio.
    - initial_balance: Starting portfolio balance.
    - confidence_threshold: Minimum confidence to execute trades.
    - contract_multiplier: Number of shares per contract.
    Returns:
    - portfolio: DataFrame summarizing the trades, signals, and returns for each ticker.
    """
    portfolio = pd.DataFrame(columns=["Ticker", "Contracts", "Entry Price", "Exit Price", "Returns"])
    balance = initial_balance
    entry_trades = {ticker: None for ticker in tickers} # Track open trades by ticker
    total_returns = 0
    min_profit = 1
    transaction_fee = 0.5
    for ticker in tickers:
        print(f"Processing ticker: {ticker}")
        for timestamp in data.index:
            price = data[('Close', ticker)].loc[timestamp]
            prediction = data[('Predictions', ticker)].loc[timestamp]
            confidence = data.get(('Confidence', ticker), pd.Series(1, index=data.index)).loc[timestamp]
            if price > 0:
                if prediction == 1: # Buy
                    if entry_trades[ticker] is None: # Only buy if no open position
                        contracts = (balance*0.1) // (price * contract_multiplier)
                    if contracts > 0:
                        cost = contracts * price * contract_multiplier
                        balance -= cost
                        entry_trades[ticker] = {"contracts": contracts, "entry_price": price}
                        print(f"{ticker} - BUY {contracts} contracts at {price}")
                elif prediction == 0: # Sell
                    if entry_trades[ticker] and entry_trades[ticker]['entry_price'] < price: # Only sell if there's an open position
                        last_trade = entry_trades[ticker]
                        profit = last_trade["contracts"] * (price - last_trade["entry_price"]) * contract_multiplier
                        profit -= last_trade["contracts"] * transaction_fee
                    if profit >= min_profit:
                        balance += profit
                        total_returns += profit
                        portfolio = pd.concat([
                        portfolio,
                        pd.DataFrame({
                        "Ticker": [ticker],
                        "Contracts": [last_trade["contracts"]],
                        "Entry Price": [last_trade["entry_price"]],
                        "Exit Price": [price],
                        "Returns": [profit]
                        })
                        ], ignore_index=True)
                        entry_trades[ticker] = None # Clear open position
                        print(f"{ticker} - SELL {last_trade['contracts']} contracts at {price}, Profit: {profit}")

    

    return portfolio , total_returns

  
  

def evaluate_strategy(portfolio, initial_balance):
    """
    Evaluate the performance of the backtested strategy.
    Parameters:
    portfolio (pd.DataFrame): DataFrame containing 'Portfolio Value' over time.
    initial_balance (float): The initial portfolio balance.
    Returns:
    None
    """
    final_balance = portfolio['Portfolio Value'].iloc[-1]
    cumulative_return = (final_balance - initial_balance) / initial_balance
    max_drawdown = ((portfolio['Portfolio Value'].cummax() - portfolio['Portfolio Value']) /
    portfolio['Portfolio Value'].cummax()).max()
    print(f"Final Portfolio Value: ${final_balance:.2f}")
    print(f"Cumulative Return: {cumulative_return:.2%}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    # Plot portfolio value
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio['Portfolio Value'], label='Portfolio Value', color='blue')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

  

def calculate_sharpe_ratio(portfolio, risk_free_rate=0.01, periods_per_year=252):
    """
    Calculate the Sharpe Ratio for the backtested strategy:
    Parameters:
    portfolio (pd.DataFrame): DataFrame containing 'Portfolio Value' over time.
    risk_free_rate (float): Annual risk-free rate (e.g., 0.01 for 1%).
    periods_per_year (int): Number of periods in a year (252 for daily, 52 for weekly).
    Returns:
    float: Sharpe Ratio.
    """
    # Calculate daily returns
    portfolio['Daily Returns'] = portfolio['Portfolio Value'].pct_change().dropna()
    # Average daily return and standard deviation
    avg_daily_return = portfolio['Daily Returns'].mean()
    std_dev_daily_return = portfolio['Daily Returns'].std()
    # Annualize returns and standard deviation
    risk_free_rate_daily = risk_free_rate / periods_per_year
    sharpe_ratio = (avg_daily_return - risk_free_rate_daily) / std_dev_daily_return * np.sqrt(periods_per_year)
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    return sharpe_ratio

  

if __name__ == "__main__":

# Parameters
    etfs = ["XLY", "XLE", "XLC", "XLP", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"]
    start_date = "2015-01-01"
    end_date = "2025-01-01"
    data = fetch_data(etfs, start_date, end_date)
    etf = ["XLY"]
    data = calculate_indicators(data,etfs)
    data = create_labels(data)
    # Features and Target
    features = [ 'MACD', 'Signal_Line', 'Upper_Band', 'Lower_Band','RSI']
    target = 'Target'

    #split = train_per_ticker(data,features,target)
    # Optimize model
    #rf_model = optimize_model(X_train, y_train)
    data = train_and_evaluate_oob(data, features)
    portfolio = backtest_portfolio(data, etfs)
    print(portfolio)

    #evaluate_strategy(portfolio, initial_balance=10000)

    # Evaluate performance per ETF
    #evaluate_per_etf(rf_model, combined_data, features, target)
    # Plot feature importance
    #plot_feature_importance(rf_model, features)

