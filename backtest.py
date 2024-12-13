# backtesting.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import load_model

import joblib

from dataCollection import download_stock_data, download_macro_data
from dataPreparation import (
    add_technical_indicators, add_performance_labels, add_time_features,
    merge_with_macro, prepare_dataset
)
from utils import predict_top_stocks

# Constants
TRAINING_TICKERS = [
    'MSFT', 'NFLX', 'AMZN', 'GOOG', 'META', 'TXN', 'NVDA', 'AAPL',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP',
    'BA', 'CAT', 'MMM', 'GE', 'HON',
    'PG', 'KO', 'PEP', 'WMT',
    'MCD', 'DIS', 'TGT', 'HD', 'LOW', 'SBUX', 'NKE', 'COST',
    'JNJ', 'PFE', 'MRK', 'ABT', 'AMGN', 'BMY', 'UNH',
    'T', 'VZ',
    'XOM', 'CVX', 'SLB',
    'IBM', 'INTC', 'CSCO', 'ORCL', 'ADBE', 'QCOM', 'HPQ',
    'FDX', 'UPS', 'DAL', 'LMT', 'BK', 'CHD', 'DE'
]

BACKTEST_TICKERS = [
    'BA', 'CAT', 'MMM', 'GE', 'HON',  # Industrial
    'PG', 'KO', 'PEP', 'WMT',           # Consumer Staples
    'MCD', 'DIS', 'TGT', 'HD', 'LOW', 'SBUX', 'NKE', 'COST',  # Consumer Discretionary
    'JNJ', 'PFE', 'MRK', 'ABT', 'AMGN', 'BMY', 'UNH',  # Healthcare
    'T', 'VZ',  # Telecommunications
    'XOM', 'CVX', 'SLB',  # Energy
    'IBM', 'INTC', 'CSCO', 'ORCL', 'ADBE', 'QCOM', 'HPQ',  # Technology
    'FDX', 'UPS', 'DAL', 'LMT', 'BK', 'CHD', 'DE'  # Additional Large Caps
]

# Ensure BACKTEST_TICKERS are not in TRAINING_TICKERS
BACKTEST_TICKERS = list(set(BACKTEST_TICKERS) - set(TRAINING_TICKERS))

# Parameters
START_DATE = '2023-01-01'  # Define your backtesting period
END_DATE = '2024-12-31'
WINDOW_SIZE = 60
TOP_N = 5  # Number of top performers to invest in each period
INITIAL_CAPITAL = 100000  # Starting with $100,000
ARTIFACT_DIR = 'artifacts'
PLOT_DIR = os.path.join(ARTIFACT_DIR, 'plots')

# Create directories if they don't exist
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def load_artifacts(artifact_dir='artifacts'):
    """Load the trained model, scaler, and label encoder without compiling."""
    model_path = os.path.join(artifact_dir, 'model.keras')
    scaler_path = os.path.join(artifact_dir, 'scaler.joblib')
    label_encoder_path = os.path.join(artifact_dir, 'label_encoder.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label Encoder file not found at {label_encoder_path}")
    
    # Load the Keras model without compiling
    model = load_model(model_path, compile=False)
    
    # Load scaler and label encoder using joblib
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    
    return model, scaler, label_encoder

def compute_portfolio_returns(predictions, data, top_n=5):
    """
    Compute portfolio returns based on model predictions.
    
    Args:
        predictions (pd.DataFrame): DataFrame with Date, Ticker, Prediction
        data (pd.DataFrame): Original stock data with Date, Ticker, Close
        top_n (int): Number of top stocks to invest in each period

    Returns:
        portfolio (pd.Series): Daily portfolio returns
        benchmark (pd.Series): Daily benchmark returns
    """
    # Initialize portfolio
    portfolio_returns = []
    benchmark_returns = []

    # Sort predictions by date
    predictions_sorted = predictions.sort_values('Date')

    # Unique dates in order
    unique_dates = predictions_sorted['Date'].unique()

    # Iterate over each date
    for current_date in unique_dates:
        # Select top N predictions
        daily_preds = predictions_sorted[predictions_sorted['Date'] == current_date]
        top_stocks = daily_preds.sort_values('Prediction', ascending=False).head(top_n)['Ticker'].values

        # Get next day's close prices for top stocks
        next_day = current_date + pd.Timedelta(days=1)
        next_day_data = data[data['Date'] == next_day]

        # Calculate returns
        daily_return = 0
        count = 0
        for stock in top_stocks:
            stock_price_today = data[(data['Date'] == current_date) & (data['Ticker'] == stock)]['Close'].values
            stock_price_next = data[(data['Date'] == next_day) & (data['Ticker'] == stock)]['Close'].values
            if len(stock_price_today) > 0 and len(stock_price_next) > 0:
                ret = (stock_price_next[0] - stock_price_today[0]) / stock_price_today[0]
                daily_return += ret
                count += 1
        if count > 0:
            daily_return /= count  # Average return

            # Append to portfolio returns
            portfolio_returns.append(daily_return)

        # Benchmark: S&P 500
        # Assuming SPY as a proxy
        spy_today = data[(data['Date'] == current_date) & (data['Ticker'] == 'SPY')]['Close'].values
        spy_next = data[(data['Date'] == next_day) & (data['Ticker'] == 'SPY')]['Close'].values
        if len(spy_today) > 0 and len(spy_next) > 0:
            spy_ret = (spy_next[0] - spy_today[0]) / spy_today[0]
            benchmark_returns.append(spy_ret)
        else:
            benchmark_returns.append(0)  # No change if SPY data not available

    # Convert to Series
    portfolio_returns = pd.Series(portfolio_returns, index=unique_dates[:len(portfolio_returns)])
    benchmark_returns = pd.Series(benchmark_returns, index=unique_dates[:len(benchmark_returns)])

    # Cumulative returns
    portfolio_cum = (1 + portfolio_returns).cumprod() * INITIAL_CAPITAL
    benchmark_cum = (1 + benchmark_returns).cumprod() * INITIAL_CAPITAL

    return portfolio_returns, benchmark_returns, portfolio_cum, benchmark_cum

def calculate_performance_metrics(portfolio_returns, benchmark_returns):
    """
    Calculate performance metrics for the portfolio and benchmark.
    
    Args:
        portfolio_returns (pd.Series): Daily portfolio returns
        benchmark_returns (pd.Series): Daily benchmark returns

    Returns:
        metrics (dict): Dictionary containing various performance metrics
    """
    metrics = {}

    # Portfolio metrics
    metrics['Portfolio_Cumulative_Return'] = (1 + portfolio_returns).prod() - 1
    metrics['Portfolio_Annualized_Return'] = portfolio_returns.mean() * 252
    metrics['Portfolio_Annualized_Volatility'] = portfolio_returns.std() * np.sqrt(252)
    metrics['Portfolio_Sharpe_Ratio'] = metrics['Portfolio_Annualized_Return'] / metrics['Portfolio_Annualized_Volatility']
    metrics['Portfolio_Max_Drawdown'] = (portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min()

    # Benchmark metrics
    metrics['Benchmark_Cumulative_Return'] = (1 + benchmark_returns).prod() - 1
    metrics['Benchmark_Annualized_Return'] = benchmark_returns.mean() * 252
    metrics['Benchmark_Annualized_Volatility'] = benchmark_returns.std() * np.sqrt(252)
    metrics['Benchmark_Sharpe_Ratio'] = metrics['Benchmark_Annualized_Return'] / metrics['Benchmark_Annualized_Volatility']
    metrics['Benchmark_Max_Drawdown'] = (benchmark_returns.cumsum() - benchmark_returns.cumsum().cummax()).min()

    return metrics

def plot_cumulative_returns(portfolio_cum, benchmark_cum):
    """Plot cumulative returns of the portfolio and benchmark."""
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_cum.index, portfolio_cum.values, label='Portfolio', color='blue')
    plt.plot(benchmark_cum.index, benchmark_cum.values, label='Benchmark (SPY)', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns ($)')
    plt.title('Cumulative Returns: Portfolio vs. Benchmark')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'cumulative_returns.png'), dpi=300)
    plt.close()
    print(f"Cumulative Returns plot saved to {os.path.join(PLOT_DIR, 'cumulative_returns.png')}")

def plot_drawdowns(portfolio_returns, benchmark_returns):
    """Plot drawdowns for the portfolio and benchmark."""
    portfolio_cum_returns = (1 + portfolio_returns).cumprod()
    portfolio_peak = portfolio_cum_returns.cummax()
    portfolio_drawdown = (portfolio_cum_returns - portfolio_peak) / portfolio_peak

    benchmark_cum_returns = (1 + benchmark_returns).cumprod()
    benchmark_peak = benchmark_cum_returns.cummax()
    benchmark_drawdown = (benchmark_cum_returns - benchmark_peak) / benchmark_peak

    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_drawdown.index, portfolio_drawdown.values, label='Portfolio Drawdown', color='red')
    plt.plot(benchmark_drawdown.index, benchmark_drawdown.values, label='Benchmark Drawdown', color='green')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.title('Drawdown Comparison: Portfolio vs. Benchmark')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'drawdown_comparison.png'), dpi=300)
    plt.close()
    print(f"Drawdown Comparison plot saved to {os.path.join(PLOT_DIR, 'drawdown_comparison.png')}")

def plot_daily_returns_distribution(portfolio_returns, benchmark_returns):
    """Plot distribution of daily returns for portfolio and benchmark."""
    plt.figure(figsize=(14, 7))
    sns.histplot(portfolio_returns, bins=50, color='blue', alpha=0.5, label='Portfolio', kde=True)
    sns.histplot(benchmark_returns, bins=50, color='orange', alpha=0.5, label='Benchmark (SPY)', kde=True)
    plt.xlabel('Daily Returns')
    plt.ylabel('Frequency')
    plt.title('Distribution of Daily Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'daily_returns_distribution.png'), dpi=300)
    plt.close()
    print(f"Daily Returns Distribution plot saved to {os.path.join(PLOT_DIR, 'daily_returns_distribution.png')}")

def plot_equity_curve(portfolio_cum, benchmark_cum):
    """Plot the equity curve of the portfolio and benchmark."""
    plt.figure(figsize=(14, 7))
    plt.plot(portfolio_cum.index, portfolio_cum.values, label='Portfolio Equity Curve', color='purple')
    plt.plot(benchmark_cum.index, benchmark_cum.values, label='Benchmark Equity Curve (SPY)', color='cyan')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.title('Equity Curve: Portfolio vs. Benchmark')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'equity_curve.png'), dpi=300)
    plt.close()
    print(f"Equity Curve plot saved to {os.path.join(PLOT_DIR, 'equity_curve.png')}")

def plot_performance_metrics(metrics):
    """Plot a table of performance metrics."""
    # Prepare data
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    metrics_df = metrics_df.reset_index()
    metrics_df.columns = ['Metric', 'Value']
    
    # Formatting
    def format_metric(row):
        if 'Return' in row['Metric'] or 'Drawdown' in row['Metric']:
            return f"{row['Value']:.2%}"
        elif 'Sharpe' in row['Metric']:
            return f"{row['Value']:.2f}"
        else:
            return row['Value']
    
    metrics_df['Formatted Value'] = metrics_df.apply(format_metric, axis=1)
    
    # Plot table
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    table = plt.table(cellText=metrics_df[['Metric', 'Formatted Value']].values,
                      colLabels=metrics_df[['Metric', 'Formatted Value']].columns,
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    plt.title('Performance Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'performance_metrics.png'), dpi=300)
    plt.close()
    print(f"Performance Metrics table saved to {os.path.join(PLOT_DIR, 'performance_metrics.png')}")

def plot_rolling_sharpe(portfolio_returns, window=252):
    """Plot rolling Sharpe ratio for the portfolio."""
    rolling_mean = portfolio_returns.rolling(window).mean() * 252
    rolling_std = portfolio_returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_std

    plt.figure(figsize=(14, 7))
    plt.plot(rolling_sharpe.index, rolling_sharpe.values, label=f'Rolling {window}-Day Sharpe Ratio', color='magenta')
    plt.xlabel('Date')
    plt.ylabel('Rolling Sharpe Ratio')
    plt.title(f'Rolling {window}-Day Sharpe Ratio')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'rolling_sharpe_ratio.png'), dpi=300)
    plt.close()
    print(f"Rolling Sharpe Ratio plot saved to {os.path.join(PLOT_DIR, 'rolling_sharpe_ratio.png')}")

def plot_stock_returns_heatmap(predictions, data, top_n=5):
    """Plot heatmap of stock returns based on predictions."""
    # Merge predictions with actual returns
    predictions = predictions.copy()
    predictions['Next_Date'] = predictions['Date'] + pd.Timedelta(days=1)
    
    # Initialize lists to store returns
    returns_list = []
    
    for idx, row in predictions.iterrows():
        ticker = row['Ticker']
        date = row['Date']
        next_date = row['Next_Date']
        
        # Get today's and next day's close prices
        today_close = data[(data['Date'] == date) & (data['Ticker'] == ticker)]['Close'].values
        next_close = data[(data['Date'] == next_date) & (data['Ticker'] == ticker)]['Close'].values
        
        if len(today_close) > 0 and len(next_close) > 0:
            ret = (next_close[0] - today_close[0]) / today_close[0]
            returns_list.append({'Date': date, 'Ticker': ticker, 'Return': ret})
    
    returns_df = pd.DataFrame(returns_list)
    
    if returns_df.empty:
        print("No return data available for heatmap.")
        return
    
    # Pivot table
    heatmap_data = returns_df.pivot_table(index='Date', columns='Ticker', values='Return', aggfunc='mean')
    
    # Plot heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(heatmap_data, cmap='coolwarm', center=0, cbar_kws={'label': 'Daily Return'})
    plt.xlabel('Ticker')
    plt.ylabel('Date')
    plt.title('Heatmap of Daily Returns for Selected Stocks')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'stock_returns_heatmap.png'), dpi=300)
    plt.close()
    print(f"Stock Returns Heatmap saved to {os.path.join(PLOT_DIR, 'stock_returns_heatmap.png')}")

def main():
    print("=== Backtesting Started ===")
    
    # Step 1: Load artifacts
    try:
        print("Loading trained model and artifacts...")
        model, scaler, label_encoder = load_artifacts(ARTIFACT_DIR)
        print("Artifacts loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading artifacts: {e}")
        return

    # Step 2: Data Collection
    print("Downloading stock data for backtesting...")
    stock_data = download_stock_data(BACKTEST_TICKERS + ['SPY'], START_DATE, END_DATE)  # Including SPY for benchmark
    if stock_data.empty:
        print("Error: No stock data downloaded. Please check your data source.")
        return
    print(f"Stock data downloaded. Shape: {stock_data.shape}")
    print("Sample stock data:")
    print(stock_data.head())

    print("Downloading macro data...")
    macro_data = download_macro_data(START_DATE, END_DATE)
    if macro_data.empty:
        print("Error: No macro data downloaded. Please check your data source.")
        return
    print(f"Macro data downloaded. Shape: {macro_data.shape}")
    print("Sample macro data:")
    print(macro_data.head())

    # Step 3: Data Preprocessing
    print("Preprocessing data...")
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    macro_data['Date'] = pd.to_datetime(macro_data['Date'])

    print("Adding technical indicators...")
    stock_data = add_technical_indicators(stock_data)
    print("Technical indicators added.")

    print("Adding time features...")
    stock_data = add_time_features(stock_data)
    print("Time features added.")

    print("Merging with macro data...")
    stock_data = merge_with_macro(stock_data, macro_data)
    print("Merged with macro data.")

    # Add performance labels (assuming we can compute future performance)
    print("Adding performance labels...")
    stock_data = add_performance_labels(stock_data, n=TOP_N, k=10)  # Adjust n and k as per strategy
    print("Performance labels added.")

    # Drop NaNs
    print("Dropping NaN values...")
    print(f"NaN count before dropping:\n{stock_data.isna().sum()}")
    stock_data.dropna(inplace=True)
    print(f"NaN count after dropping:\n{stock_data.isna().sum()}")

    # Prepare dataset
    print("Preparing dataset for model...")
    X, y, _, _, features = prepare_dataset(stock_data, BACKTEST_TICKERS + ['SPY'], WINDOW_SIZE)
    if X.size == 0:
        print("Error: Prepared dataset is empty. Check your data preprocessing steps.")
        return
    print("Dataset prepared.")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Feature scaling (assuming scaler was fitted on training data)
    print("Applying scaler to features...")
    try:
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        print("Scaling applied.")
    except Exception as e:
        print(f"Error during feature scaling: {e}")
        return

    # Predict probabilities
    print("Generating predictions...")
    try:
        predictions = model.predict(X_scaled, batch_size=1024)
        # Assuming binary classification and model outputs probabilities for class 1
        predictions = predictions.flatten()
        stock_data = stock_data.iloc[WINDOW_SIZE:]
        stock_data['Prediction'] = predictions
        print("Predictions generated.")
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return

    # Select top N predictions per day
    print("Selecting top N predictions per day...")
    try:
        top_predictions = stock_data.groupby('Date').apply(
            lambda x: x.nlargest(TOP_N, 'Prediction')
        ).reset_index(drop=True)
        if top_predictions.empty:
            print("Warning: No top predictions selected. Check your prediction outputs.")
        print("Top predictions selected.")
    except Exception as e:
        print(f"Error during selecting top predictions: {e}")
        return

    # Simulate trading strategy
    print("Simulating trading strategy...")
    try:
        portfolio_returns, benchmark_returns, portfolio_cum, benchmark_cum = compute_portfolio_returns(
            top_predictions, stock_data, top_n=TOP_N
        )
        if portfolio_returns.empty or benchmark_returns.empty:
            print("Warning: Portfolio or Benchmark returns are empty.")
        print("Trading simulation completed.")
    except Exception as e:
        print(f"Error during trading simulation: {e}")
        return

    # Calculate performance metrics
    print("Calculating performance metrics...")
    try:
        metrics = calculate_performance_metrics(portfolio_returns, benchmark_returns)
        print("Performance metrics calculated:")
        for key, value in metrics.items():
            if 'Drawdown' in key:
                print(f"{key}: {value:.2%}")
            elif 'Sharpe' in key:
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value:.2%}")
    except Exception as e:
        print(f"Error during performance metrics calculation: {e}")
        return

    # Visualization
    print("Generating performance plots...")
    try:
        plot_cumulative_returns(portfolio_cum, benchmark_cum)
        plot_drawdowns(portfolio_returns, benchmark_returns)
        plot_daily_returns_distribution(portfolio_returns, benchmark_returns)
        plot_equity_curve(portfolio_cum, benchmark_cum)
        plot_performance_metrics(metrics)
        plot_rolling_sharpe(portfolio_returns)
        plot_stock_returns_heatmap(top_predictions, stock_data, top_n=TOP_N)
        print("Performance plots generated.")
    except Exception as e:
        print(f"Error during plotting: {e}")
        return

    # Save backtest results
    try:
        backtest_results = {
            'metrics': metrics,
            'portfolio_returns': portfolio_returns.to_dict(),
            'benchmark_returns': benchmark_returns.to_dict(),
            'portfolio_cum': portfolio_cum.to_dict(),
            'benchmark_cum': benchmark_cum.to_dict()
        }

        with open(os.path.join(ARTIFACT_DIR, 'backtest_results.json'), 'w') as f:
            json.dump(backtest_results, f, default=str)
        print(f"Backtest results saved to {os.path.join(ARTIFACT_DIR, 'backtest_results.json')}.")
    except Exception as e:
        print(f"Error saving backtest results: {e}")
        return

    print("=== Backtesting Completed ===")

if __name__ == "__main__":
    main()
