# dataPreparation.py

import pandas as pd
import numpy as np
import ta
import os
import json

def download_data(ticker, start_date, end_date):
    """
    Placeholder function to download stock data.
    Replace this with your actual data downloading logic.
    """
    # Example using yfinance
    import yfinance as yf
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

def download_macro_data(start_date, end_date):
    """
    Downloads and processes macro data (SPY and VIX).
    """
    # S&P 500 proxy (SPY) and VIX
    
    # Download and process SPY data
    spy = download_data('SPY', start_date, end_date)[['Date', 'Close']].copy()
    spy.rename(columns={'Close': 'SPY_Close'}, inplace=True)
    
    # Download and process VIX data
    vix = download_data('^VIX', start_date, end_date)[['Date', 'Close']].copy()
    vix.rename(columns={'Close': 'VIX_Close'}, inplace=True)
    
    # Merge SPY and VIX data on Date
    macro_df = pd.merge(spy, vix, on='Date', how='outer')
    macro_df.sort_values('Date', inplace=True)
    
    # Forward-fill and backward-fill to handle missing values using recommended methods
    macro_df = macro_df.ffill().bfill()
    
    return macro_df

def add_technical_indicators(df):
    """
    Adds various technical indicators to the DataFrame.
    """
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['Stoch_RSI'] = ta.momentum.StochRSIIndicator(close=df['Close'], window=14).stochrsi()
    df['TSI'] = ta.momentum.TSIIndicator(close=df['Close'], window_slow=25, window_fast=13).tsi()

    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    cci = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20).cci()
    df['CCI'] = cci

    dpo = ta.trend.DPOIndicator(close=df['Close'], window=20).dpo()
    df['DPO'] = dpo

    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()

    atr = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR'] = atr.average_true_range()

    ulcer = ta.volatility.UlcerIndex(close=df['Close'], window=14).ulcer_index()
    df['Ulcer_Index'] = ulcer

    obv = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df['OBV'] = obv

    cmf = ta.volume.ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=20).chaikin_money_flow()
    df['CMF'] = cmf

    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10).mean()

    return df

def add_performance_labels(df, n=5, k=20):
    """
    Adds performance labels to the DataFrame indicating top performers.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing stock data.
    - n (int): Number of days ahead to calculate performance.
    - k (int): Number of top performers to label.

    Returns:
    - pd.DataFrame: DataFrame with added 'Performance' and 'Top_Performer' columns.
    """
    # Calculate performance n days ahead for each ticker
    df['Performance'] = (df.groupby('Ticker')['Close'].shift(-n) - df['Close']) / df['Close']
    # Rank performance within each date across all tickers
    df['Top_Performer'] = df.groupby('Date')['Performance'].rank(ascending=False, method='first') <= k
    df['Top_Performer'] = df['Top_Performer'].astype(int)
    return df

def add_time_features(df):
    """
    Adds time-related features to the DataFrame.
    """
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    return df

def merge_with_macro(df, macro_df):
    """
    Merges stock data with macro data on 'Date'.
    """
    df = pd.merge(df, macro_df, on='Date', how='left')
    df.sort_values('Date', inplace=True)
    
    # Forward-fill and backward-fill to handle missing values using recommended methods
    df[["SPY_Close", "VIX_Close"]] = df[["SPY_Close", "VIX_Close"]].ffill().bfill()
    
    return df

def prepare_dataset(df, tickers, window_size=60):
    """
    Prepares the dataset for model training by creating sequences.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing all data.
    - tickers (list): List of ticker symbols.
    - window_size (int): Number of time steps in each sequence.

    Returns:
    - X (np.ndarray): Feature sequences.
    - y (np.ndarray): Labels.
    - features (list): List of feature names.
    """
    # One-Hot Encode the 'Ticker' column
    df = pd.get_dummies(df, columns=['Ticker'], drop_first=True)
    
    # Define feature columns (excluding 'Date', 'Performance', 'Top_Performer')
    features = [
        'Close', 'SMA_10', 'EMA_10', 'RSI', 'Stoch_RSI', 'TSI', 'MACD', 'MACD_Signal',
        'CCI', 'DPO', 'BB_High', 'BB_Low', 'ATR', 'Ulcer_Index', 'OBV', 'CMF',
        'SPY_Close', 'VIX_Close', 'Year', 'Month', 'DayOfWeek'
    ]
    
    # Include all One-Hot Encoded Ticker columns
    ticker_cols = [col for col in df.columns if col.startswith('Ticker_')]
    features.extend(ticker_cols)
    
    # Impute NaN values for technical indicators using recommended methods
    df[features] = df[features].ffill().bfill()
    
    # Drop rows with missing 'Top_Performer' labels
    df = df.dropna(subset=['Top_Performer']).copy()
    
    X_list, y_list = [], []
    
    # For each ticker, create sequences
    for ticker in tickers:
        ticker_col = f'Ticker_{ticker}'
        if ticker_col not in df.columns:
            # Handle tickers that were dropped due to no data
            continue
        ticker_data = df[df[ticker_col] == 1].sort_values('Date')
        ticker_features = ticker_data[features].values
        y_values = ticker_data['Top_Performer'].values

        for i in range(window_size, len(ticker_features)):
            X_list.append(ticker_features[i - window_size:i])
            y_list.append(y_values[i])
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # **Removed Global Scaling Here**
    # Scaling will be handled within each fold in modelTraining.py
    
    return X, y, features

def save_prepared_data(X, y, features, output_dir='processed_data'):
    """
    Saves the prepared dataset (X and y) along with feature metadata to the specified directory.

    Parameters:
    - X (np.ndarray): Feature sequences.
    - y (np.ndarray): Labels.
    - features (list): List of feature names.
    - output_dir (str): Directory where the data will be saved.

    Saves:
    - X.npy: NumPy array of features.
    - y.npy: NumPy array of labels.
    - features.json: JSON file containing feature names.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define file paths
    X_path = os.path.join(output_dir, 'X.npy')
    y_path = os.path.join(output_dir, 'y.npy')
    features_path = os.path.join(output_dir, 'features.json')

    # Save X and y as .npy files
    np.save(X_path, X)
    print(f"Features (X) saved to '{X_path}'.")

    np.save(y_path, y)
    print(f"Labels (y) saved to '{y_path}'.")

    # Save features list as a JSON file
    with open(features_path, 'w') as f:
        json.dump(features, f, indent=4)
    print(f"Feature metadata saved to '{features_path}'.")

def load_prepared_data(input_dir='processed_data'):
    """
    Loads the prepared dataset (X and y) along with feature metadata from the specified directory.

    Parameters:
    - input_dir (str): Directory from where the data will be loaded.

    Returns:
    - X (np.ndarray): Feature sequences.
    - y (np.ndarray): Labels.
    - features (list): List of feature names.
    """
    X_path = os.path.join(input_dir, 'X.npy')
    y_path = os.path.join(input_dir, 'y.npy')
    features_path = os.path.join(input_dir, 'features.json')

    # Load X and y
    X = np.load(X_path)
    print(f"Features (X) loaded from '{X_path}'.")

    y = np.load(y_path)
    print(f"Labels (y) loaded from '{y_path}'.")

    # Load features list
    with open(features_path, 'r') as f:
        features = json.load(f)
    print(f"Feature metadata loaded from '{features_path}'.")

    return X, y, features