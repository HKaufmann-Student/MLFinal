import yfinance as yf
import pandas as pd
import os

def download_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    df['Ticker'] = ticker
    return df

def download_stock_data(tickers, start_date, end_date):
    stock_data_list = []
    for ticker in tickers:
        data = download_data(ticker, start_date, end_date)
        stock_data_list.append(data)
    all_data = pd.concat(stock_data_list, axis=0, ignore_index=True)
    return all_data

def download_macro_data(start_date, end_date):
    # S&P 500 proxy (SPY) and VIX
    spy = download_data('SPY', start_date, end_date)[['Date', 'Close']].copy()
    spy.rename(columns={'Close': 'SPY_Close'}, inplace=True)
    
    vix = download_data('^VIX', start_date, end_date)[['Date', 'Close']].copy()
    vix.rename(columns={'Close': 'VIX_Close'}, inplace=True)
    
    macro_df = pd.merge(spy, vix, on='Date', how='outer')
    macro_df.sort_values('Date', inplace=True)
    
    macro_df.ffill(axis=0, inplace=True)
    macro_df.bfill(axis=0, inplace=True)
    
    return macro_df

def save_raw_data(stock_df, macro_df, output_dir='data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stock_df.to_csv(os.path.join(output_dir, 'raw_stock_data.csv'), index=False)
    macro_df.to_csv(os.path.join(output_dir, 'raw_macro_data.csv'), index=False)
