# main.py
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from dataCollection import download_stock_data, download_macro_data, save_raw_data
from dataPreparation import (
    add_technical_indicators, add_performance_labels, add_time_features,
    merge_with_macro, prepare_dataset, save_prepared_data
)
from LSTMModelDefinition import create_classification_model
from modelTraining import train_and_evaluate_model, save_metrics
from utils import predict_top_stocks

if __name__ == "__main__":
    tickers = list(set([
        # Original 8
        'MSFT', 'NFLX', 'AMZN', 'GOOG', 'META', 'TXN', 'NVDA', 'AAPL',

        # Software and Services
        'CRM', 'INTU', 'NOW', 'ADSK', 'WDAY', 'TEAM', 'PANW', 'FTNT',
        'CRWD', 'OKTA', 'DDOG', 'SNOW', 'ZM', 'DOCU', 'DBX', 'BOX',

        # Semiconductors and Semiconductor Equipment
        'AMD', 'AVGO', 'MU', 'AMAT', 'LRCX', 'KLAC', 'ADI', 'MRVL', 'ON', 
        'NXPI', 'SWKS', 'MCHP', 'ASML', 'TER',

        # Technology Hardware, Storage, and Peripherals
        'DELL', 'HPQ', 'STX', 'WDC', 'NTAP', 'PSTG', 'LOGI', 'XRX', 
        'FJTSY', 'LNVGY', 'SSNLF', 'SONY', 'RICOY', 'BRTHY', 
        'NIPNF', 'WDC',

        # Internet and Direct Marketing Retail
        'EBAY', 'BABA', 'JD', 'PDD', 'ETSY', 'W', 'SHOP', 'MELI', 'RKUNY', 
        'REAL', 'CHWY', 'CPNG', 'ZLNDY', 'ASOMY', 'BHOOY', 
        'OCDDY', 'SFIX',

        # IT Services
        'IBM', 'ACN', 'CTSH', 'INFY', 'TCS', 'WIT', 'CAPMF', 
        'DXC', 'EPAM', 'GLOB', 'DAVA', 'AEXAY', 'GIB', 
        'LTI',

        # Communications Equipment
        'CSCO', 'JNPR', 'ANET', 'CIEN', 'FFIV', 'MSI', 'NOK', 'ERIC', 'GLW', 
        'LITE', 'INFN', 'VIAV', 'UI', 'COMM', 'EXTR', 'CALX', 'ADTN', 
        'CMBM', 'RBBN',

        # Electronic Equipment, Instruments, and Components
        'TEL', 'APH', 'GLW', 'KEYS', 'FLEX', 'JBL', 'ARW', 'AVT', 'CDW', 
        'LFUS', 'TTMI', 'SANM', 'BHE', 'PLXS', 'BDC', 
        'VSH', 'ROG',

        # Interactive Media and Services
        'PINS', 'SNAP', 'GOOG', 'META', 'TCEHY', 'BIDU', 'WB', 
        'YELP', 'IAC', 'GRPN', 'MOMO', 'DOYU', 'YY', 'HUYA', 
        'IQ', 'BILI', 'KWEB', 'SE',

        # Additional Technology Stocks
        'FDX', 'UPS', 'DAL', 'LMT', 'BK', 'CHD', 'DE', 'RBLX', 'PLTR', 
        'U', 'NET', 'FSLY', 'ZS', 'RNG', 'TWLO', 'CRSP', 'EDIT', 'NTLA', 
        'VEEV', 'TDOC'
    ]))

    start_date = '2010-01-01'
    end_date = '2023-10-01'
    window_size = 60

    print("=== Starting Data Collection ===")
    stock_data = download_stock_data(tickers, start_date, end_date)
    print(f"Stock data downloaded. Shape: {stock_data.shape}")
    print("Stock data sample:")
    print(stock_data.head())

    macro_data = download_macro_data(start_date, end_date)
    print("Macro data downloaded.")
    print(f"Macro data shape: {macro_data.shape}")
    print("Macro data sample:")
    print(macro_data.head())

    save_raw_data(stock_data, macro_data, output_dir='data')
    print("Raw data saved to 'data' directory.")

    print("=== Data Preparation ===")
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    macro_data['Date'] = pd.to_datetime(macro_data['Date'])

    print("Adding technical indicators...")
    stock_data = add_technical_indicators(stock_data)
    print("Technical indicators added. Sample after indicators:")
    print(stock_data.head())

    print("Adding time features...")
    stock_data = add_time_features(stock_data)
    print("Time features added. Sample of time-related columns:")
    print(stock_data[['Date', 'Year', 'Month', 'DayOfWeek']].head())

    print("Merging with macro data...")
    stock_data = merge_with_macro(stock_data, macro_data)
    print("Merged with macro data. Sample of macro columns:")
    print(stock_data[['Date', 'SPY_Close', 'VIX_Close']].head())

    # Change k according to how many stocks you want labelled to "top performers"
    print("Adding performance labels...")
    stock_data = add_performance_labels(stock_data, n=5, k=20)
    print("Performance labels added. Sample of performance and label columns:")
    print(stock_data[['Date', 'Ticker', 'Close', 'Performance', 'Top_Performer']].head(10))

    print("NaN count before dropping:")
    print(stock_data.isna().sum())
    before_drop = len(stock_data)
    stock_data.dropna(inplace=True)
    after_drop = len(stock_data)
    print(f"Dropped {before_drop - after_drop} rows due to NaN values. Remaining rows: {after_drop}")

    top_dist = stock_data['Top_Performer'].value_counts()
    print("Distribution of Top_Performer:")
    print(top_dist)
    if 1 in top_dist:
        pos_ratio = top_dist[1] / (top_dist[0] + top_dist[1]) * 100
        print(f"Percentage of top performers: {pos_ratio:.2f}%")
    else:
        print("No top performers found after filtering.")

    top_by_day = stock_data.groupby('Date')['Top_Performer'].sum()
    print("Some stats about daily top performers:")
    print(f"Mean top performers per day: {top_by_day.mean():.2f}")
    print(f"Median top performers per day: {top_by_day.median():.2f}")
    print("Example days with their top performer counts:")
    print(top_by_day.head(10))

    print("Preparing dataset for model...")
    X, y, features = prepare_dataset(stock_data, tickers, window_size)
    print("Dataset prepared.")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("Feature list:", features)
    print("Sample y values:", y[:10])

    '''
    #Save the prepared data
    print("Saving prepared data...")
    save_prepared_data(X, y, features, output_dir='processed_data')
    print("Prepared data saved to 'processed_data' directory.")
    '''

    y_counts = np.bincount(y)
    print("Distribution of labels after sequencing:")
    for i, count in enumerate(y_counts):
        perc = count / len(y) * 100
        print(f"Label {i}: {count} samples ({perc:.2f}%)")

    '''
    print("Checking feature variability in the final dataset...")
    X_flat = X.reshape(-1, X.shape[-1])
    std_values = np.std(X_flat, axis=0)
    low_variance_indices = np.where(std_values < 1e-6)[0]
    for idx in low_variance_indices:
        feat = features[idx]
        std_val = std_values[idx]
        print(f"Feature '{feat}' has very low std. dev. ({std_val:.6f}), may not be informative.")
    print("Feature variability check complete.")
    '''

    # Compute class weights for imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights = dict(enumerate(class_weights))
    print("Class weights computed:", class_weights)

    print("=== Training & Evaluation ===")
    results, trained_model = train_and_evaluate_model(
        X, y, 
        create_classification_model, 
        class_weight=class_weights,  # Corrected parameter name
        n_splits=5, 
        features=features, 
        output_dir='artifacts'
    )
    print("Cross-validation completed.")
    for fold_idx, (acc, prec, rec, f1) in enumerate(results, start=1):
        print(f"Fold {fold_idx}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    print("Saving metrics...")
    save_metrics(results, 'artifacts')
    print("Metrics saved to 'artifacts' directory.")

    print("=== Prediction Example ===")
    recent_date = stock_data['Date'].max()
    recent_data = stock_data[stock_data['Date'] >= (recent_date - pd.Timedelta(days=window_size*2))].sort_values('Date')

    print(f"Recent data for prediction. Shape: {recent_data.shape}")
    print("Recent data sample:")
    print(recent_data[['Date', 'Ticker', 'Close', 'Top_Performer']].head())

    top_stocks = predict_top_stocks(trained_model, recent_data, None, None, features, window_size)  # Adjust if scaler is needed
    print("Top Stocks Prediction:")
    print(top_stocks.head(10))
    print("=== Pipeline Completed ===")