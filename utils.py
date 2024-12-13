import pandas as pd
import numpy as np

def predict_top_stocks(model, recent_data, scaler, label_encoder, features, window_size=60):
    scaled_data = scaler.transform(recent_data[features])
    seq = scaled_data[-window_size:].reshape(1, window_size, -1)
    prob = model.predict(seq).flatten()

    unique_ticker_enc = recent_data['Ticker_encoded'].unique()
    preds = []
    for enc in unique_ticker_enc:
        preds.append((enc, prob[0]))

    df_preds = pd.DataFrame(preds, columns=['Ticker_encoded', 'Prediction'])
    df_preds['Ticker'] = label_encoder.inverse_transform(df_preds['Ticker_encoded'])
    df_preds.sort_values('Prediction', ascending=False, inplace=True)
    return df_preds

def compute_portfolio_returns(predictions, data, top_n=5):
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
            daily_return /= count
            portfolio_returns.append(daily_return)

        # Benchmark: Assuming SPY as proxy
        spy_today = data[(data['Date'] == current_date) & (data['Ticker'] == 'SPY')]['Close'].values
        spy_next = data[(data['Date'] == next_day) & (data['Ticker'] == 'SPY')]['Close'].values
        if len(spy_today) > 0 and len(spy_next) > 0:
            spy_ret = (spy_next[0] - spy_today[0]) / spy_today[0]
            benchmark_returns.append(spy_ret)
        else:
            benchmark_returns.append(0)

    # Convert to Series
    portfolio_returns = pd.Series(portfolio_returns, index=unique_dates[:len(portfolio_returns)])
    benchmark_returns = pd.Series(benchmark_returns, index=unique_dates[:len(benchmark_returns)])

    return portfolio_returns, benchmark_returns

def calculate_performance_metrics(portfolio_returns, benchmark_returns):
    metrics = {}

    # Portfolio metrics
    metrics['Portfolio_Cumulative_Return'] = (1 + portfolio_returns).prod() - 1
    metrics['Portfolio_Annualized_Return'] = portfolio_returns.mean() * 252
    metrics['Portfolio_Annualized_Volatility'] = portfolio_returns.std() * (252**0.5)
    metrics['Portfolio_Sharpe_Ratio'] = metrics['Portfolio_Annualized_Return'] / metrics['Portfolio_Annualized_Volatility']
    metrics['Portfolio_Max_Drawdown'] = (portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min()

    # Benchmark metrics
    metrics['Benchmark_Cumulative_Return'] = (1 + benchmark_returns).prod() - 1
    metrics['Benchmark_Annualized_Return'] = benchmark_returns.mean() * 252
    metrics['Benchmark_Annualized_Volatility'] = benchmark_returns.std() * (252**0.5)
    metrics['Benchmark_Sharpe_Ratio'] = metrics['Benchmark_Annualized_Return'] / metrics['Benchmark_Annualized_Volatility']
    metrics['Benchmark_Max_Drawdown'] = (benchmark_returns.cumsum() - benchmark_returns.cumsum().cummax()).min()

    return metrics