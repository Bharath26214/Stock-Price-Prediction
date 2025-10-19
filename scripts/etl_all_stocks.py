import os
import pandas as pd
import yfinance as yf
import ta
from datetime import datetime, timedelta

# folder setup
data_dir = "data/historical"
tickers_path = "data/tickers.csv"
years = 2  # exactly 2 years

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def add_indicators(df):
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['RSI_14'] = ta.momentum.RSIIndicator(pd.Series(df['Close']), window=14).rsi()
    df['Daily_Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df

def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return None
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return df

def update_or_create(ticker):
    make_dir(data_dir)
    path = os.path.join(data_dir, f"{ticker}_data.csv")
    end = datetime.now()
    start = end - timedelta(days=years * 365)

    df = fetch_data(ticker, start, end)
    if df is None or df.empty:
        print(f"{ticker}: no data found")
        return

    df = add_indicators(df)
    df.to_csv(path, index=False)
    print(f"{ticker}: new 2-year file created")

def main():
    if not os.path.exists(tickers_path):
        print("Tickers file not found. Run get_tickers.py first.")
        return

    tickers = pd.read_csv(tickers_path)['Ticker'].tolist()
    print(f"Total tickers: {len(tickers)}")

    for t in tickers:
        try:
            update_or_create(t)
        except Exception as e:
            print(f"{t}: failed -> {e}")

    print("All tickers processed.")

if __name__ == "__main__":
    main()
