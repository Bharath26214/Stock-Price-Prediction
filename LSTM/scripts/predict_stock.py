import os
import pandas as pd
import numpy as np
from model_utils import load_model, get_scaler, predict_future

LOOKBACK = 30
FUTURE_DAYS = 10

def run_prediction(ticker):
    ticker = ticker.upper()
    data_dir = "data/historical"
    pred_dir = "data/predictions"
    os.makedirs(pred_dir, exist_ok=True)

    data_path = os.path.join(data_dir, f"{ticker}_data.csv")
    if not os.path.exists(data_path):
        print(f"No data found for {ticker}")
        return

    df = pd.read_csv(data_path)
    if df.empty:
        print(f"No valid data for {ticker}")
        return

    scaler = get_scaler(ticker)
    model = load_model(ticker)

    scaled_data = scaler.transform(df[['Close']])
    preds = predict_future(model, scaled_data, scaler, lookback=LOOKBACK, future_days=FUTURE_DAYS)

    last_date = pd.to_datetime(df['Date'].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FUTURE_DAYS)
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": preds})

    pred_path = os.path.join(pred_dir, f"{ticker}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    print(f"\nPredictions for {ticker} (next {FUTURE_DAYS} days):\n")
    print(pred_df.to_string(index=False))
    print(f"\nSaved predictions at {pred_path}\n")

def main():
    ticker = input("Enter ticker: ").strip().upper()
    run_prediction(ticker)

if __name__ == "__main__":
    main()
