import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import joblib
from datetime import timedelta
import os

model_file = r"/Users/bharathkumar/Documents/Stock Price Prediction/CatBoost/data/vectrastock_catboost_model_norm.cbm"
meta_file = r"/Users/bharathkumar/Documents/Stock Price Prediction/CatBoost/data/model_meta_norm.pkl"
price_file = r"/Users/bharathkumar/Documents/Stock Price Prediction/CatBoost/data/us_stock_data_2years_with_cleaned.csv"
sent_file = r"/Users/bharathkumar/Documents/Stock Price Prediction/CatBoost/data/sentiment_scores.csv"
output_dir = r"predictions"

os.makedirs(output_dir, exist_ok=True)

print("\n===================================")
print("LOADING DATA...")
print("===================================\n")

p = pd.read_csv(price_file)
s = pd.read_csv(sent_file)
p["Date"] = pd.to_datetime(p["Date"])
s["date"] = pd.to_datetime(s["date"])

print("Merging sentiment data...")
s_sorted = s.sort_values(["ticker", "date"])
closest = []
for t in p["Ticker"].unique():
    tmp_p = p[p["Ticker"] == t]
    tmp_s = s_sorted[s_sorted["ticker"] == t]
    if len(tmp_s) == 0:
        continue
    for d in tmp_p["Date"]:
        idx = (tmp_s["date"] - d).abs().idxmin()
        r = tmp_s.loc[idx]
        closest.append({
            "Ticker": t,
            "Date": d,
            "Sentiment_Mean": r["Sentiment_Mean"],
            "Sentiment_Std": r["Sentiment_Std"],
            "Count": r["Count"]
        })

sent_fix = pd.DataFrame(closest)
df = p.merge(sent_fix, how="left", on=["Ticker", "Date"])
df = df.fillna(0)

print("Adding features...")
df["mean_price"] = df.groupby("Ticker")["Close"].transform("mean")
df["std_price"] = df.groupby("Ticker")["Close"].transform("std")
df["Close_norm"] = (df["Close"] - df["mean_price"]) / df["std_price"]

for lag in [1, 2, 3]:
    df[f"Close_lag{lag}"] = df.groupby("Ticker")["Close"].shift(lag)

df["rolling_mean_3"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(3).mean())
df["rolling_std_3"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(3).std())

df["Close_change"] = df.groupby("Ticker")["Close"].pct_change()

df["Sentiment_Mean_Scaled"] = df["Sentiment_Mean"] * 10
df["Sentiment_Std_Scaled"] = df["Sentiment_Std"] * 10
df["day_of_week"] = df["Date"].dt.dayofweek
df["month"] = df["Date"].dt.month
df["day_of_year"] = df["Date"].dt.dayofyear

df = df.dropna(subset=["Close_lag1", "rolling_mean_3"])

print("\nLoading model...")
model = CatBoostRegressor()
model.load_model(model_file)
meta = joblib.load(meta_file)
features = meta["features"]
cat_features = meta["cat_features"]

latest_date = df["Date"].max()
print(f"Latest available date: {latest_date}")

intervals = {
    "1D": {"steps": 12, "delta": timedelta(hours=2)},
    "1W": {"steps": 28, "delta": timedelta(hours=6)},
    "2W": {"steps": 14, "delta": timedelta(days=1)},
    "1M": {"steps": 30, "delta": timedelta(days=1)},
    "2M": {"steps": 60, "delta": timedelta(days=1)}
}

print("\n===================================")
print("PREDICTING FUTURE PRICES...")
print("===================================\n")

results_by_category = {key: [] for key in intervals.keys()}

# ========================================
# Per-ticker forecast generation with drift
# ========================================
for ticker in df["Ticker"].unique():
    tmp = df[df["Ticker"] == ticker].copy()
    if tmp.empty:
        continue

    print(f"Processing: {ticker}")
    for label, cfg in intervals.items():
        cur_date = latest_date
        pred_records = []
        past = tmp.sort_values("Date").tail(3).copy()

        for i in range(cfg["steps"]):
            current = past.tail(1).copy()
            pool = Pool(current[features], cat_features=cat_features)
            current["yhat_norm"] = model.predict(pool)
            current["yhat"] = current["yhat_norm"] * current["std_price"] + current["mean_price"]

            cur_date += cfg["delta"]

            # Record prediction
            res = current[["Ticker", "Company", "yhat",
                           "Sentiment_Mean_Scaled", "Sentiment_Std_Scaled", "Count"]].copy()
            res["Date"] = cur_date
            pred_records.append(res)

            # --- Next input with safe drift ---
            next_close = current["yhat"].values[0]
            scale = max(abs(next_close) * 0.0005, 1e-6)  # prevent scale < 0
            drift = np.random.normal(0, scale)
            if np.isnan(drift):
                drift = 0
            next_close = next_close + drift

            next_row = current.copy()
            next_row["Date"] = cur_date
            next_row["Close"] = next_close

            # Lag and rolling updates
            lag_values = list(past["Close"].values[-3:]) + [next_close]
            next_row["Close_lag1"] = lag_values[-2]
            next_row["Close_lag2"] = lag_values[-3]
            next_row["Close_lag3"] = lag_values[-4] if len(lag_values) > 3 else lag_values[-3]
            next_row["rolling_mean_3"] = np.mean(lag_values[-3:])
            next_row["rolling_std_3"] = np.std(lag_values[-3:])
            next_row["Close_change"] = (next_close - lag_values[-2]) / lag_values[-2]

            # Slight sentiment drift ±1%
            next_row["Sentiment_Mean_Scaled"] *= (1 + np.random.uniform(-0.01, 0.01))
            next_row["Sentiment_Std_Scaled"] *= (1 + np.random.uniform(-0.01, 0.01))

            # Time updates
            next_row["day_of_week"] = (current["day_of_week"].values[0] + 1) % 7
            next_row["month"] = cur_date.month
            next_row["day_of_year"] = cur_date.timetuple().tm_yday

            past = pd.concat([past, next_row], ignore_index=True)


print("\nSaving results...\n")
all_forecasts = []

for label, data_list in results_by_category.items():
    if not data_list:
        continue
    combined_df = pd.concat(data_list)
    combined_df = combined_df.sort_values(by=["Ticker", "Date"])
    out_file = os.path.join(output_dir, f"{label}_predictions.csv")
    combined_df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")
    all_forecasts.append(combined_df)

if all_forecasts:
    all_combined = pd.concat(all_forecasts)
    all_combined = all_combined.sort_values(by=["Ticker", "Date"])
    combined_file = os.path.join(output_dir, "all_predictions.csv")
    all_combined.to_csv(combined_file, index=False)
    print(f"Combined file saved: {combined_file}")

print("\nAll category CSVs saved successfully.")
print("Location:", output_dir)

print("\nCalculating model performance metrics...")

if "Close_norm" in df.columns:
    y_true = df["Close_norm"]
    pool = Pool(df[features], cat_features=cat_features)
    preds = model.predict(pool)

    mae = np.mean(np.abs(preds - y_true))
    rmse = np.sqrt(np.mean((preds - y_true) ** 2))
    r2 = 1 - np.sum((preds - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    df["preds"] = preds * df["std_price"] + df["mean_price"]
    df["true"] = df["Close"]
    mae_real = np.mean(np.abs(df["preds"] - df["true"]))
    rmse_real = np.sqrt(np.mean((df["preds"] - df["true"]) ** 2))
    mean_price = np.mean(df["true"])
    accuracy = (1 - mae_real / mean_price) * 100 if mean_price != 0 else 0

    df["pred_dir"] = np.sign(df["preds"].diff())
    df["true_dir"] = np.sign(df["true"].diff())
    direction_acc = np.mean(df["pred_dir"] == df["true_dir"]) * 100

    print("\n==== MODEL PERFORMANCE ====")
    print(f"MAE (normalized): {mae:.4f}")
    print(f"RMSE (normalized): {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print("---------------------------")
    print(f"MAE (price): ${mae_real:.2f}")
    print(f"RMSE (price): ${rmse_real:.2f}")
    print(f"Approx Accuracy: {accuracy:.2f}%")
    print(f"Direction Accuracy: {direction_acc:.2f}%")
    print("===========================\n")
else:
    print("No target data found for metrics.")

