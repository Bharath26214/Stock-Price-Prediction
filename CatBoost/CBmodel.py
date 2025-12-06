import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import joblib
import os

price_file = r"/Users/bharathkumar/Documents/Stock Price Prediction/CatBoost/data/us_stock_data_2years_with_cleaned.csv"
sent_file  = r"/Users/bharathkumar/Documents/Stock Price Prediction/CatBoost/data/sentiment_scores.csv"
model_file = r"/Users/bharathkumar/Documents/Stock Price Prediction/CatBoost/data/vectrastock_catboost_model_norm.cbm"
meta_file  = r"/Users/bharathkumar/Documents/Stock Price Prediction/CatBoost/data/model_meta_norm.pkl"

print("Loading data...")
p = pd.read_csv(price_file)
s = pd.read_csv(sent_file)
p["Date"] = pd.to_datetime(p["Date"])
s["date"] = pd.to_datetime(s["date"])

print("Merging sentiment...")
df = p.merge(s, how="left", left_on=["Ticker", "Date"], right_on=["ticker", "date"]).fillna(0)

print("Adding temporal features...")
df["day_of_week"] = df["Date"].dt.dayofweek
df["month"] = df["Date"].dt.month
df["day_of_year"] = df["Date"].dt.dayofyear
df["Close_change"] = df.groupby("Ticker")["Close"].pct_change()
df["rolling_mean_3"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(3).mean())
df["rolling_std_3"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(3).std())

features = [
    "Ticker", "Open", "High", "Low", "Volume",
    "Sentiment_Mean", "Sentiment_Std", "Count",
    "day_of_week", "month", "day_of_year",
    "Close_change", "rolling_mean_3", "rolling_std_3"
]
cat_features = ["Ticker", "day_of_week", "month"]
target = "Close_norm"

print("Splitting data per ticker...")
train_list, test_list = [], []
for t, g in df.groupby("Ticker"):
    g = g.sort_values("Date")
    split = int(len(g) * 0.8)
    train_list.append(g.iloc[:split])
    test_list.append(g.iloc[split:])

train = pd.concat(train_list)
test  = pd.concat(test_list)

print("Normalizing per ticker...")
norm_map = train.groupby("Ticker")["Close"].agg(["mean","std"]).rename(columns={"mean":"mean_price","std":"std_price"})
train = train.merge(norm_map, on="Ticker", how="left")
test  = test.merge(norm_map, on="Ticker", how="left")

train["Close_norm"] = (train["Close"] - train["mean_price"]) / train["std_price"]
test["Close_norm"]  = (test["Close"] - test["mean_price"]) / test["std_price"]

X_train, y_train = train[features], train[target]
X_test, y_test   = test[features], test[target]

print("Training CatBoost...")
model = CatBoostRegressor(
    iterations=1500,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=5.0,
    random_strength=1.5,
    loss_function="RMSE",
    verbose=200,
)
model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test))

print("Saving model...")
model.save_model(model_file)
joblib.dump({"features": features, "cat_features": cat_features}, meta_file)

print("Evaluating performance...")
preds = model.predict(X_test)
mae = np.mean(np.abs(preds - y_test))
rmse = np.sqrt(np.mean((preds - y_test)**2))
r2 = 1 - np.sum((preds - y_test)**2) / np.sum((y_test - np.mean(y_test))**2)

test["preds"] = preds * test["std_price"] + test["mean_price"]
test["true"]  = test["Close"]

mae_price = np.mean(np.abs(test["preds"] - test["true"]))
rmse_price = np.sqrt(np.mean((test["preds"] - test["true"])**2))
accuracy = 100 - (mae_price / np.mean(test["true"])) * 100
dir_acc = np.mean(np.sign(test["preds"].diff()) == np.sign(test["true"].diff())) * 100

print("\n==== MODEL PERFORMANCE (Actual Prices) ====")
print(f"MAE: ${mae_price:.2f}")
print(f"RMSE: ${rmse_price:.2f}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Direction Accuracy: {dir_acc:.2f}%")
print("==========================================\n")

print("Model training complete.")
