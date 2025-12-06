import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timedelta

data_dir = "/Users/bharathkumar/Documents/Stock Price Prediction/LSTM/data/historical"
model_dir = "/Users/bharathkumar/Documents/Stock Price Prediction/LSTM/data/models"
pred_dir = "/Users/bharathkumar/Documents/Stock Price Prediction/LSTM/data/predictions"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)

lookback = 30
future_days = 10
epochs = 30
batch_size = 32
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StockDataset(Dataset):
    def __init__(self, data, lookback):
        self.data = data
        self.lookback = lookback

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, i):
        x = self.data[i:i+self.lookback]
        y = self.data[i+self.lookback]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def prepare_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])
    return scaled, scaler

def train_model(ticker, data, scaler):
    dataset = StockDataset(data, lookback)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = LSTMModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MAELoss()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)  # FIXED: no unsqueeze
            loss = criterion(out, y)  # FIXED: no unsqueeze
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"{ticker} | Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.6f}")

    model_path = os.path.join(model_dir, f"{ticker}_lstm.pt")
    torch.save(model.state_dict(), model_path)
    print(f"{ticker}: model saved at {model_path}")
    return model

def predict_future(model, data, scaler):
    model.eval()

    # ensure data is a proper numpy array
    if isinstance(data, list):
        data = np.array(data)
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    if len(data) < lookback:
        raise ValueError("not enough data for prediction sequence")

    seq = data[-lookback:].tolist()
    preds = []

    for _ in range(future_days):
        seq_array = np.array(seq[-lookback:]).reshape(1, lookback, 1)
        seq_tensor = torch.tensor(seq_array, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(seq_tensor).cpu().item()
        preds.append(pred)
        seq.append([pred])  # keep it list of lists to preserve 2D shape

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    return preds.flatten()

def process_ticker(ticker):
    path = os.path.join(data_dir, f"{ticker}_data.csv")
    if not os.path.exists(path):
        print(f"{ticker}: data not found")
        return

    df = pd.read_csv(path)
    if df.empty or 'Close' not in df.columns:
        print(f"{ticker}: invalid data")
        return

    data, scaler = prepare_data(df)
    if len(data) <= lookback:
        print(f"{ticker}: not enough data to train")
        return

    model = train_model(ticker, data, scaler)

    try:
        preds = predict_future(model, data, scaler)
    except Exception as e:
        print(f"{ticker}: failed during prediction -> {e}")
        return

    last_date = pd.to_datetime(df['Date'].iloc[-1])
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': preds})
    pred_path = os.path.join(pred_dir, f"{ticker}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"{ticker}: predictions saved at {pred_path}")

def main():
    tickers = [f.split('_')[0] for f in os.listdir(data_dir) if f.endswith('_data.csv')]
    print(f"Training models for {len(tickers)} tickers")

    for t in tickers:
        try:
            process_ticker(t)
        except Exception as e:
            print(f"{t}: failed -> {e}")

    print("All models trained and predictions saved.")

if __name__ == "__main__":
    main()
