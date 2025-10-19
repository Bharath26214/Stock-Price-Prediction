import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# --- Reuse same model structure as training ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- Load saved model ---
def load_model(ticker, model_dir="data/models"):
    path = os.path.join(model_dir, f"{ticker}_lstm.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model file found for {ticker}")
    model = LSTMModel()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

# --- Prepare scaler using historical data ---
def get_scaler(ticker, data_dir="data/historical"):
    path = os.path.join(data_dir, f"{ticker}_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No historical data found for {ticker}")
    df = pd.read_csv(path)
    scaler = MinMaxScaler()
    scaler.fit(df[['Close']])
    return scaler

# --- Predict N future days ---
def predict_future(model, data, scaler, lookback=30, future_days=10):
    seq = data[-lookback:].tolist()
    preds = []

    for _ in range(future_days):
        seq_array = np.array(seq[-lookback:]).reshape(1, lookback, 1)
        seq_tensor = torch.tensor(seq_array, dtype=torch.float32)
        with torch.no_grad():
            pred = model(seq_tensor).cpu().item()
        preds.append(pred)
        seq.append([pred])

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    return preds.flatten()
