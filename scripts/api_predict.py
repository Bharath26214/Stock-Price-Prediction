from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import time

app = FastAPI(
    title="VectraStock Prediction API",
    version="1.0",
    description="Public API to serve pre-trained stock predictions"
)

# --- CORS (allow your frontend only) ---
origins = [
    "http://localhost:5173",   # Vite dev
    "https://vectrastock.app", # production frontend (update if needed)
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(BASE_DIR, "..", "data", "predictions")

# --- Health check ---
@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "VectraStock ML API"}

# --- Prediction endpoint ---
@app.get("/api/v1/predictions/{ticker}")
async def get_predictions(ticker: str, request: Request):
    start_time = time.time()
    ticker = ticker.upper()
    path = os.path.join(PRED_DIR, f"{ticker}_predictions.csv")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No predictions found for {ticker}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    latency = round((time.time() - start_time) * 1000, 2)
    print(f"[INFO] Served {ticker} in {latency} ms")

    return {
        "ticker": ticker,
        "count": len(df),
        "latency_ms": latency,
        "data": df.to_dict(orient="records")
    }

# --- Optional: list available tickers ---
@app.get("/api/v1/tickers")
def list_tickers():
    files = [f.replace("_predictions.csv", "") for f in os.listdir(PRED_DIR) if f.endswith("_predictions.csv")]
    return {"tickers": sorted(files)}
