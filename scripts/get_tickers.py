import requests
import pandas as pd
import os

def fetch_tickers():
    tickers = []

    urls = [
        "https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives&count=200&offset=0",
        "https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives&count=200&offset=200"
    ]

    for url in urls:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            print("Failed to fetch:", url)
            continue
        data = r.json()
        quotes = data.get("finance", {}).get("result", [])[0].get("quotes", [])
        for q in quotes:
            symbol = q.get("symbol")
            if symbol:
                tickers.append(symbol)

    if not tickers:
        print("No tickers found. Check API or connection.")
        return

    # build output path correctly based on your project root
    save_path = os.path.join("data", "tickers.csv")
    os.makedirs("data", exist_ok=True)

    df = pd.DataFrame(tickers, columns=["Ticker"])
    df.to_csv(save_path, index=False)
    print(f"Saved {len(tickers)} tickers to {save_path}")

if __name__ == "__main__":
    fetch_tickers()
