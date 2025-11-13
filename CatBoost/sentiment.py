import pandas as pd
import numpy as np
import feedparser
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime

tickers_file = r"data/us_stock_data_2years_with_cleaned.csv"
sent_file = r"data/sentiment_scores.csv"

print("Loading tickers...")
df = pd.read_csv(tickers_file)
tickers = df[["Ticker","Company"]].drop_duplicates().reset_index(drop=True)
print("Tickers loaded:", len(tickers))

print("Loading FinBERT...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

def get_score(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return (probs[:,2] - probs[:,0]).cpu().numpy()

def grab_news(ticker):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    d = feedparser.parse(url)
    items = []
    for e in d.entries:
        pub_date = e.get("published", "")
        try:
            date_parsed = datetime(*e.published_parsed[:6]) if e.get("published_parsed") else None
        except Exception:
            date_parsed = None
        items.append({
            "title": e.title,
            "link": e.link,
            "published": date_parsed,
            "ticker": ticker
        })
    return items

print("Fetching headlines...")
all_news = []
for i, row in tickers.iterrows():
    t = str(row["Ticker"]).upper()
    name = str(row["Company"]).split()[0].upper() if pd.notna(row["Company"]) else None
    data = grab_news(t)
    for n in data:
        title = n["title"].upper()
        if t in title or (name and name in title):
            all_news.append(n)
    if i % 20 == 0:
        print(i, "/", len(tickers))
    time.sleep(0.3)

news = pd.DataFrame(all_news)
print("Headlines found:", len(news))

if len(news) > 0:
    print("Running sentiment model...")
    batch = 16
    scores = []
    for i in range(0, len(news), batch):
        texts = news.iloc[i:i+batch]["title"].tolist()
        sc = get_score(texts)
        scores.extend(sc)
    news["score"] = scores
    news["date"] = pd.to_datetime(news["published"]).dt.normalize()
    sent = news.groupby(["ticker","date"]).agg(
        Sentiment_Mean=("score","mean"),
        Sentiment_Std=("score","std"),
        Count=("score","size")
    ).reset_index()
else:
    sent = pd.DataFrame(columns=["ticker","date","Sentiment_Mean","Sentiment_Std","Count"])

# Fill NaNs and merge with past sentiment history
sent["Sentiment_Mean"] = sent["Sentiment_Mean"].fillna(0).round(5)
sent["Sentiment_Std"] = sent["Sentiment_Std"].fillna(0).round(5)

print("Merging with previous sentiment history...")
try:
    old = pd.read_csv(sent_file)
    old["date"] = pd.to_datetime(old["date"])
    sent = pd.concat([old, sent]).drop_duplicates(subset=["ticker","date"], keep="last")
except FileNotFoundError:
    pass

sent = sent.sort_values(["ticker","date"]).reset_index(drop=True)
sent.to_csv(sent_file, index=False)
print("Saved updated sentiment data:", sent_file)
print(sent.tail(10))
