# ============================================================
# Stock Market Sentiment Analytics — Data Pipeline
# pipeline.py  |  Run: python pipeline.py
#
# What this does:
#   1. Fetches live stock prices via yfinance (free, no key)
#   2. Fetches financial headlines via NewsAPI (free key at newsapi.org)
#   3. Scores each headline with VADER NLP sentiment
#   4. Saves everything to CSV files in data/
#
# Setup:
#   pip install -r requirements.txt
#   Get free NewsAPI key at https://newsapi.org
#   Set env var:  export NEWS_API_KEY="your_key_here"
#   Or paste key directly in the CONFIG block below
# ============================================================

import os, datetime, logging
import pandas as pd
import numpy as np
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── CONFIG ───────────────────────────────────────────────────
TICKERS   = ["AAPL", "TSLA", "NVDA"]
NEWS_KEY  = os.getenv("NEWS_API_KEY", "")   # paste your key here if not using env var
BASE      = "."
TODAY     = datetime.date.today()
YESTERDAY = TODAY - datetime.timedelta(days=1)

# ── FETCH STOCK PRICES (yfinance — no API key needed) ────────
def fetch_prices(days: int = 90) -> pd.DataFrame:
    start = TODAY - datetime.timedelta(days=days)
    log.info(f"Fetching prices for {TICKERS} from {start}")
    raw = yf.download(TICKERS, start=str(start), auto_adjust=True, progress=False)
    rows = []
    for ticker in TICKERS:
        try:
            closes = raw["Close"][ticker]
            for date, close in closes.items():
                if pd.isna(close): continue
                rows.append({
                    "ticker":     ticker,
                    "date":       date.date(),
                    "close":      round(float(close), 2),
                    "open":       round(float(raw["Open"][ticker][date]), 2),
                    "high":       round(float(raw["High"][ticker][date]), 2),
                    "low":        round(float(raw["Low"][ticker][date]), 2),
                    "volume":     int(raw["Volume"][ticker][date]),
                })
        except Exception as e:
            log.warning(f"Price error {ticker}: {e}")
    df = pd.DataFrame(rows)
    df["pct_change"] = df.groupby("ticker")["close"].pct_change() * 100
    return df

# ── FETCH NEWS + SCORE SENTIMENT ─────────────────────────────
def fetch_sentiment(days: int = 90) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    rows = []

    if NEWS_KEY:
        try:
            from newsapi import NewsApiClient
            api = NewsApiClient(api_key=NEWS_KEY)
            for ticker in TICKERS:
                log.info(f"Fetching news for {ticker}...")
                resp = api.get_everything(q=ticker, from_param=str(YESTERDAY),
                                          to=str(TODAY), language="en",
                                          sort_by="relevancy", page_size=30)
                for art in resp.get("articles", []):
                    hl = art.get("title", "")
                    if not hl or hl == "[Removed]": continue
                    sc = analyzer.polarity_scores(hl)
                    rows.append({
                        "ticker":          ticker,
                        "date":            TODAY,
                        "headline":        hl,
                        "source":          art.get("source", {}).get("name", ""),
                        "compound_score":  round(sc["compound"], 4),
                        "sentiment_label": ("positive" if sc["compound"] >= 0.05
                                            else "negative" if sc["compound"] <= -0.05
                                            else "neutral"),
                        "url": art.get("url", ""),
                    })
        except Exception as e:
            log.warning(f"NewsAPI error: {e} — using pre-generated data")

    if not rows:
        log.info("No live news fetched — loading pre-generated demo data")
        existing = f"{BASE}/data/news_sentiment.csv"
        if os.path.exists(existing):
            return pd.read_csv(existing)

    return pd.DataFrame(rows)

# ── BUILD DAILY SUMMARY ──────────────────────────────────────
def build_summary(prices: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    sent_agg = (sentiment.groupby(["ticker","date"])
                .agg(avg_compound=("compound_score","mean"),
                     num_articles=("compound_score","count"),
                     pct_positive=("sentiment_label", lambda x: (x=="positive").mean()*100),
                     pct_negative=("sentiment_label", lambda x: (x=="negative").mean()*100))
                .reset_index())
    sent_agg["date"] = pd.to_datetime(sent_agg["date"])
    prices["date"]   = pd.to_datetime(prices["date"])
    merged = sent_agg.merge(prices[["ticker","date","close"]], on=["ticker","date"], how="left")
    merged["next_day_change"] = merged.groupby("ticker")["close"].pct_change(periods=-1) * 100
    merged = merged.rename(columns={"close":"close_price"})
    return merged

# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=== Pipeline start ===")
    os.makedirs(f"{BASE}/data", exist_ok=True)

    prices = fetch_prices()
    prices.to_csv(f"{BASE}/data/stock_prices.csv", index=False)
    log.info(f"Prices saved: {len(prices)} rows")

    sentiment = fetch_sentiment()
    if not sentiment.empty:
        sentiment.to_csv(f"{BASE}/data/news_sentiment.csv", index=False)
        log.info(f"Sentiment saved: {len(sentiment)} rows")

    summary = build_summary(prices, sentiment)
    summary.to_csv(f"{BASE}/data/daily_summary.csv", index=False)
    log.info(f"Summary saved: {len(summary)} rows")

    print("\n=== Today's Sentiment Snapshot ===")
    snap = sentiment.groupby("ticker")["compound_score"].agg(["count","mean"]).round(3)
    print(snap)
    log.info("=== Pipeline complete ===")
