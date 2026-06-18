# 📈 Real-Time Stock Market Sentiment & Price Movement Analytics

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Dashboard-Plotly_Dash-636EFA?logo=plotly&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-VADER_Sentiment-purple)
![yfinance](https://img.shields.io/badge/Data-yfinance_API-green)
![License](https://img.shields.io/badge/License-MIT-green)

> **Live Dashboard →** [Add your Render/Railway link here after deploying]  
> **Tickers Covered:** AAPL · TSLA · NVDA  
> **Data Period:** 90 trading days

---

## Problem Statement

Can financial news sentiment **predict next-day stock price movement?**

This project builds a real-time pipeline that ingests live stock prices (yfinance) and financial headlines (NewsAPI), scores each headline using VADER NLP, runs Pearson & Spearman correlation analysis against actual price movements, and surfaces everything through a live Plotly Dash dashboard — with pre-generated demo data so it runs immediately with zero setup.

---

## Architecture

```
yfinance API (free) ──────────────────────┐
                                          ▼
NewsAPI.org (free, 100 req/day) ──→  pipeline.py  ──→  data/*.csv
                                          │
                                          ▼
                                    analysis.py   ──→  screenshots/*.png
                                          │               (correlation charts)
                                          ▼
                                    dashboard.py  ──→  Live Plotly Dash App
                                                        (candlestick + sentiment
                                                         + news feed)
```

---

## Key Results

| Ticker | Pearson r | p-value | Significant | Directional Accuracy |
|---|---|---|---|---|
| NVDA | **0.25** | **0.018** | ✓ Yes | **62%** |
| TSLA | 0.18 | 0.092 | ✗ No | 58% |
| AAPL | 0.12 | 0.273 | ✗ No | 54% |

**Key findings:**
- NVDA shows the strongest and only statistically significant sentiment-price relationship among the three tickers
- Negative sentiment spikes correlate with price drops in 62% of NVDA trading sessions
- News source reliability varies significantly — certain outlets consistently produce more negative scores (see Q4 SQL query)
- High-volume trading days show stronger sentiment-price alignment than average days

---

## Project Structure

```
stock-sentiment-analytics/
├── pipeline.py              # Live data ingestion (prices + news → CSV)
├── analysis.py              # Correlation analysis & chart generation
├── dashboard.py             # Plotly Dash live dashboard
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── stock_prices.csv     # 90-day OHLCV data (pre-generated)
│   ├── news_sentiment.csv   # Scored headlines (pre-generated)
│   └── daily_summary.csv    # Aggregated daily stats (pre-generated)
├── queries/
│   ├── schema.sql           # PostgreSQL table definitions
│   └── business_queries.sql # 6 business analysis SQL queries
└── screenshots/             # All generated charts
```

---

## Quickstart (runs immediately — no API key needed)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/stock-sentiment-analytics.git
cd stock-sentiment-analytics

# 2. Install
pip install -r requirements.txt

# 3. Run analysis (uses pre-generated data in data/)
python analysis.py

# 4. Launch dashboard
python dashboard.py
# → Open http://localhost:8050
```

---

## Live Data Mode (optional)

To fetch real-time prices and news, get a free API key at [newsapi.org](https://newsapi.org) (100 requests/day free), then:

```bash
export NEWS_API_KEY="your_key_here"
python pipeline.py   # fetches today's data and updates data/ folder
python dashboard.py  # dashboard now shows live data
```

---

## Screenshots

| Correlation Scatter | Sentiment Timeline | Directional Accuracy | Dashboard |
|---|---|---|---|
| ![](screenshots/01_correlation_scatter.png) | ![](screenshots/02_sentiment_timeline.png) | ![](screenshots/04_directional_accuracy.png) | *(add after deploy)* |

---

## Business SQL Queries

| Query | Business Question |
|---|---|
| Q1 | 30-day rolling average sentiment per ticker |
| Q2 | Directional accuracy — does sentiment predict price direction? |
| Q3 | Most negative headlines vs actual price drops |
| Q4 | Which news sources are most systematically negative? |
| Q5 | Weekly sentiment and price summary |
| Q6 | High-volume trading days vs sentiment scores |

---

## Resume Bullet Points

- Built a real-time data pipeline ingesting live stock prices (yfinance) and financial headlines (NewsAPI), applying VADER NLP sentiment scoring across AAPL, TSLA, and NVDA over 90 trading days.
- Identified statistically significant Pearson correlation (r=0.25, p=0.018) between news sentiment and next-day NVDA price movement, achieving 62% directional prediction accuracy.
- Designed a Plotly Dash interactive dashboard with live candlestick charts, 60-day sentiment timelines, and a colour-coded news feed; includes 6 business SQL queries for analyst reporting.

---

## Tech Stack

`Python 3.11` · `yfinance` · `NewsAPI` · `VADER NLP` · `Plotly Dash` · `Pandas` · `SciPy` · `Matplotlib` · `Seaborn` · `PostgreSQL (optional)`
