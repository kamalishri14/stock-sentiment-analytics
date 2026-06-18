# ============================================================
# Stock Market Sentiment Analytics — Plotly Dash Dashboard
# dashboard.py  |  Run: python dashboard.py
# Open: http://localhost:8050
# ============================================================
import os, datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from dash import Dash, dcc, html, dash_table, Input, Output, callback
import dash_bootstrap_components as dbc

BASE    = "."
TICKERS = ["AAPL", "TSLA", "NVDA"]
COLORS  = {"AAPL": "#7F77DD", "TSLA": "#D85A30", "NVDA": "#1D9E75"}

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Stock Sentiment Analytics")

# ── LOAD PRE-GENERATED DATA ───────────────────────────────────
def load_data():
    try:
        prices  = pd.read_csv(f"{BASE}/data/stock_prices.csv")
        sent    = pd.read_csv(f"{BASE}/data/news_sentiment.csv")
        summary = pd.read_csv(f"{BASE}/data/daily_summary.csv")
        prices["date"]  = pd.to_datetime(prices["date"])
        sent["date"]    = pd.to_datetime(sent["date"])
        summary["date"] = pd.to_datetime(summary["date"])
        return prices, sent, summary
    except Exception as e:
        print(f"Data load error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# ── LIVE PRICE (yfinance, optional) ──────────────────────────
def get_live_price(ticker):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="2d")
        if len(hist) >= 2:
            last  = hist["Close"].iloc[-1]
            prev  = hist["Close"].iloc[-2]
            return round(last, 2), round((last-prev)/prev*100, 2)
    except Exception:
        pass
    return None, None

# ── LAYOUT ───────────────────────────────────────────────────
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("📈 Stock Sentiment Analytics Dashboard",
                        className="my-3 fw-bold"), width=9),
        dbc.Col(dbc.Select(id="ticker-sel",
                           options=[{"label": t, "value": t} for t in TICKERS],
                           value="NVDA", className="my-3"), width=3),
    ]),
    dbc.Row(id="kpi-row", className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="candlestick"), width=8),
        dbc.Col(dcc.Graph(id="sent-gauge"),  width=4),
    ], className="mb-3"),
    dbc.Row([dbc.Col(dcc.Graph(id="sent-bar"), width=12)], className="mb-3"),
    dbc.Row([
        dbc.Col([
            html.H5("📰 Latest Headlines & Sentiment Scores", className="mb-2"),
            dash_table.DataTable(
                id="news-tbl",
                columns=[{"name": c, "id": c} for c in
                         ["headline","source","compound_score","sentiment_label"]],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "fontSize": 13,
                            "padding": "6px 10px", "whiteSpace": "normal"},
                style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
                style_data_conditional=[
                    {"if": {"filter_query": '{sentiment_label} = "positive"'},
                     "backgroundColor": "#E1F5EE"},
                    {"if": {"filter_query": '{sentiment_label} = "negative"'},
                     "backgroundColor": "#FAECE7"},
                ],
                page_size=8,
            )
        ], width=12),
    ]),
    dcc.Interval(id="refresh", interval=5*60*1000, n_intervals=0),
], fluid=True)

# ── CALLBACKS ────────────────────────────────────────────────
@callback(
    [Output("kpi-row","children"), Output("candlestick","figure"),
     Output("sent-gauge","figure"), Output("sent-bar","figure"),
     Output("news-tbl","data")],
    [Input("ticker-sel","value"), Input("refresh","n_intervals")]
)
def update(ticker, _):
    prices, sent, summary = load_data()
    color = COLORS.get(ticker, "#7F77DD")

    # KPI cards
    live_price, live_chg = get_live_price(ticker)
    sub_s = summary[summary["symbol"] == ticker]
    avg_sent = sub_s["avg_compound"].mean() if not sub_s.empty else 0
    n_arts   = int(sub_s["num_articles"].sum()) if not sub_s.empty else 0
    if live_price is None and not sub_s.empty:
        live_price = sub_s["close_price"].iloc[-1]
        live_chg   = sub_s["next_day_change"].iloc[-1]
    live_price = live_price or 0; live_chg = live_chg or 0
    chg_color  = "success" if live_chg >= 0 else "danger"
    chg_str    = f"{'+' if live_chg>=0 else ''}{live_chg:.2f}%"

    kpi_cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Current Price", className="text-muted mb-1", style={"fontSize":13}),
            html.H4(f"${live_price:.2f}", className="fw-bold mb-0")])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Price Change", className="text-muted mb-1", style={"fontSize":13}),
            html.H4(chg_str, className=f"fw-bold text-{chg_color} mb-0")])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Avg Sentiment (90d)", className="text-muted mb-1", style={"fontSize":13}),
            html.H4(f"{avg_sent:+.3f}", className="fw-bold mb-0")])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Articles Analysed", className="text-muted mb-1", style={"fontSize":13}),
            html.H4(f"{n_arts:,}", className="fw-bold mb-0")])), width=3),
    ])

    # Candlestick
    print("Available columns:", prices.columns)
    sub_p = prices[prices["ticker"]==ticker].sort_values("date")
    candle = go.Figure(go.Candlestick(
        x=sub_p["date"], open=sub_p["open"], high=sub_p["high"],
        low=sub_p["low"], close=sub_p["close"],
        increasing_line_color=color, decreasing_line_color="#D85A30",
    ))
    candle.update_layout(title=f"{ticker} — 90-Day Price Chart", height=380,
                         xaxis_rangeslider_visible=False, template="plotly_white",
                         margin=dict(l=40,r=20,t=45,b=30))

    # Gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=round(float(avg_sent), 3),
        title={"text": "90d Avg Sentiment"},
        delta={"reference": 0},
        gauge={"axis": {"range": [-1, 1]}, "bar": {"color": color},
               "steps": [{"range":[-1,-0.05],"color":"#FAECE7"},
                         {"range":[-0.05,0.05],"color":"#F1EFE8"},
                         {"range":[0.05,1],"color":"#E1F5EE"}]}
    ))
    gauge.update_layout(height=380, margin=dict(l=30,r=30,t=50,b=30), template="plotly_white")

    # Sentiment bar
    sub_s2 = sub_s.sort_values("date").tail(60)
    bar_colors = ["#1D9E75" if v >= 0 else "#D85A30" for v in sub_s2["avg_compound"]]
    sent_bar = go.Figure()
    sent_bar.add_bar(x=sub_s2["date"], y=sub_s2["avg_compound"],
                     marker_color=bar_colors, name="Daily Sentiment")
    sent_bar.update_layout(title=f"{ticker} — 60-Day Sentiment Timeline",
                           height=260, template="plotly_white",
                           margin=dict(l=40,r=20,t=45,b=30),
                           yaxis=dict(range=[-1,1], title="Compound Score"))

    # News table
    sub_n = sent[sent["ticker"]==ticker].sort_values("date", ascending=False).head(20)
    news_data = sub_n[["headline","source","compound_score","sentiment_label"]].to_dict("records")

    return kpi_cards, candle, gauge, sent_bar, news_data

if __name__ == "__main__":
    print("Dashboard → http://localhost:8050")
    app.run(debug=True, host="0.0.0.0", port=10000)
