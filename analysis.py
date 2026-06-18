# ============================================================
# Stock Market Sentiment Analytics — Correlation Analysis
# analysis.py  |  Run: python analysis.py
# ============================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

BASE   = "."
COLORS = {"AAPL": "#7F77DD", "TSLA": "#D85A30", "NVDA": "#1D9E75"}
TICKERS = ["AAPL", "TSLA", "NVDA"]

df = pd.read_csv(f"{BASE}/data/daily_summary.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.dropna(subset=["avg_compound", "next_day_change"])
print(f"Loaded {len(df)} records across {df['ticker'].nunique()} tickers\n")

# ── 1. CORRELATION ANALYSIS ──────────────────────────────────
print("=== Pearson Correlation: Sentiment → Next-Day Price Change ===")
corr_results = {}
for ticker in TICKERS:
    sub = df[df["ticker"] == ticker]
    r,  p  = stats.pearsonr(sub["avg_compound"], sub["next_day_change"])
    sr, sp = stats.spearmanr(sub["avg_compound"], sub["next_day_change"])
    correct = ((sub["avg_compound"]>0)&(sub["next_day_change"]>0)) | \
              ((sub["avg_compound"]<0)&(sub["next_day_change"]<0))
    dir_acc = round(correct.mean()*100, 1)
    corr_results[ticker] = {"r":r,"p":p,"sr":sr,"sp":sp,"dir_acc":dir_acc,"n":len(sub)}
    sig = "✓ Significant" if p < 0.05 else "✗ Not significant"
    print(f"  {ticker}: Pearson r={r:.3f} p={p:.4f} {sig} | "
          f"Spearman r={sr:.3f} | Directional accuracy={dir_acc}%")

# ── 2. SCATTER PLOTS ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, ticker in zip(axes, TICKERS):
    sub = df[df["ticker"]==ticker]; color = COLORS[ticker]
    r, p = corr_results[ticker]["r"], corr_results[ticker]["p"]
    ax.scatter(sub["avg_compound"], sub["next_day_change"],
               color=color, alpha=0.55, edgecolors="white", s=55)
    m, b = np.polyfit(sub["avg_compound"], sub["next_day_change"], 1)
    xl = np.linspace(sub["avg_compound"].min(), sub["avg_compound"].max(), 100)
    ax.plot(xl, m*xl+b, color=color, lw=2)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    sig = "✓ Significant" if p < 0.05 else "✗ Not significant"
    ax.set_title(f"{ticker}  r={r:.3f}, p={p:.3f}\n{sig}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sentiment Score"); ax.set_ylabel("Next-Day Price Change (%)")
plt.suptitle("News Sentiment → Next-Day Price Movement Correlation",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{BASE}/screenshots/01_correlation_scatter.png", dpi=150, bbox_inches="tight")
plt.close(); print("\nSaved: screenshots/01_correlation_scatter.png")

# ── 3. SENTIMENT + PRICE TIMELINE ────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
for row, ticker in enumerate(TICKERS):
    sub = df[df["ticker"]==ticker].sort_values("date"); color = COLORS[ticker]
    ax1 = fig.add_subplot(gs[row, 0]); ax2 = ax1.twinx()
    bar_colors = ["#1D9E75" if v >= 0 else "#D85A30" for v in sub["avg_compound"]]
    ax1.bar(sub["date"], sub["avg_compound"], color=bar_colors, alpha=0.7, width=1)
    ax2.plot(sub["date"], sub["close_price"], color=color, lw=2)
    ax1.set_ylabel("Sentiment", fontsize=9); ax2.set_ylabel("Price ($)", color=color, fontsize=9)
    ax1.set_title(f"{ticker} — Sentiment vs Price", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="x", rotation=30, labelsize=8)
    ax3 = fig.add_subplot(gs[row, 1])
    ax3.plot(sub["date"], sub["pct_positive"], color="#1D9E75", lw=1.5, label="% Positive")
    ax3.plot(sub["date"], sub["pct_negative"], color="#D85A30", lw=1.5, label="% Negative")
    ax3.fill_between(sub["date"], sub["pct_positive"], sub["pct_negative"], alpha=0.08, color=color)
    ax3.set_title(f"{ticker} — Positive vs Negative News %", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9); ax3.tick_params(axis="x", rotation=30, labelsize=8)
plt.suptitle("90-Day Sentiment & Price Timeline", fontsize=14, fontweight="bold")
plt.savefig(f"{BASE}/screenshots/02_sentiment_timeline.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved: screenshots/02_sentiment_timeline.png")

# ── 4. CROSS-TICKER HEATMAP ──────────────────────────────────
pivot = df.pivot(index="date", columns="ticker", values="avg_compound")
plt.figure(figsize=(7, 5))
sns.heatmap(pivot.corr(), annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, annot_kws={"size": 13})
plt.title("Cross-Ticker Sentiment Correlation", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{BASE}/screenshots/03_cross_ticker_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved: screenshots/03_cross_ticker_heatmap.png")

# ── 5. DIRECTIONAL ACCURACY BAR ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
accs = [corr_results[t]["dir_acc"] for t in TICKERS]
bars = ax.bar(TICKERS, accs, color=[COLORS[t] for t in TICKERS], edgecolor="white", width=0.5)
ax.axhline(50, color="gray", ls="--", lw=1.2, label="Random baseline (50%)")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{acc}%", ha="center", va="bottom", fontweight="bold", fontsize=13)
ax.set_ylim(0, 80); ax.set_ylabel("Directional Accuracy (%)")
ax.set_title("Sentiment Directional Accuracy vs Next-Day Price Movement",
             fontsize=13, fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig(f"{BASE}/screenshots/04_directional_accuracy.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved: screenshots/04_directional_accuracy.png")

print("\n=== KEY FINDINGS ===")
for t, res in corr_results.items():
    sig = "significant" if res["p"] < 0.05 else "not significant"
    print(f"  {t}: r={res['r']:.3f} ({sig}), directional accuracy={res['dir_acc']}%")
print("\nAnalysis complete. Run: python dashboard.py")
