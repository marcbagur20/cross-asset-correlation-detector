import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from statsmodels.tsa.stattools import coint

def download_prices(tickers, start_date):
    """Download adjusted close prices from Yahoo."""
    data = yf.download(tickers, start=start_date)["Close"]
    return data

def compute_returns(prices):
    """Compute daily log returns."""
    return np.log(prices / prices.shift(1)).dropna()

def rolling_correlation_pair(returns, a, b, window):
    """Compute rolling correlation between two assets."""
    return returns[a].rolling(window).corr(returns[b])

def compute_corr_zscore(series):
    """Compute the z-score of a correlation series."""
    mean = series.mean()
    std = series.std()
    z = (series - mean) / std
    return z, mean, std

def detect_anomalies_from_z(z_series, threshold):
    """Detect anomalies where |z| > threshold."""
    return z_series[np.abs(z_series) > threshold]


def compute_corr_matrix(returns, tickers, window=None):
    """
    Compute correlation matrix for the given tickers.
    If window is provided, use only the last `window` rows of returns.
    """
    df = returns[tickers].dropna()
    if window is not None and len(df) > window:
        df = df.tail(window)
    return df.corr()

def make_corr_heatmap_figure(corr_matrix):
    """
    Create a matplotlib figure with a correlation heatmap.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

    fig.colorbar(cax)

    ticks = range(len(corr_matrix.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="left")
    ax.set_yticklabels(corr_matrix.index)

    ax.set_title("Correlation heatmap", pad=20)

    return fig

def compute_corr_matrix(returns, tickers, window):
    """
    Compute a rolling-window correlation matrix for multiple assets.
    Returns the MOST RECENT correlation matrix.
    """
    corr_matrix = returns.rolling(window).corr()
    # Última fecha disponible
    last_date = corr_matrix.index.get_level_values(0).max()
    # Extraer matriz de correlaciones de la última fecha
    latest_corr = corr_matrix.xs(last_date, level=0)
    return latest_corr

import matplotlib.pyplot as plt
import seaborn as sns

def make_corr_heatmap_figure(corr_matrix):
    """Generate correlation heatmap figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    return fig

# --- New utilities for the Streamlit "pro" app ---

def rolling_correlation_full(returns, a, b, window):
    """
    Compute a full rolling correlation series between two assets.
    Returns a pandas Series indexed by date.
    """
    return returns[a].rolling(window).corr(returns[b])


def prepare_corr_with_z(corr_series):
    """
    Given a correlation series, compute its z-score series and
    return (z_series, mean, std).
    """
    mean_corr = corr_series.mean()
    std_corr = corr_series.std()
    z_series = (corr_series - mean_corr) / std_corr
    return z_series, mean_corr, std_corr


def make_corr_and_z_figure(corr_series, z_series, mean_corr, threshold, pair_name):
    """
    Create a two-panel matplotlib figure:
    - Top: rolling correlation with mean line.
    - Bottom: z-score with +/- threshold and anomalies highlighted.
    """
    import matplotlib.dates as mdates
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Panel 1: correlation
    ax1.plot(corr_series.index, corr_series.values, label="Rolling correlation")
    ax1.axhline(mean_corr, color="gray", linestyle="--", label="Historical mean")
    ax1.set_ylabel("Correlation")
    ax1.set_title(f"Rolling correlation – {pair_name}")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Panel 2: z-score
    ax2.plot(z_series.index, z_series.values, label="Correlation z-score")
    ax2.axhline(threshold, color="red", linestyle="--", label=f"+{threshold} threshold")
    ax2.axhline(-threshold, color="red", linestyle="--", label=f"-{threshold} threshold")

    # highlight anomalies
    anomalies = z_series[np.abs(z_series) > threshold]
    ax2.scatter(anomalies.index, anomalies.values, color="red", s=20, label="Anomalies")

    ax2.set_ylabel("Z-score")
    ax2.set_title("Z-score of rolling correlation")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")

    # nicer x-axis
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    return fig

# --- Clustering utilities ---


def compute_clustered_corr(corr_matrix, method="ward"):
    """
    Given a correlation matrix, compute a hierarchical clustering and
    return the matrix re-ordered according to the cluster leaves.

    method: linkage method, e.g. 'ward', 'average', 'complete'.
    """
    # Distance = 1 - correlation
    dist = 1 - corr_matrix
    dist = dist.fillna(0)

    # Convert to condensed form for linkage
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method=method)

    order = leaves_list(Z)
    labels = corr_matrix.index[order]
    corr_clustered = corr_matrix.loc[labels, labels]

    return corr_clustered, Z


def make_clustered_corr_heatmap_figure(corr_matrix, method="ward"):
    """
    Build a clustered correlation heatmap figure and return
    (figure, clustered_matrix).
    """
    corr_clustered, Z = compute_clustered_corr(corr_matrix, method=method)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr_clustered,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(f"Clustered correlation matrix ({method} linkage)")
    plt.tight_layout()

    return fig, corr_clustered

# --- Cointegration utilities ---


def run_cointegration_analysis(price_a, price_b):
    """
    Run Engle-Granger cointegration test for two price series.

    Parameters
    ----------
    price_a, price_b : pd.Series
        Price series for asset A and B (indexed by date).

    Returns
    -------
    result : dict with keys:
        - 'pvalue'
        - 'score'
        - 'crit_values'
        - 'beta'
        - 'spread' (pd.Series)
        - 'spread_z' (pd.Series)
    """
    # Align the two series on the same dates and drop missing values
    df = pd.concat([price_a, price_b], axis=1, join="inner").dropna()
    y = df.iloc[:, 0]
    x = df.iloc[:, 1]

    # Engle-Granger cointegration test
    score, pvalue, crit_values = coint(y, x)

    # Estimate hedge ratio (beta) via simple linear regression y ~ beta * x
    beta = np.polyfit(x, y, 1)[0]

    # Build the spread and its z-score
    spread = y - beta * x
    spread_mean = spread.mean()
    spread_std = spread.std()
    spread_z = (spread - spread_mean) / spread_std

    return {
        "pvalue": float(pvalue),
        "score": float(score),
        "crit_values": crit_values,
        "beta": float(beta),
        "spread": spread,
        "spread_z": spread_z,
    }


def make_spread_and_z_figure(spread, spread_z, threshold=2.0, pair_name="A-B"):
    """
    Create a figure with:
      - Top panel: cointegration spread
      - Bottom panel: z-score of the spread with ±threshold lines
    """
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # Top: spread
    ax0 = axes[0]
    spread.plot(ax=ax0)
    ax0.set_title(f"Cointegration spread for {pair_name}")
    ax0.set_ylabel("Spread")
    ax0.grid(True, alpha=0.3)

    # Bottom: spread z-score
    ax1 = axes[1]
    spread_z.plot(ax=ax1)
    ax1.axhline(threshold, color="red", linestyle="--", alpha=0.7)
    ax1.axhline(-threshold, color="red", linestyle="--", alpha=0.7)
    ax1.set_title(f"Spread z-score (threshold = ±{threshold:.2f})")
    ax1.set_ylabel("Z-score")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# --- Backtesting utilities for cointegration trading ---


def run_pairs_backtest(price_a, price_b, beta, spread_z, entry_z=2.0, exit_z=0.0):
    """
    Simple pairs trading backtest based on z-score of the cointegration spread.

    Strategy:
        - Enter LONG spread when z < -entry_z  (long A, short B*beta)
        - Enter SHORT spread when z > entry_z  (short A, long B*beta)
        - Exit when z crosses exit_z

    Returns:
        dict with:
            - trades (list of dict)
            - pnl (pd.Series): cumulative PnL
            - daily_pnl (pd.Series)
            - positions (pd.Series): +1 long, -1 short, 0 flat
    """

    df = pd.DataFrame({
        "A": price_a,
        "B": price_b,
        "z": spread_z
    }).dropna()

    position = 0  # +1 long spread, -1 short spread
    trades = []
    daily_pnl = []

    entry_price_a = entry_price_b = None

    for i in range(1, len(df)):
        z_prev = df["z"].iloc[i-1]
        z = df["z"].iloc[i]

        priceA_prev = df["A"].iloc[i-1]
        priceA = df["A"].iloc[i]

        priceB_prev = df["B"].iloc[i-1]
        priceB = df["B"].iloc[i]

        # --- ENTRY LOGIC ---
        if position == 0:
            if z < -entry_z:
                # LONG SPREAD
                position = 1
                entry_price_a = priceA
                entry_price_b = priceB
                trades.append({"type": "long", "entry_idx": i})

            elif z > entry_z:
                # SHORT SPREAD
                position = -1
                entry_price_a = priceA
                entry_price_b = priceB
                trades.append({"type": "short", "entry_idx": i})

        # --- EXIT LOGIC ---
        elif position != 0:
            if (position == 1 and z >= exit_z) or (position == -1 and z <= exit_z):
                trades[-1]["exit_idx"] = i
                trades[-1]["pnl"] = 0  # will compute later
                position = 0

        # --- DAILY PNL CONTRIBUTION ---
        pnl = 0
        if position == 1:
            pnl = (priceA - priceA_prev) - beta * (priceB - priceB_prev)
        elif position == -1:
            pnl = -(priceA - priceA_prev) + beta * (priceB - priceB_prev)

        daily_pnl.append(pnl)

    daily_pnl = pd.Series(daily_pnl, index=df.index[1:])
    pnl_cum = daily_pnl.cumsum()

    for t in trades:
        if "exit_idx" in t:
            idx_entry = df.index[t["entry_idx"]]
            idx_exit = df.index[t["exit_idx"]]
            t["entry_date"] = idx_entry
            t["exit_date"] = idx_exit
            t["holding_days"] = (idx_exit - idx_entry).days
            t["pnl"] = pnl_cum.loc[idx_exit] - pnl_cum.loc[idx_entry]
        else:
            t["exit_date"] = None
            t["pnl"] = None

    return {
        "trades": trades,
        "pnl": pnl_cum,
        "daily_pnl": daily_pnl,
        "df": df,
    }

