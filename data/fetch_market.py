"""Market data loading and (optional) live fetching.

``load_csv`` reads a daily OHLCV CSV with columns ``date,close,volume``. That is
the only hard dependency for running the project.

``fetch_yfinance`` is optional: if you have network access and ``yfinance``
installed it downloads real daily data (e.g. BTC-USD) and writes the same CSV
format, so the whole pipeline can run on genuine market history.
"""

from __future__ import annotations

import csv
import os

import numpy as np


def load_csv(path: str) -> dict:
    dates, close, volume = [], [], []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        reader.fieldnames = [f.lower() for f in (reader.fieldnames or [])]
        for row in reader:
            dates.append(row["date"])
            close.append(float(row["close"]))
            volume.append(float(row.get("volume", 0.0)))
    return {
        "dates": np.array(dates),
        "close": np.array(close, dtype=float),
        "volume": np.array(volume, dtype=float),
    }


def fetch_yfinance(
    symbol: str = "BTC-USD",
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    save_path: str = "data/market_btcusd_daily.csv",
) -> dict:
    """Download real daily data via yfinance and save it as the project CSV."""
    import yfinance as yf

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=True)
    # yfinance returns a MultiIndex column (field, ticker); flatten to field names
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Close", "Volume"]].dropna()
    df.to_csv(save_path, index_label="date")
    return load_csv(save_path)
