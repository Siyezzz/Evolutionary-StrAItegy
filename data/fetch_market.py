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


def validate_market_data(data: dict) -> None:
    """Reject malformed histories before they can contaminate a backtest."""
    dates = np.asarray(data["dates"])
    close = np.asarray(data["close"], dtype=float)
    volume = np.asarray(data["volume"], dtype=float)
    if not (len(dates) == len(close) == len(volume)):
        raise ValueError("date, close and volume lengths differ")
    if len(close) < 2:
        raise ValueError("market history needs at least two rows")
    if len(np.unique(dates)) != len(dates):
        raise ValueError("market history contains duplicate dates")
    if np.any(dates[1:] <= dates[:-1]):
        raise ValueError("market dates must be strictly increasing")
    if not np.all(np.isfinite(close)) or np.any(close <= 0):
        raise ValueError("close prices must be finite and positive")
    if not np.all(np.isfinite(volume)) or np.any(volume < 0):
        raise ValueError("volume must be finite and non-negative")


def load_csv(path: str, *, allow_lockbox: bool = False) -> dict:
    if os.path.basename(path).lower().startswith("lockbox_") and not allow_lockbox:
        raise PermissionError(
            "lockbox data is sealed; use the registered one-time evaluator"
        )
    dates, close, volume = [], [], []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        reader.fieldnames = [f.lower() for f in (reader.fieldnames or [])]
        for row in reader:
            dates.append(row["date"])
            close.append(float(row["close"]))
            volume.append(float(row.get("volume", 0.0)))
    data = {
        "dates": np.array(dates),
        "close": np.array(close, dtype=float),
        "volume": np.array(volume, dtype=float),
    }
    validate_market_data(data)
    return data


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
    return load_csv(
        save_path,
        allow_lockbox=os.path.basename(save_path).lower().startswith("lockbox_"),
    )
