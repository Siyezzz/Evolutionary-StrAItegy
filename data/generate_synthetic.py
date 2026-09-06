"""Synthetic market generator (offline fallback).

Produces a realistic-looking daily price/volume series with random regime
switches (drift + volatility), so the project is runnable with zero external
dependencies. It is a *stand-in*; replace ``data/market_btcusd_daily.csv`` with
real history via ``fetch_yfinance`` for genuine backtests.

Output format matches ``fetch_market.load_csv``: columns ``date,close,volume``.
"""

from __future__ import annotations

import datetime
import os

import numpy as np


def generate(
    path: str = "data/market_btcusd_daily.csv",
    start: str = "2018-01-01",
    days: int = 2557,
    seed: int = 7,
) -> int:
    rng = np.random.default_rng(seed)
    price = 1000.0
    d = datetime.date.fromisoformat(start)
    dates, closes, vols = [], [], []

    drift, vol = 0.0005, 0.02
    for _ in range(days):
        if rng.random() < 0.01:  # occasional regime switch
            drift = float(rng.normal(0.0003, 0.002))
            vol = float(rng.choice([0.015, 0.03, 0.05, 0.08]))
        r = rng.normal(drift, vol)
        price *= 1.0 + r
        closes.append(price)
        vols.append(float(rng.gamma(2.0, 2000.0)))
        dates.append(d.isoformat())
        d += datetime.timedelta(days=1)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write("date,close,volume\n")
        for dt, c, v in zip(dates, closes, vols):
            fh.write(f"{dt},{c:.2f},{v:.2f}\n")
    return len(closes)
