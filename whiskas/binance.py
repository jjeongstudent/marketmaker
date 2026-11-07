"""Binance BTCUSDT 1-second klines (public market-data mirror)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BINANCE_API
from .http import get_json

COLS = ["open_ms", "open", "high", "low", "close", "volume", "close_ms", "quote_volume",
        "n_trades", "taker_buy_base", "taker_buy_quote", "_ignore"]


def klines_1s(start_s: int, end_s: int, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """1-second klines covering [start_s, end_s) in epoch seconds. Gaps (no trades) are absent."""
    out = []
    t = start_s * 1000
    end_ms = end_s * 1000
    while t < end_ms:
        rows = get_json(f"{BINANCE_API}/api/v3/klines",
                        {"symbol": symbol, "interval": "1s", "startTime": t, "endTime": end_ms - 1, "limit": 1000})
        if not rows:
            break
        out.extend(rows)
        t = rows[-1][0] + 1000
    if not out:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "n_trades"])
    df = pd.DataFrame(out, columns=COLS)
    df["ts"] = (df["open_ms"] // 1000).astype(np.int64)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["n_trades"] = df["n_trades"].astype(np.int64)
    return df[["ts", "open", "high", "low", "close", "volume", "n_trades"]].drop_duplicates("ts")
