"""Constants describing the target wallet and the market it trades."""

from __future__ import annotations

from pathlib import Path

# polymarket.com/@x-moneyforwhiskas  (display name "x-MoneyForWhiskas", pseudonym "Flashy-Gold")
WALLET = "0x3048d65321be3497164cdfc2996f94f98a2e7537"
USERNAME = "x-MoneyForWhiskas"

DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
BINANCE_API = "https://data-api.binance.vision"  # api.binance.com is geo-blocked (HTTP 451) from US egress

# 5-minute BTC Up/Down markets ("btc-updown-5m-<window start epoch seconds>").
SERIES_SLUG = "btc-up-or-down-5m"
SLUG_PREFIX = "btc-updown-5m-"
WINDOW_SECONDS = 300
# Resolution: Chainlink BTC/USD TWAP data stream with a 60 s lookback (cryptoMarketConfigId btc-5m-twap-60).
TWAP_SECONDS = 60

# Fee schedule reported by Gamma for these markets (feeType crypto_fees_v2).
# fee = shares * TAKER_FEE_RATE * p * (1 - p); makers pay nothing.
TAKER_FEE_RATE = 0.07
MAKER_REBATE_SHARE = 0.20  # share of taker fees in a market redistributed to its makers, fee-curve weighted
TICK = 0.01
MIN_ORDER_SIZE = 5

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "reports" / "figures"


def taker_fee(shares, price):
    """Taker fee in USDC for a fill of `shares` at `price` (vectorised)."""
    return shares * TAKER_FEE_RATE * price * (1.0 - price)
