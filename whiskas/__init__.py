"""Reverse-engineering toolkit for the x-MoneyForWhiskas Polymarket bot.

Subpackages / modules:
  config      constants (wallet, fee schedule, market geometry)
  http        retrying HTTP session
  polymarket  Data API v2, Gamma and CLOB read clients
  binance     BTCUSDT 1-second klines
  collect     CLI that downloads the full dataset into data/
  fairvalue   TWAP-aware fair value for 5-minute Up/Down markets
  features    joins fills with market metadata and BTC prices
"""

__all__ = ["config"]
