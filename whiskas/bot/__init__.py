"""Reference re-implementation of the x-MoneyForWhiskas strategy.

model     online TWAP-aware fair value (Chainlink or Binance ticks)
params    strategy parameters with presets fitted to each regime of the real bot
strategy  pure decision function: book + fair value + inventory -> orders
sim       replay of recorded live sessions (whiskas.live.record) through the strategy
"""
