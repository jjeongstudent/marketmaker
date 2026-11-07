"""Strategy parameters. Presets are fitted to the two regimes observed in the wallet's history.

Regime A ("taker-heavy", 2026-06-16 .. 2026-10-26): ~73% of notional taken, price band [0.12, 0.89],
fixed clips of 30-60 shares (50 in September, 60 in October).
Regime B ("maker-heavy", 2026-10-27 ..): ~70% of notional made, band [0.07, 0.94], clips 30/60.

Edges are in price units (0.01 = 1 cent) and are measured against the strategy's own fair value.
They were chosen so that the simulated fills reproduce the real bot's realised edge per share
(maker ~ +1.2c, taker ~ +1.6c gross before fees in regime A); see REPORT.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Params:
    # --- timing (seconds since the window opened)
    quote_start_s: float = 3.0      # no resting quotes in the opening seconds
    quote_stop_s: float = 270.0     # pull resting quotes after this
    take_stop_s: float = 292.0      # no new taker orders after this
    # --- price band for any purchase (token price)
    p_min: float = 0.12
    p_max: float = 0.89
    # --- passive quotes: a ladder of BUYs per outcome token, top level at floor(fv_token - maker_edge)
    #     (joining the touch when the touch is at or below that), then one tick apart
    maker_edge: float = 0.015
    maker_size: float = 50.0        # shares per ladder level
    ladder_levels: int = 1
    join_only: bool = True          # top level never improves the existing best bid (observed live)
    # --- aggressive orders: buy when ask + effective fee + taker_edge <= fv_token
    taker_edge: float = 0.01
    taker_size: float = 50.0
    fee_rate: float = 0.07          # crypto_fees_v2: fee = shares * rate * p * (1-p)
    taker_rebate: float = 0.50      # Obsidian tier taker rebate (the wallet's tier)
    # --- inventory / risk per window
    skew_per_share: float = 0.00002  # fair-value shift per share of Up-minus-Down imbalance
    max_skew: float = 0.03
    max_imbalance: float = 600.0     # stop adding to the long side beyond this many shares
    max_window_cost: float = 3500.0  # stop buying in a window once this much USDC is spent
    tick: float = 0.01
    latency_ms: int = 150            # simulation only: order-to-book delay

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


REGIME_A = Params()
# Live observation (2026-11-07): 30-share BUYs joining the touch on both tokens with further 30-share
# levels one cent apart below, first placed ~6 s into the window and refreshed every few seconds.
REGIME_B = Params(p_min=0.07, p_max=0.94, quote_start_s=6.0, maker_edge=0.01, maker_size=30.0, ladder_levels=4,
                  taker_edge=0.02, taker_size=30.0, quote_stop_s=250.0, max_imbalance=450.0,
                  max_window_cost=2200.0)

PRESETS = {"A": REGIME_A, "B": REGIME_B}
