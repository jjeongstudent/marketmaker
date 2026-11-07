"""Strategy parameters, with presets for the two regimes observed in the wallet's history.

Regime A ("taker-heavy", 2026-06-16 .. 2026-10-26): ~75% of notional taken, price band [0.12, 0.89],
fixed clips of 30-60 shares (50 in September, 60 in October). Band and clip sizes come from the fill
history; the edges are illustrative defaults (no live book data exists for this regime).
Regime B ("maker-heavy", 2026-10-27 ..): ~70% of notional made, band [0.07, 0.94], 30-share clips.
Calibrated against a 75-minute live recording (see REGIME_B below and REPORT.md section 5).

Edges are in price units (0.01 = 1 cent) and are measured against the strategy's decision value.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Params:
    # --- timing (seconds since the window opened)
    quote_start_s: float = 3.0      # no resting quotes in the opening seconds
    quote_stop_s: float = 270.0     # pull resting quotes after this
    take_stop_s: float = 292.0      # no new taker orders after this
    # --- fair value used for decisions: w * model + (1 - w) * book mid (the bot quotes around the market)
    model_weight: float = 1.0
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
# Live observation (2026-11-07, 17 windows): 30-share BUYs joining the touch on both tokens (81% of
# maker fills at the token's best bid, median order age 5 s) and 30-share taker clips. Calibrated with
# whiskas.bot.sim.calibrate against the wallet's fills in the 14 fully recorded windows (54-point grid):
# decision value = 80% book mid + 20% model; one level at min(best bid, value - 3c); takes need 6c of
# edge after fees. Fit: maker shares 3,667 sim vs 2,229 real, taker 362 vs 491, realised edges within ~2c.
REGIME_B = Params(p_min=0.07, p_max=0.94, quote_start_s=6.0, quote_stop_s=250.0, model_weight=0.2,
                  maker_edge=0.03, maker_size=30.0, ladder_levels=1, taker_edge=0.06, taker_size=30.0,
                  max_imbalance=450.0, max_window_cost=2200.0)

PRESETS = {"A": REGIME_A, "B": REGIME_B}
