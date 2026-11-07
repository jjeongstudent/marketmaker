"""Queue-model checks for the session simulator on a tiny synthetic recording (no network)."""

import numpy as np
import pandas as pd
import pytest

from whiskas.bot.params import Params
from whiskas.bot.sim import Simulator
from whiskas.bot.strategy import Action
from whiskas.live.analyze import LEVEL_DTYPE, Session

WS = 1200  # window start (multiple of 300 s)
T = WS * 1000


def make_session(levels, trades, ticks_ms):
    windows = pd.DataFrame([{"window_start": WS, "slug": f"btc-updown-5m-{WS}", "condition_id": "0x0",
                             "token_up": "U", "token_down": "D"}])
    tokens = ["U", "D"]
    lv = np.array([(ts, tokens.index(tok), side == "BUY", px, sz) for ts, tok, side, px, sz in levels], dtype=LEVEL_DTYPE)
    snaps = [(T + 100, 0, np.array([[0.50, 100.0]]), np.array([[0.52, 100.0]]))]
    s = Session(windows, tokens, lv, snaps)
    s.trades = pd.DataFrame(trades, columns=["ts_ms", "token", "price", "size", "side"]).assign(rt=0, tx="", fee_bps="0")
    s.fills = pd.DataFrame()
    s.chainlink = pd.DataFrame({"ts_ms": ticks_ms, "rt": ticks_ms, "px": [80_000.0] * len(ticks_ms)})
    s.binance = pd.DataFrame()
    return s


def run_with_one_bid(session, price=0.50, size=30.0, token="Up"):
    sim = Simulator(session, Params(latency_ms=150), seed_vol=False)
    st = sim.windows[WS]
    placed = []

    def decide(tau, fv_up, book, inv):
        if placed:
            return []
        placed.append(True)
        return [Action("place", token, price, size)]

    st.strat.decide = decide
    return sim.run(), sim


def test_queue_ahead_then_cleared_level_fills_from_print_remainder():
    levels = [
        (T + 5_000, "U", "BUY", 0.50, 40.0),   # 60 traded in front of us -> queue 100 -> 40
        (T + 6_000, "U", "BUY", 0.50, 30.0),   # 10 cancelled, no print -> queue 40 -> 30 (pro-rata)
        (T + 7_000, "U", "BUY", 0.50, 0.0),    # print of 45 clears the last 30 and 15 more would hit us
    ]
    trades = [(T + 5_000, "U", 0.50, 60.0, "SELL"), (T + 7_000, "U", 0.50, 45.0, "SELL")]
    fills, _ = run_with_one_bid(make_session(levels, trades, [T + 1_000, T + 1_200, T + 8_000]))
    assert len(fills) == 1
    assert fills.iloc[0]["role"] == "maker"
    assert fills.iloc[0]["size"] == pytest.approx(15.0)
    assert fills.iloc[0]["price"] == pytest.approx(0.50)


def test_cancellations_alone_never_fill():
    levels = [(T + 5_000, "U", "BUY", 0.50, 0.0)]  # whole level cancelled, nobody traded
    fills, sim = run_with_one_bid(make_session(levels, [], [T + 1_000, T + 1_200, T + 8_000]))
    assert fills.empty
    assert sim.windows[WS].resting[("Up", 0.50)].queue_ahead == pytest.approx(0.0)


def test_down_token_buy_rests_on_up_offer_side():
    # BUY Down at 0.48 is the same liquidity as an offer on Up at 0.52; an Up buyer sweeping
    # through 0.52 trades against it once the 100 shares ahead of it are gone
    levels = [(T + 5_000, "U", "SELL", 0.52, 0.0)]
    trades = [(T + 5_000, "U", 0.52, 130.0, "BUY")]
    fills, _ = run_with_one_bid(make_session(levels, trades, [T + 1_000, T + 1_200, T + 8_000]),
                                price=0.48, token="Down")
    assert len(fills) == 1
    assert fills.iloc[0]["token"] == "Down"
    assert fills.iloc[0]["size"] == pytest.approx(30.0)
    assert fills.iloc[0]["price"] == pytest.approx(0.48)
