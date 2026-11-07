import math

import numpy as np
import pytest

from whiskas.bot.model import FairValue, SecondBars
from whiskas.bot.params import Params
from whiskas.bot.strategy import Book, Inventory, Strategy
from whiskas.config import taker_fee
from whiskas.fairvalue import ewma_vol, p_up_spot, p_up_twap


def synthetic_path(n=8000, s0=80_000.0, sig=7e-5, seed=3):
    rng = np.random.default_rng(seed)
    return s0 * np.exp(np.cumsum(rng.normal(0, sig, n)))


def test_secondbars_sums_survive_trimming():
    b = SecondBars(keep_s=50)
    for i in range(200):
        b.add(i * 1000 + 500, float(i))
    # only the last 50 seconds are kept; sums over them must be exact
    assert b.t0 == 150
    assert b.sum(150, 160) == pytest.approx(sum(range(150, 160)))
    assert b.mean(190, 200) == pytest.approx(np.mean(range(190, 200)))
    assert b.sum(100, 160) is None  # trimmed history is not invented


def test_secondbars_forward_fills_gaps():
    b = SecondBars()
    b.add(1_000, 10.0)
    b.add(4_000, 13.0)
    assert [b.at(s) for s in (1, 2, 3, 4)] == [10.0, 10.0, 10.0, 13.0]


@pytest.mark.parametrize("tau", [10, 120, 239, 241, 270, 295])
def test_online_fair_value_matches_batch_model(tau):
    path = synthetic_path()
    ws = 7_200  # window start (epoch seconds of the synthetic clock)
    # re-level the in-window path so it sits near the strike late in the window (non-trivial p)
    path[ws:] *= path[ws - 60: ws].mean() / path[ws + 240: ws + 300].mean()
    fv = FairValue(twap_s=60, halflife_s=600.0, sigma_mult=1.1, basis_noise_bps=1.0)
    now = ws + tau
    for s in range(0, now + 1):
        fv.on_price(s * 1000 + 999, path[s])
    online = fv.p_up(now, ws)  # same whole-second clock as the batch model

    # batch: same estimator, vectorised (whiskas.fairvalue / features.BTC.sigma_series)
    lc = np.log(path)
    k = 5
    rk = np.concatenate([np.full(k, np.nan), lc[k:] - lc[:-k]]) / math.sqrt(k)
    sigma = ewma_vol(rk, halflife_s=600.0)[now] * 1.1
    strike = path[ws - 60: ws].mean()
    integ = path[ws + 240: now].sum() if now > ws + 240 else 0.0
    batch = float(p_up_twap(tau, path[now], strike, sigma, running_integral=integ, basis_noise_bps=1.0))
    assert 0.01 < batch < 0.99  # a non-trivial comparison
    assert online == pytest.approx(batch, abs=1e-3)


def test_twap_lock_in_and_spot_limits():
    # deep inside the averaging window with the average far above the strike -> ~certain Up
    p = p_up_twap(295, 101.0, 100.0, 7e-5, running_integral=55 * 101.0)
    assert p > 0.999
    # before the window: symmetric at the money
    assert p_up_twap(0, 100.0, 100.0, 7e-5) == pytest.approx(0.5)
    assert p_up_spot(0, 100.0, 100.0, 7e-5) == pytest.approx(0.5)


def test_fee_formula_matches_published_table():
    # docs.polymarket.com/trading/fees: crypto, 100 shares at 0.50 -> $1.75, at 0.10 -> $0.63
    assert taker_fee(100, 0.50) == pytest.approx(1.75)
    assert round(taker_fee(100, 0.10), 2) == 0.63


def test_book_mirroring_offers():
    b = Book(bids={0.48: 100, 0.47: 50}, asks={0.50: 80, 0.51: 20})
    assert b.offers("Up")[0] == (0.50, 80)
    assert b.offers("Down")[0] == (0.52, 100)  # BUY Down lifts the best Up bid at 1 - 0.48
    assert b.best_bid("Down") == 0.50          # best resting BUY Down = 1 - best Up ask


def test_strategy_takes_only_cheap_offers_and_quotes_both_sides():
    p = Params(taker_edge=0.01, maker_edge=0.02, taker_rebate=0.5)
    s = Strategy(p)
    book = Book(bids={0.40: 100}, asks={0.45: 100})
    acts = s.decide(tau=60, fv_up=0.50, book=book, inv=Inventory())
    takes = [a for a in acts if a.kind == "take"]
    assert [a.token for a in takes] == ["Up"]        # Up offered at 0.45 vs fv 0.50 -> take
    assert takes[0].price >= 0.45
    places = {a.token: a.price for a in acts if a.kind == "place"}
    assert places["Up"] == pytest.approx(0.40)       # joins the existing 0.40 bid, never steps in front
    assert places["Down"] == pytest.approx(0.48)     # floor(0.50 - 0.02), below the 0.55 Down bid
    improving = Strategy(p.with_(join_only=False)).decide(tau=60, fv_up=0.50, book=book, inv=Inventory())
    assert {a.token: a.price for a in improving if a.kind == "place"}["Up"] == pytest.approx(0.44)  # below the offer


def test_strategy_respects_price_band_and_time():
    p = Params(p_max=0.89)
    s = Strategy(p)
    book = Book(bids={0.05: 100}, asks={0.92: 100})
    acts = s.decide(tau=200, fv_up=0.97, book=book, inv=Inventory())
    assert not [a for a in acts if a.kind == "take"]  # 0.92 is above the 0.89 cap
    assert s.decide(tau=299, fv_up=0.5, book=book, inv=Inventory()) == []


def test_ladder_places_levels_one_tick_apart_and_cancels_stale_ones():
    p = Params(maker_edge=0.01, ladder_levels=3, maker_size=30, taker_edge=0.5)
    s = Strategy(p)
    book = Book(bids={0.49: 100}, asks={0.60: 100})
    acts = s.decide(tau=30, fv_up=0.50, book=book, inv=Inventory())
    ups = sorted(a.price for a in acts if a.kind == "place" and a.token == "Up")
    assert ups == [0.47, 0.48, 0.49]
    for a in acts:
        if a.kind == "place":
            s.resting[(a.token, a.price)] = a.size
    # fair value moves down one cent: the 0.49 Up level is cancelled, 0.46 added, the rest kept
    acts = s.decide(tau=31, fv_up=0.49, book=book, inv=Inventory())
    assert {(a.kind, a.price) for a in acts if a.token == "Up"} == {("cancel", 0.49), ("place", 0.46)}
