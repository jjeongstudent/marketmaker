"""Decision logic of the reconstructed strategy (pure: no I/O, no clock).

The market is one shared book per window (BUY Down at q == SELL Up at 1-q). The strategy only ever
BUYs outcome tokens, which makes it a two-sided quoter on that book:
    BUY Up  at b  -> a bid on Up
    BUY Down at q -> an offer on Up at 1-q
Matched Up+Down pairs are held to settlement (worth exactly $1), never merged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .params import Params

TOKENS = ("Up", "Down")


@dataclass
class Book:
    """Up-token view of the window's book: {price: size}."""

    bids: dict = field(default_factory=dict)
    asks: dict = field(default_factory=dict)

    def offers(self, token: str) -> list[tuple[float, float]]:
        """Liquidity a buyer of `token` can lift, best first, in that token's price."""
        if token == "Up":
            return sorted(self.asks.items())
        return sorted(((round(1 - p, 4), s) for p, s in self.bids.items()))

    def best_bid(self, token: str) -> float | None:
        """Best resting buy for `token` (what a passive buyer competes with)."""
        if token == "Up":
            return max(self.bids) if self.bids else None
        return round(1 - min(self.asks), 4) if self.asks else None

    def best_offer(self, token: str) -> float | None:
        o = self.offers(token)
        return o[0][0] if o else None


@dataclass
class Inventory:
    q: dict = field(default_factory=lambda: {"Up": 0.0, "Down": 0.0})
    cost: dict = field(default_factory=lambda: {"Up": 0.0, "Down": 0.0})
    fees: float = 0.0

    def add(self, token: str, size: float, price: float, fee: float = 0.0) -> None:
        self.q[token] += size
        self.cost[token] += size * price
        self.fees += fee

    @property
    def imbalance(self) -> float:
        return self.q["Up"] - self.q["Down"]

    @property
    def spent(self) -> float:
        return self.cost["Up"] + self.cost["Down"] + self.fees

    def pnl(self, winner: str) -> float:
        return self.q[winner] - self.spent


@dataclass(frozen=True)
class Action:
    kind: str            # "take" | "place" | "cancel"
    token: str
    price: float = 0.0
    size: float = 0.0


def floor_tick(x: float, tick: float) -> float:
    return round(math.floor(x / tick + 1e-9) * tick, 4)


class Strategy:
    def __init__(self, params: Params):
        self.p = params
        self.resting: dict[tuple[str, float], float] = {}  # (token, price) -> size of our resting BUY

    def effective_fee(self, price: float) -> float:
        return self.p.fee_rate * price * (1 - price) * (1 - self.p.taker_rebate)

    def _can_buy(self, token: str, inv: Inventory, size: float, price: float) -> bool:
        if inv.spent + size * price > self.p.max_window_cost:
            return False
        sign = 1 if token == "Up" else -1
        after = inv.imbalance + sign * size
        return abs(after) <= self.p.max_imbalance or abs(after) < abs(inv.imbalance)

    def ladder(self, tok: str, fv_tok: float, book: Book, inv: Inventory) -> list[float]:
        p = self.p
        top = floor_tick(fv_tok - p.maker_edge, p.tick)
        bid = book.best_bid(tok)
        if p.join_only and bid is not None:
            top = min(top, bid)  # join the touch, never step in front of it
        offer = book.best_offer(tok)
        if offer is not None and top >= offer:
            top = round(offer - p.tick, 4)  # post-only: never cross
        levels = [round(top - i * p.tick, 4) for i in range(p.ladder_levels)]
        return [x for x in levels if p.p_min <= x <= p.p_max and self._can_buy(tok, inv, p.maker_size, x)]

    def decide(self, tau: float, fv_up: float | None, book: Book, inv: Inventory) -> list[Action]:
        p = self.p
        acts: list[Action] = []
        if fv_up is None or tau < 0 or tau > p.take_stop_s:
            return [Action("cancel", t, px) for t, px in list(self.resting)]

        if p.model_weight < 1.0 and book.bids and book.asks:
            mid = (max(book.bids) + min(book.asks)) / 2
            fv_up = p.model_weight * fv_up + (1 - p.model_weight) * mid
        skew = max(-p.max_skew, min(p.max_skew, inv.imbalance * p.skew_per_share))
        fv = {"Up": fv_up - skew, "Down": 1.0 - fv_up + skew}

        # 1) aggressive: lift offers that are cheap after fees
        for tok in TOKENS:
            best = book.best_offer(tok)
            if best is None or not (p.p_min <= best <= p.p_max):
                continue
            if best + self.effective_fee(best) + p.taker_edge <= fv[tok] and self._can_buy(tok, inv, p.taker_size, best):
                limit = best
                while limit + p.tick <= p.p_max and \
                        (limit + p.tick) + self.effective_fee(limit + p.tick) + p.taker_edge <= fv[tok]:
                    limit = round(limit + p.tick, 4)
                acts.append(Action("take", tok, limit, p.taker_size))

        # 2) passive: keep a ladder of resting BUYs per token; cancel levels that are no longer wanted
        quoting = p.quote_start_s <= tau <= p.quote_stop_s
        for tok in TOKENS:
            want = set(self.ladder(tok, fv[tok], book, inv)) if quoting else set()
            have = {px for (t, px) in self.resting if t == tok}
            for px in sorted(have - want):
                acts.append(Action("cancel", tok, px))
            for px in sorted(want - have, reverse=True):
                acts.append(Action("place", tok, px, p.maker_size))
        return acts
