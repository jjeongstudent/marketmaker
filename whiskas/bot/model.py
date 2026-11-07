"""Online fair value for the 5-minute BTC Up/Down token.

Feed price ticks (Chainlink BTC/USD from RTDS is the settlement source; Binance is ~1 s faster)
with `on_price`, then ask `p_up(now, window_start)`. Same maths as whiskas.fairvalue, which was
calibrated on ~376k historical (window, second) points.
"""

from __future__ import annotations

import math
from collections import deque

from scipy.special import ndtr

WINDOW = 300


class SecondBars:
    """Last price per wall-clock second, gaps forward-filled, bounded history."""

    def __init__(self, keep_s: int = 3 * 3600):
        self.keep = keep_s
        self.t0: int | None = None  # epoch second of px[0]
        self.px: deque[float] = deque()
        self.cum: deque[float] = deque()  # cum[i] = base + sum(px[0..i])
        self.base = 0.0                   # running sum of seconds already trimmed from the left

    def add(self, ts_ms: int, price: float) -> bool:
        """Record a tick; returns True when a new second was opened."""
        s = int(ts_ms // 1000)
        if self.t0 is None:
            self.t0 = s
            self.px.append(price)
            self.cum.append(price)
            return True
        last = self.t0 + len(self.px) - 1
        if s < last:
            return False  # stale tick
        if s == last:
            self.cum[-1] += price - self.px[-1]
            self.px[-1] = price
            return False
        for _ in range(last + 1, s):  # forward-fill gaps
            self.px.append(self.px[-1])
            self.cum.append(self.cum[-1] + self.px[-1])
        self.px.append(price)
        self.cum.append(self.cum[-1] + price)
        while len(self.px) > self.keep:
            self.px.popleft()
            self.base = self.cum.popleft()
            self.t0 += 1
        return True

    @property
    def last_second(self) -> int | None:
        return None if self.t0 is None else self.t0 + len(self.px) - 1

    def at(self, s: int) -> float | None:
        if self.t0 is None:
            return None
        i = s - self.t0
        if i < 0:
            return None
        return self.px[min(i, len(self.px) - 1)]

    def sum(self, a: int, b: int) -> float | None:
        """Sum of closes over seconds [a, b)."""
        if self.t0 is None or b <= a:
            return 0.0
        ia, ib = a - self.t0, b - self.t0
        if ia < 0 or ib - 1 > len(self.px) - 1:
            return None
        hi = self.cum[ib - 1]
        lo = self.cum[ia - 1] if ia > 0 else self.base
        return hi - lo

    def mean(self, a: int, b: int) -> float | None:
        s = self.sum(a, b)
        return None if s is None else s / (b - a)


class FairValue:
    """P(Up) for a window whose settlement compares the TWAP-L at the end with the TWAP-L at the start
    (L = 0 means spot-vs-spot, the rule used before 2026-08-07)."""

    def __init__(self, twap_s: int = 60, halflife_s: float = 600.0, sigma_mult: float = 1.1,
                 basis_noise_bps: float = 1.0, ret_k: int = 5):
        self.L = twap_s
        self.lam = 0.5 ** (1.0 / halflife_s)
        self.mult = sigma_mult
        self.eta = basis_noise_bps * 1e-4
        self.k = ret_k
        self.halflife = halflife_s
        self.bars = SecondBars()
        self.var = (6e-5) ** 2  # used only until the first return arrives
        self.n_ret = 0          # returns seen; until one half-life of data, var is their plain mean
        self.strikes: dict[int, float] = {}

    # ------------------------------------------------------------------------------- feed
    def on_price(self, ts_ms: int, price: float) -> None:
        if self.bars.add(ts_ms, price):
            s = self.bars.last_second
            p_now, p_then = self.bars.at(s - 1), self.bars.at(s - 1 - self.k)
            if p_now and p_then and s - 1 - self.k >= self.bars.t0:
                r2 = math.log(p_now / p_then) ** 2 / self.k
                self.n_ret += 1
                if self.n_ret < self.halflife:  # warm-up: running mean, no arbitrary prior
                    self.var += (r2 - self.var) / self.n_ret if self.n_ret > 1 else r2 - self.var
                else:
                    self.var = self.lam * self.var + (1 - self.lam) * r2

    def seed_variance(self, per_second_var: float) -> None:
        """Start from a known per-second variance (e.g. from the last 30 min of 1 s klines) instead of
        warming up from scratch."""
        self.var = per_second_var
        self.n_ret = int(self.halflife)

    def set_strike(self, window_start: int, price: float) -> None:
        """Use the official price-to-beat when known (polymarket.com crypto-price `openPrice`)."""
        self.strikes[window_start] = price

    def strike(self, window_start: int) -> float | None:
        if window_start in self.strikes:
            return self.strikes[window_start]
        if self.L == 0:
            return self.bars.at(window_start - 1)
        return self.bars.mean(window_start - self.L, window_start)

    @property
    def sigma(self) -> float:
        return math.sqrt(self.var) * self.mult

    # ---------------------------------------------------------------------------- pricing
    def p_up(self, now: float, window_start: int) -> float | None:
        k = self.strike(window_start)
        s_now = self.bars.at(int(now))
        if k is None or s_now is None:
            return None
        t = now - window_start
        T, L = WINDOW, self.L
        sig2 = (self.sigma * s_now) ** 2
        if L == 0:
            var = sig2 * max(T - t, 0.0)
            mean = s_now
        elif t <= T - L:
            var = sig2 * (max(T - L - t, 0.0) + L / 3.0)
            mean = s_now
        else:
            a = window_start + T - L
            observed = self.bars.sum(a, int(now)) or 0.0  # closes in [T-L, now)
            rem = max(T - (int(now) - window_start), 0)
            var = sig2 * rem ** 3 / (3.0 * L * L)
            mean = (observed + rem * s_now) / L
        var += (self.eta * s_now) ** 2
        return float(ndtr((mean - k) / math.sqrt(var)))
