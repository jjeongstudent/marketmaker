"""Fair value of the 5-minute BTC Up/Down token.

Window [0, T] with T = 300 s. Settlement (Gamma: cryptoMarketConfigId `btc-5m-twap-60`):

    Up  iff  A >= K
    A = TWAP of BTC/USD over the last L = 60 s of the window      (Chainlink TWAP-60 stream at T)
    K = reference price at the start of the window                (stream value at 0)

We model log-price as driftless Brownian motion with per-second volatility sigma and use the
arithmetic (small-move) approximation, which is exact to O(sigma^2 T) ~ 1e-6 here.

At time t with spot S_t and, inside the averaging window, the running integral I_t of
S over [T-L, t]:

    t <= T-L:  E[A] = S_t,                              Var[A] = sigma^2 S^2 [(T-L-t) + L/3]
    t >  T-L:  E[A] = (I_t + (T-t) S_t) / L,            Var[A] = sigma^2 S^2 (T-t)^3 / (3 L^2)

    where I_t is the sum of the 1 s closes already observed in [T-L, t) (t - (T-L) of them), so
    I_t plus the (T-t) remaining seconds valued at S_t covers exactly L seconds.

    P(Up) = Phi((E[A] - K) / sd[A])

`p_up_spot` is the plain digital (A = S_T), the rule before 2026-08-07.

`basis_noise_bps` adds independent noise between the pricing feed (Binance) and the settlement feed
(Chainlink) to Var[A - K]; ~1 bp fits the historical outcomes best.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

T_WINDOW = 300.0
L_TWAP = 60.0


def p_up_twap(t, spot, strike, sigma, running_integral=None, T=T_WINDOW, L=L_TWAP, basis_noise_bps=0.0):
    """Probability that the end-of-window TWAP finishes >= strike (vectorised).

    t: seconds since window start (may be negative: pre-window trading).
    spot, strike: prices. sigma: per-second log-vol. running_integral: integral of spot over
    [T-L, t] (only used when t > T-L; pass the sum of 1 s closes).
    """
    t = np.asarray(t, dtype=float)
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    pre = t <= T - L
    tt = np.clip(t, None, T)
    var_pre = (np.maximum(T - L - tt, 0.0) + L / 3.0)
    rem = np.clip(T - tt, 0.0, L)
    var_in = rem ** 3 / (3.0 * L * L)
    var = np.where(pre, var_pre, var_in) * (sigma * spot) ** 2 + (basis_noise_bps * 1e-4 * spot) ** 2
    if running_integral is None:
        running_integral = np.where(pre, 0.0, (tt - (T - L)) * spot)
    running_integral = np.asarray(running_integral, dtype=float)
    mean = np.where(pre, spot, (running_integral + rem * spot) / L)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (mean - strike) / np.sqrt(var)
    z = np.where(var > 0, z, np.where(mean >= strike, np.inf, -np.inf))
    return ndtr(z)


def p_up_spot(t, spot, strike, sigma, T=T_WINDOW, basis_noise_bps=0.0):
    """Plain digital: P(S_T >= strike) under driftless BM (vectorised)."""
    t = np.asarray(t, dtype=float)
    spot = np.asarray(spot, dtype=float)
    rem = np.maximum(T - t, 0.0)
    sd = np.sqrt((np.asarray(sigma) * spot) ** 2 * rem + (basis_noise_bps * 1e-4 * spot) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (spot - np.asarray(strike)) / sd
    z = np.where(sd > 0, z, np.where(spot >= strike, np.inf, -np.inf))
    return ndtr(z)


def ewma_vol(log_ret: np.ndarray, halflife_s: float = 300.0, floor: float = 2e-5) -> np.ndarray:
    """EWMA per-second volatility of a 1 s log-return series, using only past data (no look-ahead)."""
    from scipy.signal import lfilter

    lam = 0.5 ** (1.0 / halflife_s)
    r2 = np.nan_to_num(np.asarray(log_ret, dtype=float), nan=0.0) ** 2
    if not len(r2):
        return r2
    v0 = np.mean(r2[: int(halflife_s)])
    y, _ = lfilter([1 - lam], [1, -lam], r2, zi=[lam * v0])
    out = np.concatenate([[v0], y[:-1]])  # value known before observing r2[i]
    return np.maximum(np.sqrt(out), floor)
