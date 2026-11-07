"""Build the enriched fill table: role labels, window geometry, BTC state and fair value.

    from whiskas import features
    fills = features.load_fills()          # every fill, labelled maker/taker
    btc = features.BTC.load()              # dense 1 s Binance BTCUSDT series
    enriched = features.enrich(fills, btc) # adds tau, strike, fair value, edge, markouts
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from .config import DATA, TAKER_FEE_RATE, TWAP_SECONDS, WINDOW_SECONDS
from .fairvalue import ewma_vol, p_up_spot, p_up_twap

KEY = ["transaction_hash", "token_id", "side", "size", "price"]


def _read_dir(name: str) -> pd.DataFrame:
    """Concatenate per-day parquet files; a day's .partial file is used only if no full file exists."""
    files = sorted((DATA / name).glob("*.parquet"))
    full = {f.name.removesuffix(".parquet") for f in files if not f.name.endswith(".partial.parquet")}
    use = [f for f in files
           if not f.name.endswith(".partial.parquet") or f.name.removesuffix(".partial.parquet") not in full]
    return pd.concat([pd.read_parquet(f) for f in use], ignore_index=True)


def load_fills() -> pd.DataFrame:
    """All fills with `role` in {maker, taker}. Rows are from the wallet's perspective."""
    allf = _read_dir("trades_all")
    tkr = _read_dir("trades_taker")
    for d in (allf, tkr):
        d["size"] = d["size"].round(6)
        d["price"] = d["price"].round(10)
        d["_n"] = d.groupby(KEY).cumcount()
    tkr = tkr[KEY + ["_n"]].assign(role="taker")
    df = allf.merge(tkr, on=KEY + ["_n"], how="left")
    df["role"] = df["role"].fillna("maker")
    df = df.drop(columns="_n")
    df["asset"] = df["slug"].str.extract(r"^([a-z]+)-updown-5m-")[0]
    ws = pd.to_numeric(df["slug"].str.extract(r"-updown-5m-(\d+)$")[0], errors="coerce")
    df["window_start"] = ws.astype("Int64")
    df["tau"] = df["timestamp"] - df["window_start"]
    df["notional"] = df["size"] * df["price"]
    df["fee"] = np.where(df["role"] == "taker", df["size"] * TAKER_FEE_RATE * df["price"] * (1 - df["price"]), 0.0)
    df["fee_equiv"] = df["size"] * TAKER_FEE_RATE * df["price"] * (1 - df["price"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.sort_values(["timestamp", "transaction_hash"]).reset_index(drop=True)


def load_markets() -> pd.DataFrame:
    m = pd.read_parquet(DATA / "markets.parquet")
    for c in ("window_start", "window_end", "created_at", "accepting_orders_ts"):
        m[c] = pd.to_datetime(m[c], utc=True, format="ISO8601")
    # pandas >= 3 keeps the parsed resolution (often "us"), so convert explicitly to epoch seconds.
    m["ws"] = (m["window_start"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
    m["era"] = m["crypto_config"].fillna("spot").replace({"btc-5m": "spot"})
    return m


@dataclass
class BTC:
    """Dense per-second Binance BTCUSDT close series (gaps forward-filled)."""

    t0: int
    close: np.ndarray

    @classmethod
    def load(cls) -> "BTC":
        files = sorted((DATA / "klines_1s").glob("*.parquet"))
        k = pd.concat([pd.read_parquet(f, columns=["ts", "close"]) for f in files], ignore_index=True)
        k = k.drop_duplicates("ts").sort_values("ts")
        t0, t1 = int(k["ts"].iloc[0]), int(k["ts"].iloc[-1])
        dense = pd.Series(np.nan, index=np.arange(t0, t1 + 1))
        dense.loc[k["ts"].to_numpy()] = k["close"].to_numpy()
        return cls(t0, dense.ffill().to_numpy())

    @property
    def t1(self) -> int:
        return self.t0 + len(self.close) - 1

    def at(self, ts) -> np.ndarray:
        """Close of the 1 s bar starting at ts (i.e. last trade price by ts+1)."""
        idx = np.asarray(ts, dtype=np.int64) - self.t0
        out = np.full(idx.shape, np.nan)
        ok = (idx >= 0) & (idx < len(self.close))
        out[ok] = self.close[idx[ok]]
        return out

    @cached_property
    def cumsum(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(self.close)])

    def mean(self, a, b) -> np.ndarray:
        """Mean close over seconds [a, b)."""
        a = np.asarray(a, dtype=np.int64) - self.t0
        b = np.asarray(b, dtype=np.int64) - self.t0
        ok = (a >= 0) & (b <= len(self.close)) & (b > a)
        out = np.full(a.shape, np.nan)
        out[ok] = (self.cumsum[b[ok]] - self.cumsum[a[ok]]) / (b[ok] - a[ok])
        return out

    def integral(self, a, b) -> np.ndarray:
        """Sum of closes over seconds [a, b) (b >= a)."""
        a = np.asarray(a, dtype=np.int64) - self.t0
        b = np.asarray(b, dtype=np.int64) - self.t0
        a = np.clip(a, 0, len(self.close)); b = np.clip(b, 0, len(self.close))
        return self.cumsum[np.maximum(a, b)] - self.cumsum[a]

    def sigma_series(self, halflife_s: float = 600.0, k: int = 5) -> np.ndarray:
        """Per-second EWMA vol from overlapping k-second log returns (less 1 s bid/ask bounce), past-only."""
        lc = np.log(self.close)
        rk = np.concatenate([np.full(k, np.nan), lc[k:] - lc[:-k]]) / np.sqrt(k)
        return ewma_vol(rk, halflife_s=halflife_s)

    @cached_property
    def sigma(self) -> np.ndarray:
        return self.sigma_series()

    def sigma_at(self, ts) -> np.ndarray:
        idx = np.clip(np.asarray(ts, dtype=np.int64) - self.t0, 0, len(self.close) - 1)
        return self.sigma[idx]


# ------------------------------------------------------------------------- settlement rules
#
# Settlement eras (Gamma resolutionSource / cryptoMarketConfigId):
#   until 2026-08-06  Chainlink BTC/USD spot          -> L = 0  (Up iff S_end >= S_start)
#   2026-08-07..08-13 Chainlink BTC/USD TWAP-30 stream -> L = 30
#   since 2026-08-14  Chainlink BTC/USD TWAP-60 stream -> L = 60 (Up iff TWAP60(end) >= TWAP60(start))
# Chainlink trails Binance by ~1 s (best outcome match with a 1-2 s shift), modelled as CL_LAG.

ERA_TWAP = {"spot": 0, "btc-5m-twap-30": 30, "btc-5m-twap-60": 60}
CL_LAG = 1
# Calibrated on ~1.1M (window, second) points against realised outcomes: EWMA half-life 600 s on
# overlapping 5 s returns, scaled by 1.1, plus 1 bp Binance-vs-Chainlink basis noise
# -> log-loss 0.448 (0.693 for a coin flip) and a near-diagonal reliability curve.
SIGMA_MULT = 1.1
BASIS_NOISE_BPS = 1.0


def twap_len(era) -> np.ndarray:
    return pd.Series(era).map(ERA_TWAP).fillna(0).astype(int).to_numpy()


def strike(btc: BTC, ws, L) -> np.ndarray:
    """Binance proxy for the window's reference price ("price to beat")."""
    ws = np.asarray(ws, dtype=np.int64) - CL_LAG
    L = np.broadcast_to(np.asarray(L, dtype=np.int64), ws.shape)
    return np.where(L > 0, btc.mean(ws - np.maximum(L, 1), ws), btc.at(ws - 1))


def settle_value(btc: BTC, ws, L) -> np.ndarray:
    end = np.asarray(ws, dtype=np.int64) + WINDOW_SECONDS - CL_LAG
    L = np.broadcast_to(np.asarray(L, dtype=np.int64), end.shape)
    return np.where(L > 0, btc.mean(end - np.maximum(L, 1), end), btc.at(end - 1))


def fair_up(btc: BTC, ts, ws, L, lag: int = 0, sigma_mult: float = SIGMA_MULT, strike_px=None) -> np.ndarray:
    """Model P(Up) at wall-clock second `ts` for the window starting at `ws` with TWAP length L,
    using Binance information up to ts - lag."""
    info = np.asarray(ts, dtype=np.int64) - lag
    ws = np.asarray(ws, dtype=np.int64)
    L = np.broadcast_to(np.asarray(L, dtype=np.int64), ws.shape)
    w0 = ws - CL_LAG                           # window start on the Binance clock
    t = (info - w0).astype(float)              # seconds into the window
    spot = btc.at(info)
    k = strike(btc, ws, L) if strike_px is None else np.asarray(strike_px, dtype=float)
    sig = btc.sigma_at(info) * sigma_mult
    out = np.empty(ws.shape)
    spot_rule = L == 0
    if spot_rule.any():
        out[spot_rule] = p_up_spot(t[spot_rule], spot[spot_rule], k[spot_rule], sig[spot_rule],
                                   basis_noise_bps=BASIS_NOISE_BPS)
    for l in np.unique(L[~spot_rule]):
        sel = L == l
        a = w0[sel] + WINDOW_SECONDS - l       # averaging window [a, end) on the Binance clock
        # closes already observed inside the averaging window: seconds [a, info). The remaining
        # T - t seconds (including the current one) are valued at the current spot by p_up_twap.
        integ = btc.integral(a, np.maximum(info[sel], a))
        out[sel] = p_up_twap(t[sel], spot[sel], k[sel], sig[sel], running_integral=integ, L=float(l),
                             basis_noise_bps=BASIS_NOISE_BPS)
    return out


def enrich(fills: pd.DataFrame, btc: BTC, markets: pd.DataFrame | None = None, lag: int = 0,
           markouts=(1, 2, 5, 10, 30, 60), sigma_mult: float = SIGMA_MULT) -> pd.DataFrame:
    """Add strike/spot/fair value/edge/markout columns to BTC fills inside the BTC series range."""
    if markets is None:
        markets = load_markets()
    f = fills[(fills["asset"] == "btc") & fills["window_start"].notna()].copy()
    f = f.merge(markets[["slug", "era", "winner"]], on="slug", how="left")
    f = f[(f["timestamp"] - 70 > btc.t0) & (f["window_start"] + WINDOW_SECONDS + 70 < btc.t1)]
    ts = f["timestamp"].to_numpy(np.int64)
    ws = f["window_start"].to_numpy(np.int64)
    L = twap_len(f["era"])
    f["twap_len"] = L
    is_up = (f["outcome_index"].to_numpy() == 0)
    f["strike"] = strike(btc, ws, L)
    f["spot"] = btc.at(ts - lag)
    f["moneyness_bps"] = (f["spot"] / f["strike"] - 1) * 1e4
    f["sigma"] = btc.sigma_at(ts - lag)
    fv_up = fair_up(btc, ts, ws, L, lag=lag, sigma_mult=sigma_mult)
    f["fv_up"] = fv_up
    f["fv_tok"] = np.where(is_up, fv_up, 1 - fv_up)
    f["edge"] = f["fv_tok"] - f["price"]
    end = ws + WINDOW_SECONDS - 1
    for h in markouts:
        fv_h = fair_up(btc, np.minimum(ts + h, end), ws, L, sigma_mult=sigma_mult)
        f[f"mo_{h}s"] = np.where(is_up, fv_h, 1 - fv_h) - f["price"]
    f["won"] = np.where(is_up, f["winner"] == "Up", f["winner"] == "Down")
    f["payoff_edge"] = f["won"].astype(float) - f["price"]
    return f


# ----------------------------------------------------------------------------- derived tables

SPLIT_TS = 1793059200  # 2026-10-27 00:00 UTC: switch from taker-heavy (A) to maker-heavy (B)


def inventory_path(e: pd.DataFrame) -> pd.DataFrame:
    """Per-window inventory before each fill and whether the fill opens, adds to, or completes a pair."""
    e = e.sort_values(["slug", "timestamp"], kind="stable").reset_index(drop=True)
    up = (e["outcome_index"] == 0).to_numpy()
    cols = {"q_up": np.where(up, e["size"], 0.0), "q_dn": np.where(~up, e["size"], 0.0),
            "c_up": np.where(up, e["notional"], 0.0), "c_dn": np.where(~up, e["notional"], 0.0)}
    for name, arr in cols.items():
        x = pd.Series(arr, index=e.index)
        e[name + "_before"] = x.groupby(e["slug"], sort=False).cumsum() - x
    e["q_this_before"] = np.where(up, e["q_up_before"], e["q_dn_before"])
    e["q_other_before"] = np.where(up, e["q_dn_before"], e["q_up_before"])
    e["avg_other"] = np.where(up, e["c_dn_before"] / e["q_dn_before"].replace(0, np.nan),
                              e["c_up_before"] / e["q_up_before"].replace(0, np.nan))
    e["imb_before"] = e["q_this_before"] - e["q_other_before"]
    e["kind"] = np.select([(e["q_this_before"] + e["q_other_before"] == 0).to_numpy(), (e["imb_before"] < 0).to_numpy()],
                          ["first", "completing"], default="adding")
    e["implied_pair_cost"] = e["price"] + e["avg_other"]
    return e


def per_market(fills: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """One row per BTC window: shares/cost per side, fees, payout, PnL split into the riskless pair part
    (min(Up, Down) * (1 - avg Up - avg Down)) and the unpaired residual."""
    f = fills[fills["asset"] == "btc"].merge(markets[["slug", "winner", "crypto_config"]], on="slug")
    f["up"] = f["outcome_index"] == 0
    g = f.groupby(["slug", "up"]).agg(q=("size", "sum"), c=("notional", "sum"), fee=("fee", "sum")).unstack("up").fillna(0)
    g.columns = [f"{a}_{'up' if b else 'dn'}" for a, b in g.columns]
    g = g.join(markets.set_index("slug")[["winner", "crypto_config"]])
    g["t"] = g.index.str.rsplit("-", n=1).str[1].astype(int)
    g["dt"] = pd.to_datetime(g["t"], unit="s", utc=True)
    g["era"] = np.where(g["t"] < SPLIT_TS, "A", "B")
    g["payout"] = np.where(g["winner"] == "Up", g["q_up"], g["q_dn"])
    g["cost"] = g["c_up"] + g["c_dn"]
    g["fees"] = g["fee_up"] + g["fee_dn"]
    g["pnl"] = g["payout"] - g["cost"] - g["fees"]
    g["pair_q"] = np.minimum(g["q_up"], g["q_dn"])
    g["pu"] = g["c_up"] / g["q_up"].replace(0, np.nan)
    g["pd"] = g["c_dn"] / g["q_dn"].replace(0, np.nan)
    g["pair_cost"] = g["pu"] + g["pd"]
    g["pair_pnl"] = (g["pair_q"] * (1 - g["pair_cost"])).fillna(0)
    g["res_q"] = (g["q_up"] - g["q_dn"]).abs()
    g["res_side_up"] = g["q_up"] > g["q_dn"]
    g["res_won"] = np.where(g["res_side_up"], g["winner"] == "Up", g["winner"] == "Down")
    g["res_px"] = np.where(g["res_side_up"], g["pu"], g["pd"])
    g["res_pnl"] = g["pnl"] + g["fees"] - g["pair_pnl"]  # gross directional part
    g["net_ratio"] = g["res_q"] / (g["q_up"] + g["q_dn"])
    return g


def per_window_vol(pm: pd.DataFrame, btc: BTC) -> pd.DataFrame:
    """Per-window realised BTC volatility / range / net move next to the bot's per-window PnL."""
    w = pm.reset_index()[["slug", "t", "cost", "pnl", "pair_pnl", "res_pnl", "fees", "era"]].rename(columns={"t": "ws"})
    lc = np.log(btc.close)
    idx0 = (w["ws"] - btc.t0).to_numpy()
    ok = (idx0 > 0) & (idx0 + 300 < len(lc))
    w, idx0 = w[ok].copy(), idx0[ok]
    seg = np.stack([lc[i:i + 301] for i in idx0])
    w["rv_bps"] = np.sqrt((np.diff(seg, axis=1) ** 2).sum(axis=1)) * 1e4
    w["move_bps"] = np.abs(seg[:, -1] - seg[:, 0]) * 1e4
    w["rvq"] = pd.qcut(w["rv_bps"], 5, labels=["q1 calm", "q2", "q3", "q4", "q5 wild"])
    w["mq"] = pd.qcut(w["move_bps"], 5, labels=["q1 flat", "q2", "q3", "q4", "q5 trend"])
    return w


def main():
    """Build data/fills.parquet, enriched.parquet, per_market.parquet and per_window_vol.parquet."""
    fills = load_fills()
    fills.to_parquet(DATA / "fills.parquet", index=False)
    markets = load_markets()
    btc = BTC.load()
    e = inventory_path(enrich(fills, btc, markets))
    e["regime"] = np.where(e["timestamp"] < SPLIT_TS, "A", "B")
    e.to_parquet(DATA / "enriched.parquet", index=False)
    pm = per_market(fills, markets)
    pm.to_parquet(DATA / "per_market.parquet")
    per_window_vol(pm, btc).to_parquet(DATA / "per_window_vol.parquet")
    print(f"fills {len(fills):,}  enriched {len(e):,}  windows {len(pm):,}")


if __name__ == "__main__":
    main()
