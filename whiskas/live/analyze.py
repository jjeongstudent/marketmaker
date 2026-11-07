"""Replay a recorded live session and line the wallet's fills up against the order book.

    python -m whiskas.live.analyze data/live/session_*.jsonl.gz

Polymarket shows one shared book per binary market: a BUY Down order at q is the same liquidity as
a SELL Up order at 1-q (Up best bid + Down best ask == 1 at all times). Everything here is expressed
in Up-token terms:  bot BUY Up at p  -> a bid at p;  bot BUY Down at q -> an ask at 1-q.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def iter_lines(path):
    with gzip.open(path, "rt") as fh:
        try:
            for line in fh:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    return
        except EOFError:  # file still being written
            return


@dataclass
class Session:
    windows: pd.DataFrame                    # window_start, slug, condition_id, token_up, token_down
    book_events: list = field(default_factory=list)  # (server_ms, token, kind, payload)
    trades: pd.DataFrame | None = None       # WS last_trade_price
    fills: pd.DataFrame | None = None        # wallet fills (Data API)
    chainlink: pd.DataFrame | None = None    # RTDS chainlink btc/usd, 1 s
    binance: pd.DataFrame | None = None      # bookTicker polls

    @classmethod
    def load(cls, path) -> "Session":
        wins, books, trades, fills, cl, bn = {}, [], [], [], [], []
        for d in iter_lines(path):
            src, msg, rt = d["src"], d["msg"], d["rt"]
            if src == "meta" and "tokens" in msg:
                wins[msg["window_start"]] = {"window_start": msg["window_start"], "slug": msg["slug"],
                                             "condition_id": msg["condition_id"], "token_up": msg["tokens"][0],
                                             "token_down": msg["tokens"][1]}
            elif src == "clob":
                for m in (msg if isinstance(msg, list) else [msg]):
                    et = m.get("event_type") or ("price_change" if "price_changes" in m else None)
                    ts = int(m.get("timestamp") or 0)
                    if et == "book":
                        books.append((ts, m["asset_id"], "book", m))
                    elif et == "price_change":
                        for pc in m["price_changes"]:
                            books.append((ts, pc["asset_id"], "level", pc))
                    elif et == "last_trade_price":
                        trades.append({"ts_ms": ts, "rt": rt, "token": m["asset_id"], "price": float(m["price"]),
                                       "size": float(m["size"]), "side": m["side"],
                                       "tx": m.get("transaction_hash"), "fee_bps": m.get("fee_rate_bps")})
            elif src == "fill":
                fills.append({**msg, "rt": rt})
            elif src == "rtds":
                p = msg.get("payload") or {}
                if msg.get("topic") == "crypto_prices_chainlink" and "value" in p:
                    cl.append({"ts_ms": p["timestamp"], "rt": rt, "px": float(p["value"])})
                elif msg.get("topic") == "crypto_prices" and "value" in p:
                    bn.append({"ts_ms": p["timestamp"], "rt": rt, "px": float(p["value"]), "src": "rtds_binance"})
            elif src == "bin" and "bidPrice" in msg:
                bn.append({"ts_ms": rt, "rt": rt, "px": (float(msg["bidPrice"]) + float(msg["askPrice"])) / 2,
                           "src": "rest"})
        books.sort(key=lambda x: x[0])
        s = cls(pd.DataFrame(wins.values()), books)
        s.trades = pd.DataFrame(trades)
        s.fills = pd.DataFrame(fills).drop_duplicates(["transaction_hash", "token_id", "size", "price"]) if fills else pd.DataFrame()
        s.chainlink = pd.DataFrame(cl).drop_duplicates("ts_ms").sort_values("ts_ms") if cl else pd.DataFrame()
        s.binance = pd.DataFrame(bn).sort_values("ts_ms") if bn else pd.DataFrame()
        return s

    # --------------------------------------------------------------------------------------
    def token_map(self) -> dict:
        out = {}
        for r in self.windows.itertuples():
            out[r.token_up] = (r.window_start, "Up")
            out[r.token_down] = (r.window_start, "Down")
        return out

    def up_book_replay(self, window_start: int):
        """Yield (server_ms, bids{price: size}, asks{price: size}) for the window's Up book after
        each update. Down-token events are mirrored onto the Up book."""
        w = self.windows.set_index("window_start").loc[window_start]
        up, dn = w.token_up, w.token_down
        bids: dict = {}
        asks: dict = {}
        for ts, tok, kind, p in self.book_events:
            if tok not in (up, dn):
                continue
            mirror = tok == dn
            if kind == "book":
                b = {round(float(x["price"]), 4): float(x["size"]) for x in p["bids"]}
                a = {round(float(x["price"]), 4): float(x["size"]) for x in p["asks"]}
                if mirror:
                    b, a = {round(1 - k, 4): v for k, v in a.items()}, {round(1 - k, 4): v for k, v in b.items()}
                bids, asks = b, a
            else:
                price = round(float(p["price"]), 4)
                size = float(p["size"])
                side = p["side"]
                if mirror:
                    price = round(1 - price, 4)
                    side = "SELL" if side == "BUY" else "BUY"
                book = bids if side == "BUY" else asks
                if size <= 0:
                    book.pop(price, None)
                else:
                    book[price] = size
            yield ts, bids, asks


def match_fills(s: Session) -> pd.DataFrame:
    """Attach the WS match time (ms) and Up-book side to each wallet fill."""
    f = s.fills.copy()
    if f.empty:
        return f
    tm = s.token_map()
    f["window_start"] = [tm.get(t, (np.nan, None))[0] for t in f["token_id"]]
    f["outcome"] = [tm.get(t, (None, None))[1] for t in f["token_id"]]
    tr = s.trades.dropna(subset=["tx"]).groupby("tx").agg(ts_ms=("ts_ms", "min"))
    f = f.merge(tr, left_on="transaction_hash", right_index=True, how="left")
    f["up_price"] = np.where(f["outcome"] == "Up", f["price"], 1 - f["price"])
    f["book_side"] = np.where(f["outcome"] == "Up", "bid", "ask")  # where the bot's BUY sits in the Up book
    f["tau_ms"] = f["ts_ms"] - f["window_start"] * 1000
    return f.sort_values("ts_ms")


def price_at(df: pd.DataFrame, ts_ms) -> np.ndarray:
    """Last price at or before ts_ms (vectorised as-of lookup)."""
    if df is None or df.empty:
        return np.full(np.shape(ts_ms), np.nan)
    t = df["ts_ms"].to_numpy()
    i = np.searchsorted(t, np.asarray(ts_ms), side="right") - 1
    out = df["px"].to_numpy()[np.clip(i, 0, len(t) - 1)]
    return np.where(i >= 0, out, np.nan)


def book_context(s: Session, fills: pd.DataFrame, depth: int = 5) -> pd.DataFrame:
    """For each wallet fill, the Up book just before the match: best bid/ask and the size resting at the
    bot's price level."""
    rows = []
    for ws, fw in fills.dropna(subset=["ts_ms"]).groupby("window_start"):
        targets = sorted(fw["ts_ms"].unique())
        snap = {}
        j = 0
        prev = None
        for ts, bids, asks in s.up_book_replay(int(ws)):
            while j < len(targets) and ts >= targets[j]:
                snap[targets[j]] = prev
                j += 1
            prev = (dict(bids), dict(asks))
            if j >= len(targets):
                break
        for r in fw.itertuples():
            b = snap.get(r.ts_ms)
            if not b:
                continue
            bids, asks = b
            bb = max(bids) if bids else np.nan
            ba = min(asks) if asks else np.nan
            lvl = bids.get(round(r.up_price, 4)) if r.book_side == "bid" else asks.get(round(r.up_price, 4))
            rows.append({"transaction_hash": r.transaction_hash, "token_id": r.token_id, "size": r.size,
                         "price": r.price, "best_bid": bb, "best_ask": ba, "level_size_before": lvl,
                         "bid_depth5": sum(v for k, v in sorted(bids.items(), reverse=True)[:depth]),
                         "ask_depth5": sum(v for k, v in sorted(asks.items())[:depth])})
    ctx = pd.DataFrame(rows)
    return fills.merge(ctx, on=["transaction_hash", "token_id", "size", "price"], how="left")


def main(argv=None):
    path = (argv or sys.argv[1:])[0]
    s = Session.load(path)
    print(f"windows {len(s.windows)}  book events {len(s.book_events)}  trades {len(s.trades)}  "
          f"fills {len(s.fills)}  chainlink {len(s.chainlink)}  binance {len(s.binance)}")
    f = book_context(s, match_fills(s))
    f["mid"] = (f["best_bid"] + f["best_ask"]) / 2
    f["at_touch"] = np.where(f["book_side"] == "bid", np.isclose(f["up_price"], f["best_bid"]),
                             np.isclose(f["up_price"], f["best_ask"]))
    print(f.groupby("role").agg(n=("size", "size"), matched=("ts_ms", lambda x: x.notna().mean()),
                                at_touch=("at_touch", "mean")).to_string())


if __name__ == "__main__":
    main()
