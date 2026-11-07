"""Replay a recorded live session (whiskas.live.record) through the reconstructed strategy.

    python -m whiskas.bot.sim data/live/session_*.jsonl.gz --preset B

Execution model
  * The strategy is evaluated on every pricing tick and at most every `eval_ms` on book updates;
    its actions reach the book `latency_ms` later.
  * Taker orders walk the recorded book (token view) up to their limit and size; liquidity we take is
    remembered per level until the real book depletes it, and a token is taken at most once per
    `take_cooldown_ms`.
  * A resting BUY joins the back of its level's queue (queue_ahead = recorded level size). Each taker
    print can explain at most its own size of level depletion at levels its (average) price reached;
    that traded depletion eats the queue ahead and then our order. Depletion not explained by prints is
    treated as cancellations and shrinks the queue ahead pro-rata. Fills therefore never exceed the
    volume that actually traded at our price.
  * We never remove liquidity from the recorded book (the bot is ~1-3% of volume).
Settlement uses the on-chain resolution (Gamma outcomePrices), falling back to TWAP60(end) >=
TWAP60(start) on the recorded Chainlink ticks.
"""

from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..live.analyze import Session
from .model import FairValue
from .params import PRESETS, Params
from .strategy import Book, Inventory, Strategy

WINDOW = 300


@dataclass
class Resting:
    token: str
    price: float          # token price
    size: float
    queue_ahead: float
    placed_ms: int

    @property
    def level(self) -> tuple[str, float]:
        """(Up-book side, Up-book price) where this BUY rests."""
        return ("bid", self.price) if self.token == "Up" else ("ask", round(1 - self.price, 4))


@dataclass
class WindowState:
    ws: int
    up: str
    down: str
    book: Book = field(default_factory=Book)
    inv: Inventory = field(default_factory=Inventory)
    strat: Strategy | None = None
    resting: dict = field(default_factory=dict)  # (token, price) -> Resting
    pending: list = field(default_factory=list)  # (apply_ms, Action)
    last_eval: int = 0
    consumed: dict = field(default_factory=dict)  # (Up-book side, price) -> shares we took from it
    last_take: dict = field(default_factory=dict)  # token -> ms of our last taker order


def taker_fee(p: Params, price: float, size: float) -> float:
    return size * p.fee_rate * price * (1 - price) * (1 - p.taker_rebate)


class Simulator:
    def __init__(self, session: Session, params: Params, feed: str = "chainlink", eval_ms: int = 250,
                 take_cooldown_ms: int = 1000, seed_vol: bool = True):
        self.s = session
        self.take_cooldown_ms = take_cooldown_ms
        self.p = params
        self.feed = feed
        self.eval_ms = eval_ms
        self.fv = FairValue(twap_s=60, basis_noise_bps=0.5 if feed == "chainlink" else 1.0)
        if seed_vol:
            self._seed_vol()
        self.windows: dict[int, WindowState] = {}
        self.tok2win: dict[str, tuple[int, str]] = {}
        for r in session.windows.itertuples():
            st = WindowState(int(r.window_start), r.token_up, r.token_down, strat=Strategy(params))
            self.windows[st.ws] = st
            self.tok2win[r.token_up] = (st.ws, "Up")
            self.tok2win[r.token_down] = (st.ws, "Down")
        self.fills: list[dict] = []
        tr = session.trades
        self.trade_ts = {w: [] for w in self.windows}
        self.trade_dir = {w: [] for w in self.windows}
        self.trade_px = {w: [] for w in self.windows}  # trade price in Up-book terms (taker's average)
        self.trade_left = {w: [] for w in self.windows}  # share size not yet matched to book depletion
        for t in tr.itertuples():
            if t.token in self.tok2win:
                w, tok = self.tok2win[t.token]
                # direction in the Up book: +1 = consumes asks (taker buys Up / sells Down), -1 = consumes bids
                up_buy = (tok == "Up") == (t.side == "BUY")
                self.trade_ts[w].append(t.ts_ms)
                self.trade_dir[w].append(1 if up_buy else -1)
                self.trade_px[w].append(t.price if tok == "Up" else round(1 - t.price, 4))
                self.trade_left[w].append(float(t.size))
        for w in self.windows:  # websocket order is not guaranteed; the bisect lookups need time order
            order = np.argsort(self.trade_ts[w], kind="stable")
            for name in ("trade_ts", "trade_dir", "trade_px", "trade_left"):
                arr = getattr(self, name)[w]
                getattr(self, name)[w] = [arr[i] for i in order]

    def _seed_vol(self, lookback_s: int = 1800):
        """Seed the volatility estimate from Binance 1 s klines before the session (no warm-up bias)."""
        from ..binance import klines_1s
        px = self.s.chainlink if not self.s.chainlink.empty else self.s.binance
        if px is None or px.empty:
            return
        t0 = int(px["ts_ms"].min() // 1000)
        try:
            k = klines_1s(t0 - lookback_s, t0)
        except Exception:
            return
        if len(k) > 600:
            c = k.set_index("ts")["close"].reindex(range(t0 - lookback_s, t0)).ffill().dropna()
            r5 = np.log(c.to_numpy()[5:] / c.to_numpy()[:-5])
            self.fv.seed_variance(float(np.mean(r5 ** 2) / 5))

    # ---------------------------------------------------------------------------------- events
    def events(self):
        """Book updates and pricing ticks merged lazily in time order (book first on ties)."""
        import heapq
        book = ((ts, 0, "book", tok, kind, p) for ts, tok, kind, p in self.s.book_events if tok in self.tok2win)
        px = self.s.chainlink if self.feed == "chainlink" else self.s.binance
        ticks = []
        if px is not None and not px.empty:
            ts_col = px["ts_ms"].to_numpy()
            # Chainlink RTDS stamps are the observation second; they arrive ~1 s later (rt)
            arrive = px["rt"].to_numpy() if "rt" in px else ts_col
            ticks = sorted((int(a), 1, "px", None, int(t), float(v)) for a, t, v in zip(arrive, ts_col, px["px"].to_numpy()))
        return heapq.merge(book, ticks, key=lambda e: (e[0], e[1]))

    def _attribute(self, ws: int, ts: int, side: str, level: float, depleted: float, tol: int = 150,
                   slack: float = 0.02) -> float:
        """How much of a level's depletion is explained by taker prints (each print can explain at most its
        own size). Prints consuming `side` within `tol` ms whose (average) price reached `level` qualify;
        reported prices are the taker's average over a sweep, hence the `slack`."""
        direction = 1 if side == "ask" else -1
        tss, left = self.trade_ts[ws], self.trade_left[ws]
        i = bisect.bisect_left(tss, ts - tol)
        got = 0.0
        while i < len(tss) and tss[i] <= ts + tol and got < depleted:
            if self.trade_dir[ws][i] == direction and left[i] > 0:
                px = self.trade_px[ws][i]
                if (direction < 0 and px <= level + slack) or (direction > 0 and px >= level - slack):
                    take = min(left[i], depleted - got)
                    left[i] -= take
                    got += take
            i += 1
        return got

    def _apply_book(self, st: WindowState, ts: int, tok: str, kind: str, p: dict):
        mirror = tok == st.down
        book = st.book
        if kind == "book":
            b = {round(float(x["price"]), 4): float(x["size"]) for x in p["bids"]}
            a = {round(float(x["price"]), 4): float(x["size"]) for x in p["asks"]}
            if mirror:
                b, a = {round(1 - k, 4): v for k, v in a.items()}, {round(1 - k, 4): v for k, v in b.items()}
            changes = [("bid", k, book.bids.get(k, 0.0), v) for k, v in b.items()] + \
                      [("ask", k, book.asks.get(k, 0.0), v) for k, v in a.items()] + \
                      [("bid", k, v, 0.0) for k, v in book.bids.items() if k not in b] + \
                      [("ask", k, v, 0.0) for k, v in book.asks.items() if k not in a]
            book.bids, book.asks = b, a
            changes = [(sd, k, o, n, self._attribute(st.ws, ts, sd, k, o - n) if n < o else 0.0)
                       for sd, k, o, n in changes]
        else:
            price = round(float(p["price"]), 4)
            side = "bid" if p["side"] == "BUY" else "ask"
            if mirror:
                price = round(1 - price, 4)
                side = "ask" if side == "bid" else "bid"
            levels = book.bids if side == "bid" else book.asks
            old = levels.get(price, 0.0)
            new = float(p["size"])
            key = (side, price)
            if key in st.consumed:  # our phantom takes are released only by real depletion
                st.consumed[key] = 0.0 if new > old else max(0.0, st.consumed[key] - (old - new))
            if new <= 0:
                levels.pop(price, None)
            else:
                levels[price] = new
            traded = self._attribute(st.ws, ts, side, price, old - new) if new < old else 0.0
            changes = [(side, price, old, new, traded)]
        self._maker_fills(st, ts, changes)
        for c_side, c_price, _old, c_new, c_traded in changes:
            if c_new <= 0 and c_traded > 0:
                self._cleared(st, ts, c_side, c_price)

    def _cleared(self, st: WindowState, ts: int, side: str, price: float, tol: int = 150):
        """A print cleared the whole level and kept going: an order of ours at the front of that level
        would have absorbed part of the print's unexplained remainder."""
        r = st.resting.get(("Up", price)) if side == "bid" else st.resting.get(("Down", round(1 - price, 4)))
        if r is None or r.size <= 1e-9 or r.queue_ahead > 1e-9:
            return
        direction = 1 if side == "ask" else -1
        tss, left = self.trade_ts[st.ws], self.trade_left[st.ws]
        i = bisect.bisect_left(tss, ts - tol)
        while i < len(tss) and tss[i] <= ts + tol and r.size > 1e-9:
            if self.trade_dir[st.ws][i] == direction and left[i] > 0:
                q = min(left[i], r.size)
                left[i] -= q
                self._fill(st, ts, r.token, r.price, q, "maker")
                r.size -= q
            i += 1
        if r.size <= 1e-9:
            key = (r.token, r.price)
            st.resting.pop(key, None)
            st.strat.resting.pop(key, None)

    def _maker_fills(self, st: WindowState, ts: int, changes):
        """Queue model for our phantom resting orders: traded depletion at our level eats the queue ahead
        of us and then us; untraded depletion (cancellations) shrinks the queue ahead pro-rata."""
        for key, r in list(st.resting.items()):
            side, lvl = r.level
            for c_side, c_price, old, new, traded in changes:
                if c_side != side or abs(c_price - lvl) > 1e-9 or new >= old:
                    continue
                cancelled = (old - new) - traded
                if cancelled > 0 and old > 0:
                    r.queue_ahead = max(0.0, r.queue_ahead - cancelled * (r.queue_ahead / old))
                if traded > 0:
                    r.queue_ahead -= traded
                    if r.queue_ahead < 0:
                        q = min(r.size, -r.queue_ahead)
                        self._fill(st, ts, r.token, r.price, q, "maker")
                        r.size -= q
                        r.queue_ahead = 0.0
            if r.size <= 1e-9:
                st.resting.pop(key, None)
                st.strat.resting.pop(key, None)

    def _fill(self, st: WindowState, ts: int, tok: str, price: float, size: float, role: str):
        fee = taker_fee(self.p, price, size) if role == "taker" else 0.0
        st.inv.add(tok, size, price, fee)
        fv = self.fv.p_up(ts / 1000, st.ws)
        self.fills.append({"window_start": st.ws, "ts_ms": ts, "tau": ts / 1000 - st.ws, "token": tok,
                           "price": price, "size": size, "role": role, "fee": fee,
                           "fv_tok": None if fv is None else (fv if tok == "Up" else 1 - fv)})

    def _execute(self, st: WindowState, ts: int, act):
        key = (act.token, round(act.price, 4))
        if act.kind == "cancel":
            st.resting.pop(key, None)
            st.strat.resting.pop(key, None)
        elif act.kind == "place":
            side, lvl = ("bid", act.price) if act.token == "Up" else ("ask", round(1 - act.price, 4))
            # post-only: if it would cross now, drop it
            offer = st.book.best_offer(act.token)
            if offer is not None and act.price >= offer - 1e-9:
                st.strat.resting.pop(key, None)
                return
            q = (st.book.bids if side == "bid" else st.book.asks).get(lvl, 0.0)
            st.resting[key] = Resting(act.token, act.price, act.size, q, ts)
            st.strat.resting[key] = act.size
        elif act.kind == "take":
            if ts - st.last_take.get(act.token, -10**12) < self.take_cooldown_ms:
                return
            st.last_take[act.token] = ts
            left = act.size
            side = "ask" if act.token == "Up" else "bid"
            for price, avail in st.book.offers(act.token):
                if price > act.price + 1e-9 or left <= 1e-9:
                    break
                key = (side, price if act.token == "Up" else round(1 - price, 4))
                avail -= st.consumed.get(key, 0.0)
                if avail <= 1e-9:
                    continue
                q = min(left, avail)
                st.consumed[key] = st.consumed.get(key, 0.0) + q
                self._fill(st, ts, act.token, price, q, "taker")
                left -= q

    def _evaluate(self, st: WindowState, now_ms: int):
        tau = now_ms / 1000 - st.ws
        fv = self.fv.p_up(now_ms / 1000, st.ws)
        for act in st.strat.decide(tau, fv, st.book, st.inv):
            if act.kind == "place":
                st.strat.resting[(act.token, round(act.price, 4))] = act.size  # optimistic until it lands
            elif act.kind == "cancel":
                st.strat.resting.pop((act.token, round(act.price, 4)), None)
            st.pending.append((now_ms + self.p.latency_ms, act))
        st.last_eval = now_ms

    def run(self) -> pd.DataFrame:
        for ts, _, kind, tok, a, b in self.events():
            if kind == "px":
                self.fv.on_price(a, b)
            for st in self.windows.values():  # due actions
                if st.pending and st.pending[0][0] <= ts:
                    due = [x for x in st.pending if x[0] <= ts]
                    st.pending = [x for x in st.pending if x[0] > ts]
                    for _, act in due:
                        self._execute(st, ts, act)
            if kind == "book":
                w, _ = self.tok2win[tok]
                st = self.windows[w]
                self._apply_book(st, ts, tok, a, b)
                if st.ws * 1000 <= ts < (st.ws + WINDOW) * 1000 and ts - st.last_eval >= self.eval_ms:
                    self._evaluate(st, ts)
            else:
                w = (ts // 1000) // WINDOW * WINDOW
                st = self.windows.get(w)
                if st is not None:
                    self._evaluate(st, ts)
        return pd.DataFrame(self.fills)


def settle(session: Session, windows) -> dict[int, str]:
    """Winner per window from the on-chain resolution (Gamma outcomePrices); falls back to the
    settlement rule applied to the recorded Chainlink ticks: TWAP60(end) >= TWAP60(start).

    Note: polymarket.com's crypto-price openPrice/closePrice are Chainlink *spot* values at the window
    boundaries; comparing them disagrees with the actual resolution in ~12% of windows."""
    import json as _json

    from ..config import GAMMA_API
    from ..http import get_json
    out = {}
    slugs = [f"btc-updown-5m-{ws}" for ws in windows]
    try:
        for ev in get_json(f"{GAMMA_API}/events", [("slug", x) for x in slugs] + [("limit", str(len(slugs)))]):
            m = ev["markets"][0]
            prices = [float(x) for x in _json.loads(m.get("outcomePrices") or "[]")]
            if m.get("closed") and prices and max(prices) > 0.99:
                out[int(ev["slug"].rsplit("-", 1)[1])] = ["Up", "Down"][prices.index(max(prices))]
    except Exception:
        pass
    cl = session.chainlink
    if cl is not None and not cl.empty:
        sec = cl.assign(s=cl["ts_ms"] // 1000).groupby("s")["px"].last()
        for ws in windows:
            if ws in out:
                continue
            a = sec.loc[ws + WINDOW - 60: ws + WINDOW - 1]
            k = sec.loc[ws - 60: ws - 1]
            if len(a) > 50 and len(k) > 50:
                out[ws] = "Up" if a.mean() >= k.mean() else "Down"
    return out


def summarize(fills: pd.DataFrame, winners: dict[int, str], label: str) -> pd.DataFrame:
    if fills.empty:
        return pd.DataFrame()
    f = fills.copy()
    f["winner"] = f["window_start"].map(winners)
    f = f.dropna(subset=["winner"])
    f["won"] = (f["token"] == f["winner"]).astype(float)
    f["notional"] = f["size"] * f["price"]
    f["pnl"] = f["size"] * (f["won"] - f["price"]) - f["fee"]
    g = f.groupby("role").agg(fills=("size", "size"), shares=("size", "sum"), notional=("notional", "sum"),
                              fees=("fee", "sum"), pnl=("pnl", "sum"), tau_med=("tau", "median"),
                              px_med=("price", "median"))
    g["edge_c_per_share"] = 100 * (g["pnl"] + g["fees"]) / g["shares"]
    g.insert(0, "who", label)
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(prog="whiskas.bot.sim")
    ap.add_argument("session")
    ap.add_argument("--preset", default="B", choices=sorted(PRESETS))
    ap.add_argument("--feed", default="chainlink", choices=["chainlink", "binance"])
    args = ap.parse_args(argv)
    s = Session.load(args.session)
    sim = Simulator(s, PRESETS[args.preset], feed=args.feed)
    fills = sim.run()
    winners = settle(s, sorted(sim.windows))
    real = s.fills.copy()
    if not real.empty:
        tm = s.token_map()
        real["window_start"] = [tm.get(t, (np.nan, None))[0] for t in real["token_id"]]
        real["token"] = [tm.get(t, (None, None))[1] for t in real["token_id"]]
        real = real.dropna(subset=["window_start"])
        real["tau"] = real["timestamp"] - real["window_start"]
        real["fee"] = np.where(real["role"] == "taker",
                               real["size"] * 0.07 * real["price"] * (1 - real["price"]) * 0.5, 0.0)
    out = pd.concat([summarize(fills, winners, "sim"), summarize(real, winners, "real")])
    pd.set_option("display.width", 200)
    print(out.round(3).to_string())
    return fills, real, winners


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------------------- calibration

_SESSION_CACHE: dict = {}


def _run_one(args):
    path, params = args
    if path not in _SESSION_CACHE:
        _SESSION_CACHE[path] = Session.load(path)
    s = _SESSION_CACHE[path]
    fills = Simulator(s, params).run()
    return fills


def real_fills(s: Session) -> pd.DataFrame:
    real = s.fills.copy()
    tm = s.token_map()
    real["window_start"] = [tm.get(t, (np.nan, None))[0] for t in real["token_id"]]
    real["token"] = [tm.get(t, (None, None))[1] for t in real["token_id"]]
    real = real.dropna(subset=["window_start"])
    real["tau"] = real["timestamp"] - real["window_start"]
    real["fee"] = np.where(real["role"] == "taker", real["size"] * 0.07 * real["price"] * (1 - real["price"]) * 0.5, 0.0)
    return real


def calibrate(path: str, base: Params, grid: dict, workers: int = 4) -> pd.DataFrame:
    """Grid-search parameters so the simulated fills match the wallet's real fills in the session's windows.

    Distance = |log(sim/real maker shares)| + |log(sim/real taker shares)| + |maker edge gap| + |taker edge gap|
    (edges in cents per share, realised against the actual winners, scaled by 1/2c)."""
    import itertools
    from concurrent.futures import ProcessPoolExecutor

    s = Session.load(path)
    windows = sorted(int(w) for w in s.windows["window_start"])
    winners = settle(s, windows)
    real = real_fills(s)
    # only windows recorded from 60 s before their open (strike) through their close, and resolved
    first, last = s.chainlink["ts_ms"].min(), int(s.levels["ts"].max())
    complete = [w for w in windows if w in winners and w * 1000 >= first + 60_000 and (w + WINDOW) * 1000 <= last]
    real = real[real["window_start"].isin(complete)]
    r = summarize(real, winners, "real").set_index("who", append=True)
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_run_one, [(path, base.with_(**c)) for c in combos]))
    rows = []
    for c, fills in zip(combos, results):
        fills = fills[fills["window_start"].isin(complete)] if len(fills) else fills
        sm = summarize(fills, winners, "sim")
        row = dict(c)
        for role in ("maker", "taker"):
            rs = r.loc[(role, "real")] if (role, "real") in r.index else None
            ss = sm.loc[role] if role in sm.index else None
            row[f"{role}_shares_sim"] = 0.0 if ss is None else ss["shares"]
            row[f"{role}_shares_real"] = 0.0 if rs is None else rs["shares"]
            row[f"{role}_edge_sim"] = np.nan if ss is None else ss["edge_c_per_share"]
            row[f"{role}_edge_real"] = np.nan if rs is None else rs["edge_c_per_share"]
        d = 0.0
        for role in ("maker", "taker"):
            a, b = row[f"{role}_shares_sim"] + 1, row[f"{role}_shares_real"] + 1
            d += abs(np.log(a / b))
            if not np.isnan(row[f"{role}_edge_sim"]) and not np.isnan(row[f"{role}_edge_real"]):
                d += abs(row[f"{role}_edge_sim"] - row[f"{role}_edge_real"]) / 2
        row["distance"] = d
        rows.append(row)
    return pd.DataFrame(rows).sort_values("distance")
