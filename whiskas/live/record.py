"""Record the live order books of the current/next 5-minute BTC windows, Chainlink and Binance
prices, and the target wallet's fills, into one gzipped JSONL file.

    python -m whiskas.live.record --minutes 60

Each line: {"rt": <local receive time, ms>, "src": "clob"|"rtds"|"bin"|"fill"|"meta", "msg": ...}
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import time

import requests
import websockets

from ..config import DATA, DATA_API, GAMMA_API, WALLET

CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RTDS_WS = "wss://ws-live-data.polymarket.com"
BINANCE_TICKER = "https://data-api.binance.vision/api/v3/ticker/bookTicker"


def now_ms() -> int:
    return int(time.time() * 1000)


class Recorder:
    def __init__(self, path, minutes: float):
        self.f = gzip.open(path, "wt", compresslevel=5)
        self.stop_at = time.time() + minutes * 60
        self.tokens: dict[int, list[str]] = {}  # window start -> [up, down]
        self.ws = None
        self.seen_fills: set = set()

    def write(self, src: str, msg) -> None:
        self.f.write(json.dumps({"rt": now_ms(), "src": src, "msg": msg}, separators=(",", ":")) + "\n")

    @property
    def running(self) -> bool:
        return time.time() < self.stop_at

    # ------------------------------------------------------------------ market discovery
    async def window_tokens(self, ws0: int) -> list[str] | None:
        if ws0 in self.tokens:
            return self.tokens[ws0]
        def fetch():
            ev = requests.get(f"{GAMMA_API}/events", params={"slug": f"btc-updown-5m-{ws0}"}, timeout=20).json()
            if not ev:
                return None
            m = ev[0]["markets"][0]
            return {"tokens": json.loads(m["clobTokenIds"]), "condition_id": m["conditionId"], "slug": ev[0]["slug"]}
        info = await asyncio.to_thread(fetch)
        if info:
            self.tokens[ws0] = info["tokens"]
            self.write("meta", {"window_start": ws0, **info})
            return info["tokens"]
        return None

    # --------------------------------------------------------------------------- streams
    async def clob(self):
        while self.running:
            try:
                async with websockets.connect(CLOB_WS, open_timeout=15, max_size=2**24) as ws:
                    self.ws = ws
                    ws0 = int(time.time()) // 300 * 300
                    subs = []
                    for w in (ws0, ws0 + 300):
                        subs += (await self.window_tokens(w)) or []
                    await ws.send(json.dumps({"assets_ids": subs, "type": "market", "custom_feature_enabled": True}))
                    self.write("meta", {"subscribed": subs})
                    pinger = asyncio.create_task(self._ping(ws))
                    rot = asyncio.create_task(self._rotate(ws))
                    try:
                        while self.running:
                            msg = await asyncio.wait_for(ws.recv(), 30)
                            if msg != "PONG":
                                self.write("clob", json.loads(msg))
                    finally:
                        pinger.cancel(); rot.cancel()
            except Exception as exc:  # reconnect on any failure
                self.write("meta", {"clob_error": repr(exc)[:300]})
                await asyncio.sleep(1)

    async def _ping(self, ws):
        while True:
            await asyncio.sleep(10)
            await ws.send("PING")

    async def _rotate(self, ws):
        """At each 5-minute boundary subscribe to the window after next and drop the finished one."""
        while True:
            nxt = (int(time.time()) // 300 + 1) * 300
            await asyncio.sleep(nxt - time.time() + 0.2)
            new = await self.window_tokens(nxt + 300)
            if new:
                await ws.send(json.dumps({"assets_ids": new, "operation": "subscribe"}))
            old = self.tokens.get(nxt - 600)
            if old:
                await ws.send(json.dumps({"assets_ids": old, "operation": "unsubscribe"}))
            self.write("meta", {"rotated_at": nxt, "added": new, "removed": old})

    async def rtds(self):
        sub = {"action": "subscribe", "subscriptions": [
            {"topic": "crypto_prices_chainlink", "type": "*", "filters": "{\"symbol\":\"btc/usd\"}"},
            {"topic": "crypto_prices", "type": "update", "filters": "btcusdt"}]}
        while self.running:
            try:
                async with websockets.connect(RTDS_WS, open_timeout=15, max_size=2**24) as ws:
                    await ws.send(json.dumps(sub))
                    pinger = asyncio.create_task(self._ping(ws))
                    try:
                        while self.running:
                            msg = await asyncio.wait_for(ws.recv(), 30)
                            if msg and msg != "PONG":
                                self.write("rtds", json.loads(msg))
                    finally:
                        pinger.cancel()
            except Exception as exc:
                self.write("meta", {"rtds_error": repr(exc)[:300]})
                await asyncio.sleep(1)

    async def binance(self, hz: float = 4.0):
        s = requests.Session()
        while self.running:
            t = time.time()
            try:
                r = await asyncio.to_thread(s.get, BINANCE_TICKER, params={"symbol": "BTCUSDT"}, timeout=5)
                self.write("bin", r.json())
            except Exception as exc:
                self.write("meta", {"binance_error": repr(exc)[:200]})
            await asyncio.sleep(max(0.0, 1 / hz - (time.time() - t)))

    async def fills(self, every: float = 5.0):
        s = requests.Session()
        while self.running:
            try:
                p = {"user": WALLET, "taker_only": "false", "start": int(time.time()) - 900, "limit": 500}
                r = await asyncio.to_thread(s.get, f"{DATA_API}/v2/trades", params=p, timeout=20)
                rows = r.json().get("data") or []
                pt = {"user": WALLET, "taker_only": "true", "start": int(time.time()) - 900, "limit": 500}
                rt = await asyncio.to_thread(s.get, f"{DATA_API}/v2/trades", params=pt, timeout=20)
                taker_keys = {(x["transaction_hash"], x["token_id"], x["size"], x["price"]) for x in rt.json().get("data") or []}
                for x in rows:
                    k = (x["transaction_hash"], x["token_id"], x["size"], x["price"], x["side"])
                    if k not in self.seen_fills:
                        self.seen_fills.add(k)
                        x = {c: x.get(c) for c in ("timestamp", "side", "token_id", "condition_id", "size", "price",
                                                    "outcome", "slug", "transaction_hash")}
                        x["role"] = "taker" if k[:4] in taker_keys else "maker"
                        self.write("fill", x)
            except Exception as exc:
                self.write("meta", {"fills_error": repr(exc)[:200]})
            await asyncio.sleep(every)

    async def run(self):
        tasks = [asyncio.create_task(c) for c in (self.clob(), self.rtds(), self.binance(), self.fills())]
        while self.running:
            await asyncio.sleep(1)
            self.f.flush()
        await asyncio.sleep(20)  # let the fills poller catch the last window's fills
        for t in tasks:
            t.cancel()
        self.f.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="whiskas.live.record")
    ap.add_argument("--minutes", type=float, default=60)
    args = ap.parse_args(argv)
    out = DATA / "live"
    out.mkdir(parents=True, exist_ok=True)
    path = out / time.strftime("session_%Y%m%d_%H%M%S.jsonl.gz", time.gmtime())
    print(path, flush=True)
    asyncio.run(Recorder(path, args.minutes).run())


if __name__ == "__main__":
    main()
