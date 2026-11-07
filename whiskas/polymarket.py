"""Read-only clients for Polymarket's public APIs (Data API v2, Gamma, CLOB).

Data API v1 was retired on 2026-11-02; everything here uses the /v2 routes, which wrap
results in {"data": ..., "pagination": {"next_cursor": ...}} and page with opaque cursors.
"""

from __future__ import annotations

import json
from typing import Iterator, Sequence

from .config import CLOB_API, DATA_API, GAMMA_API
from .http import get_json

PAGE = 1000  # max page size on the v2 feeds


def _paginate(path: str, params: dict) -> Iterator[dict]:
    params = {k: v for k, v in params.items() if v is not None}
    params.setdefault("limit", PAGE)
    cursor = None
    while True:
        q = dict(params)
        if cursor:
            q["cursor"] = cursor
            q.pop("limit", None)  # the cursor carries the page size
        page = get_json(f"{DATA_API}{path}", q)
        rows = page.get("data") or []
        yield from rows
        cursor = (page.get("pagination") or {}).get("next_cursor")
        if not cursor or not rows:
            return


def iter_trades(user: str, start: int, end: int, *, taker_only: bool) -> Iterator[dict]:
    """Fills for `user` in [start, end] (block timestamps, inclusive), newest first.

    taker_only=True  -> only fills where `user` was the taker.
    taker_only=False -> maker and taker fills.
    Every row is reported from `user`'s perspective (side BUY/SELL, outcome, price, size in shares).
    """
    return _paginate(
        "/v2/trades",
        {"user": user, "start": start, "end": end, "taker_only": str(taker_only).lower()},
    )


def iter_market_trades(condition_id: str, *, taker_only: bool = False) -> Iterator[dict]:
    """Every fill in one market (all wallets). With taker_only=False each match appears once per
    side: the taker's row and one row per maker order, sharing the transaction hash."""
    return _paginate("/v2/trades", {"condition": condition_id, "taker_only": str(taker_only).lower()})


def iter_activity(user: str, start: int, end: int, types: Sequence[str] | None = None,
                  direction: str = "ASC") -> Iterator[dict]:
    """Account activity (TRADE, SPLIT, MERGE, REDEEM, REWARD, MAKER_REBATE, TAKER_REBATE, ...)."""
    return _paginate(
        "/v2/activity",
        {
            "user": user,
            "start": start,
            "end": end,
            "type": ",".join(types) if types else None,
            "sort_direction": direction,
            "exclude_deposits_withdrawals": "false",
        },
    )


def user_stats(user: str) -> dict:
    return get_json(f"{DATA_API}/v2/user-stats", {"user": user})["data"]


def user_pnl(user: str, interval: str = "all", fidelity: str = "1d") -> dict:
    return get_json(f"{DATA_API}/v2/user-pnl", {"user": user, "interval": interval, "fidelity": fidelity})["data"]


def user_volume(user: str, start: int = 0, end: int = 0) -> dict:
    return get_json(f"{DATA_API}/v2/user-volume", {"user": user, "start": start, "end": end})["data"]


def prices_history(token_id: str, start: int, end: int, bucket_seconds: int = 1) -> list[dict]:
    rows = []
    for r in _paginate("/v2/prices-history", {"token_id": token_id, "start": start, "end": end,
                                              "bucket_seconds": bucket_seconds}):
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------------- Gamma

def gamma_events_by_slug(slugs: Sequence[str]) -> list[dict]:
    """Gamma events for a batch of slugs (repeated `slug` params). Gamma defaults to 20 rows per
    response, so the limit is set explicitly."""
    if not slugs:
        return []
    return get_json(f"{GAMMA_API}/events", [("slug", s) for s in slugs] + [("limit", str(len(slugs)))])


def flatten_updown_event(ev: dict) -> dict:
    """Pick the fields we need from a 5-minute Up/Down event (one binary market per event)."""
    m = ev["markets"][0]
    outcomes = json.loads(m.get("outcomes") or "[]")
    tokens = json.loads(m.get("clobTokenIds") or "[]")
    prices = json.loads(m.get("outcomePrices") or "[]")
    winner = None
    if m.get("closed") and len(prices) == 2:
        p = [float(x) for x in prices]
        if max(p) > 0.99:
            winner = outcomes[p.index(max(p))]
    fee = m.get("feeSchedule") or {}
    return {
        "slug": ev["slug"],
        "event_id": ev.get("id"),
        "condition_id": m.get("conditionId"),
        "question_id": m.get("questionID"),
        "window_start": ev.get("startTime") or m.get("eventStartTime"),
        "window_end": m.get("endDate"),
        "created_at": m.get("createdAt"),
        "accepting_orders_ts": m.get("acceptingOrdersTimestamp"),
        "token_up": tokens[0] if len(tokens) > 0 else None,
        "token_down": tokens[1] if len(tokens) > 1 else None,
        "outcome_labels": "|".join(outcomes),
        "closed": bool(m.get("closed")),
        "winner": winner,
        "volume": float(m.get("volumeNum") or 0),
        "fees_enabled": m.get("feesEnabled"),
        "fee_type": m.get("feeType"),
        "fee_rate": fee.get("rate"),
        "fee_exponent": fee.get("exponent"),
        "rebate_rate": fee.get("rebateRate"),
        "crypto_config": m.get("cryptoMarketConfigId"),
        "tick": m.get("orderPriceMinTickSize"),
        "min_size": m.get("orderMinSize"),
        "rewards_min_size": m.get("rewardsMinSize"),
        "rewards_max_spread": m.get("rewardsMaxSpread"),
        "resolution_source": m.get("resolutionSource"),
    }


def crypto_price(symbol: str, window_start_iso: str, window_end_iso: str, variant: str = "fiveminute") -> dict:
    """Official open ("price to beat") and close of a crypto Up/Down window, as shown on polymarket.com.

    These are the Chainlink stream values the market settles on (spot before 2026-08-07, TWAP after).
    """
    return get_json("https://polymarket.com/api/crypto/crypto-price",
                    {"symbol": symbol, "eventStartTime": window_start_iso, "variant": variant,
                     "endDate": window_end_iso})


# ------------------------------------------------------------------------------------ CLOB

def clob_book(token_id: str) -> dict:
    return get_json(f"{CLOB_API}/book", {"token_id": token_id})


def clob_maker_rebates(date: str, maker: str) -> list[dict]:
    """Per-market maker rebates credited to `maker` for UTC day `date` (YYYY-MM-DD)."""
    return get_json(f"{CLOB_API}/rebates/current", {"date": date, "maker_address": maker}) or []
