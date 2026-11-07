"""Download everything needed to reverse-engineer the wallet into data/.

Usage (each step is resumable; finished UTC days are skipped):

    python -m whiskas.collect stats
    python -m whiskas.collect trades --role all      # maker + taker fills
    python -m whiskas.collect trades --role taker    # taker fills only (role labels)
    python -m whiskas.collect activity               # merges / splits / redeems / rebates / rewards
    python -m whiskas.collect markets                # Gamma metadata for every traded market
    python -m whiskas.collect klines --days 30       # Binance BTCUSDT 1 s klines
    python -m whiskas.collect rebates --days 30      # per-market maker rebates (CLOB)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from . import binance, polymarket as pm
from .config import DATA, WALLET

DAY = 86_400
JOIN_DATE = 1780950063  # profile join date (2026-06-08T20:21:03Z) from /v2/user-stats

NON_TRADE_TYPES = ["SPLIT", "MERGE", "REDEEM", "CONVERSION", "MIGRATION", "REWARD", "YIELD",
                   "MAKER_REBATE", "TAKER_REBATE", "REFERRAL_REWARD", "DEPOSIT", "WITHDRAWAL", "TIP"]


def utc_day(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d")


def day_starts(start_ts: int, end_ts: int) -> list[int]:
    d0 = start_ts - start_ts % DAY
    return list(range(d0, end_ts + 1, DAY))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------- trades

TRADE_COLS = ["timestamp", "side", "token_id", "condition_id", "size", "price", "outcome_index", "slug",
              "transaction_hash"]


def _trades_day(day0: int, role: str, out_dir: Path, now: int) -> tuple[str, int]:
    name = utc_day(day0)
    path = out_dir / f"{name}.parquet"
    complete = day0 + DAY < now - 600  # the day is over (with 10 min of indexing slack)
    if path.exists() and complete:
        return name, -1
    rows = [{k: r.get(k) for k in TRADE_COLS}
            for r in pm.iter_trades(WALLET, day0, day0 + DAY - 1, taker_only=(role == "taker"))]
    df = pd.DataFrame(rows, columns=TRADE_COLS)
    df = df.astype({"timestamp": "int64", "size": "float64", "price": "float64"})
    df["outcome_index"] = df["outcome_index"].astype("int16")
    df.to_parquet(path if complete else out_dir / f"{name}.partial.parquet", index=False)
    return name, len(df)


def cmd_trades(args):
    out_dir = DATA / f"trades_{args.role}"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = int(time.time())
    start = args.start or JOIN_DATE
    days = day_starts(start, now)
    total = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_trades_day, d, args.role, out_dir, now) for d in days]
        for f in as_completed(futs):
            name, n = f.result()
            if n >= 0:
                total += n
                log(f"trades[{args.role}] {name}: {n} rows")
    log(f"trades[{args.role}] done, {total} new rows")


# ------------------------------------------------------------------------------- activity

def cmd_activity(args):
    rows = list(pm.iter_activity(WALLET, 1, int(time.time()) + DAY, NON_TRADE_TYPES))
    df = pd.DataFrame(rows)
    keep = [c for c in ["timestamp", "type", "side", "size", "usdc_size", "price", "token_id", "condition_id",
                        "outcome", "outcome_index", "slug", "transaction_hash"] if c in df.columns]
    df = df[keep]
    df.to_parquet(DATA / "activity.parquet", index=False)
    log(f"activity: {len(df)} rows; types: {df['type'].value_counts().to_dict() if len(df) else {}}")


# ---------------------------------------------------------------------------------- stats

def cmd_stats(args):
    meta = DATA / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    (meta / "user_stats.json").write_text(json.dumps({"fetched_at": stamp, **pm.user_stats(WALLET)}, indent=1))
    (meta / "user_volume.json").write_text(json.dumps({"fetched_at": stamp, **pm.user_volume(WALLET)}, indent=1))
    for fid in ("1d", "1h"):
        pnl = pm.user_pnl(WALLET, interval="all", fidelity=fid)
        pd.DataFrame(pnl["points"]).to_parquet(meta / f"user_pnl_{fid}.parquet", index=False)
    log("stats saved")


# -------------------------------------------------------------------------------- markets

def traded_slugs() -> list[str]:
    files = sorted((DATA / "trades_all").glob("*.parquet"))
    slugs = set()
    for f in files:
        slugs.update(pd.read_parquet(f, columns=["slug"])["slug"].unique())
    return sorted(slugs)


def cmd_markets(args):
    path = DATA / "markets.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["slug", "closed"])
    done = set(have.loc[have["closed"].astype(bool), "slug"]) if len(have) else set()
    todo = [s for s in traded_slugs() if s not in done]
    log(f"markets: {len(todo)} slugs to fetch ({len(done)} cached)")
    batches = [todo[i:i + 40] for i in range(0, len(todo), 40)]
    out = []
    with ThreadPoolExecutor(args.workers) as ex:
        for i, evs in enumerate(ex.map(pm.gamma_events_by_slug, batches)):
            for ev in evs:
                if ev.get("markets"):
                    out.append(pm.flatten_updown_event(ev))
            if i % 50 == 0:
                log(f"markets: batch {i}/{len(batches)}")
    new = pd.DataFrame(out)
    merged = pd.concat([have[have["slug"].isin(done)], new], ignore_index=True).drop_duplicates("slug", keep="last")
    merged.to_parquet(path, index=False)
    log(f"markets: {len(merged)} rows saved; unresolved {int((~merged['closed'].astype(bool)).sum())}")


# ----------------------------------------------------------------------- official open/close

def cmd_strikes(args):
    """Chainlink open/close for every traded BTC window (polymarket.com crypto-price endpoint)."""
    out = DATA / "strikes.parquet"
    have = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["slug"])
    m = pd.read_parquet(DATA / "markets.parquet")
    m = m[m["slug"].str.startswith("btc-updown-5m-") & ~m["slug"].isin(set(have["slug"]))]
    ws = m["slug"].str.rsplit("-", n=1).str[1].astype(int)
    # The endpoint refuses windows older than 30 days ("Timestamp too old for Chainlink API").
    recent = (ws > time.time() - 29.5 * DAY) & (ws + 300 < time.time())
    m, ws = m[recent], ws[recent]
    iso = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    jobs = [(s, iso(w), iso(w + 300)) for s, w in zip(m["slug"], ws)]
    log(f"strikes: {len(jobs)} windows to fetch")

    def one(job):
        slug, a, b = job
        try:
            r = pm.crypto_price("BTC", a, b)
        except Exception as exc:  # keep going; failures are retried on the next run
            return {"slug": slug, "error": str(exc)[:200]}
        return {"slug": slug, "open_price": r.get("openPrice"), "close_price": r.get("closePrice"),
                "completed": r.get("completed")}

    rows = []
    with ThreadPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs)):
            rows.append(r)
            if i % 2000 == 0:
                log(f"strikes: {i}/{len(jobs)}")
                pd.concat([have, pd.DataFrame(rows)], ignore_index=True).to_parquet(out, index=False)
    df = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    if "error" in df:
        df = df[df["error"].isna()].drop(columns="error")
    df.to_parquet(out, index=False)
    log(f"strikes: {len(df)} saved")


# ------------------------------------------------------------------------------ market tapes

TAPE_COLS = ["timestamp", "proxy_wallet", "name", "side", "outcome_index", "size", "price", "transaction_hash"]


def cmd_tapes(args):
    """Full trade tapes (all wallets, maker and taker rows) for a random sample of traded windows."""
    out = DATA / "tapes"
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(DATA / "markets.parquet")
    m = m[m["slug"].str.startswith("btc-updown-5m-")].copy()
    m["ws"] = m["slug"].str.rsplit("-", n=1).str[1].astype(int)
    split = int(dt.datetime(2026, 10, 27, tzinfo=dt.timezone.utc).timestamp())
    a = m[(m["ws"] < split) & (m["ws"] > split - 75 * DAY)].sample(args.sample, random_state=7)
    b = m[(m["ws"] >= split) & (m["ws"] + 300 < time.time())].sample(args.sample, random_state=7)
    todo = [r for r in pd.concat([a, b]).itertuples() if not (out / f"{r.slug}.parquet").exists()]
    log(f"tapes: {len(todo)} markets")

    def one(r):
        rows = [{k: x.get(k) for k in TAPE_COLS} for x in pm.iter_market_trades(r.condition_id, taker_only=False)]
        tkr = {(x["transaction_hash"], x["proxy_wallet"]) for x in pm.iter_market_trades(r.condition_id, taker_only=True)}
        df = pd.DataFrame(rows, columns=TAPE_COLS)
        df["role"] = ["taker" if (h, w) in tkr else "maker" for h, w in zip(df["transaction_hash"], df["proxy_wallet"])]
        df["slug"] = r.slug
        df.to_parquet(out / f"{r.slug}.parquet", index=False)
        return len(df)

    with ThreadPoolExecutor(args.workers) as ex:
        for i, n in enumerate(ex.map(one, todo)):
            if i % 25 == 0:
                log(f"tapes: {i}/{len(todo)} ({n} rows)")
    log("tapes: done")


# --------------------------------------------------------------------------------- klines

def _klines_day(day0: int, out_dir: Path, now: int) -> tuple[str, int]:
    name = utc_day(day0)
    path = out_dir / f"{name}.parquet"
    complete = day0 + DAY < now - 120
    if path.exists() and complete:
        return name, -1
    df = binance.klines_1s(day0, min(day0 + DAY, now))
    df.to_parquet(path if complete else out_dir / f"{name}.partial.parquet", index=False)
    return name, len(df)


def cmd_klines(args):
    out_dir = DATA / "klines_1s"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = int(time.time())
    start = args.start or (now - args.days * DAY)
    days = day_starts(start, now)
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_klines_day, d, out_dir, now) for d in days]
        for f in as_completed(futs):
            name, n = f.result()
            if n >= 0:
                log(f"klines {name}: {n} rows")


# -------------------------------------------------------------------------------- rebates

def cmd_rebates(args):
    out = DATA / "maker_rebates.parquet"
    have = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["date"])
    now = int(time.time())
    dates = [utc_day(d) for d in day_starts(now - args.days * DAY, now - DAY)]
    todo = [d for d in dates if d not in set(have["date"])]

    def one(d):
        rows = pm.clob_maker_rebates(d, WALLET)
        return [{**r, "date": d} for r in rows] or [{"date": d, "condition_id": None, "rebated_fees_usdc": "0"}]

    rows = []
    with ThreadPoolExecutor(args.workers) as ex:
        for rs in ex.map(one, todo):
            rows.extend(rs)
    df = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    df["rebated_fees_usdc"] = df["rebated_fees_usdc"].astype(float)
    df.to_parquet(out, index=False)
    log(f"rebates: {len(df)} rows, {df.groupby('date')['rebated_fees_usdc'].sum().tail(5).to_dict()}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="whiskas.collect")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("trades"); p.add_argument("--role", choices=["all", "taker"], default="all")
    p.add_argument("--start", type=int, default=0); p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_trades)
    p = sub.add_parser("activity"); p.set_defaults(fn=cmd_activity)
    p = sub.add_parser("stats"); p.set_defaults(fn=cmd_stats)
    p = sub.add_parser("markets"); p.add_argument("--workers", type=int, default=8); p.set_defaults(fn=cmd_markets)
    p = sub.add_parser("klines"); p.add_argument("--days", type=int, default=30)
    p.add_argument("--start", type=int, default=0); p.add_argument("--workers", type=int, default=6)
    p.set_defaults(fn=cmd_klines)
    p = sub.add_parser("strikes"); p.add_argument("--workers", type=int, default=4); p.set_defaults(fn=cmd_strikes)
    p = sub.add_parser("tapes"); p.add_argument("--sample", type=int, default=150)
    p.add_argument("--workers", type=int, default=4); p.set_defaults(fn=cmd_tapes)
    p = sub.add_parser("rebates"); p.add_argument("--days", type=int, default=30)
    p.add_argument("--workers", type=int, default=4); p.set_defaults(fn=cmd_rebates)
    args = ap.parse_args(argv)
    DATA.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
