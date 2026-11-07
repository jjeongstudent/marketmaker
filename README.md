# marketmaker — reverse-engineering x-MoneyForWhiskas

Tooling and findings for reverse-engineering the Polymarket bot **x-MoneyForWhiskas**
(`0x3048d65321be3497164cdfc2996f94f98a2e7537`), which trades the 5-minute "Bitcoin Up or Down" markets.

**Read [REPORT.md](REPORT.md)** for the reconstructed strategy, parameters and evidence.

## Layout

| Path | What |
|---|---|
| `whiskas/collect.py` | Downloads the wallet's fills (maker + taker labelled), activity, PnL series, Gamma metadata of every traded window, Binance 1 s klines, sampled market tapes, official window prices |
| `whiskas/polymarket.py`, `binance.py`, `http.py` | Read-only API clients (Data API **v2** — v1 was retired 2026-11-02) |
| `whiskas/fairvalue.py`, `features.py` | TWAP-aware fair value of the Up token, settlement-rule eras, fill enrichment (edge, markouts, inventory path) |
| `whiskas/report.py` | Regenerates `reports/figures/*.png` and `reports/tables.md` |
| `whiskas/live/record.py`, `analyze.py` | Records live books (CLOB websocket), Chainlink (RTDS), Binance and the wallet's fills; replays books around each fill |
| `whiskas/bot/` | Reference re-implementation: online fair value, decision logic, regime presets, session replay simulator |
| `tests/` | `python -m pytest -q` |

## Quick start

```bash
pip install -r requirements.txt
python -m whiskas.collect stats
python -m whiskas.collect trades --role all
python -m whiskas.collect trades --role taker
python -m whiskas.collect markets
python -m whiskas.collect klines --start 1781049600
python -m whiskas.collect tapes
python -m whiskas.features   # fills/enriched/per-window tables in data/
python -m whiskas.report     # figures + reports/tables.md
```

Data lands in `data/` (git-ignored, ~1 GB with klines). Network access needed: `*.polymarket.com`
(data-api, gamma-api, clob, ws-subscriptions-clob, ws-live-data, polymarket.com, docs) and
`data-api.binance.vision` (`api.binance.com` returns HTTP 451 from US egress).

The bot code is analysis-only: it simulates against recorded sessions and contains no order
submission.
