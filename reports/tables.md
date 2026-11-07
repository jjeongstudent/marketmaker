### Scale

|  | fills | notional | shares | BTC 5m share of notional | BTC windows traded | BUY share of fills |
|---|---|---|---|---|---|---|
| all history | 1,625,913 | $21,203,434 | 43,936,001 | 98.9% | 37,647 | 99.9994% |

### Role split

| regime / role | fills | notional | fees | median_size | median_price | median_tau_s | notional share |
|---|---|---|---|---|---|---|---|
| A (Jun 16 - Oct 26) / maker | 658,049 | 5,030,254.31 | 0.00 | 7.00 | 0.44 | 126.00 | 0.25 |
| A (Jun 16 - Oct 26) / taker | 846,238 | 15,096,279.81 | 463,875.65 | 30.00 | 0.48 | 119.00 | 0.75 |
| B (Oct 27 - Nov 7) / maker | 70,904 | 592,353.68 | 0.00 | 9.72 | 0.43 | 117.00 | 0.71 |
| B (Oct 27 - Nov 7) / taker | 17,206 | 247,498.54 | 7,241.55 | 30.00 | 0.51 | 196.00 | 0.29 |

### Edge per share (size-weighted, cents)

| regime / role | shares | model edge at fill, c | 60 s markout, c | realised (won - price), c | taker fee, c |
|---|---|---|---|---|---|
| A / maker | 10,275,313.76 | 0.84 | 1.21 | 1.17 | 0.00 |
| A / taker | 31,410,986.12 | 1.01 | 1.60 | 1.62 | 1.48 |
| B / maker | 1,258,995.71 | 0.26 | 0.64 | 0.38 | 0.00 |
| B / taker | 487,246.02 | 0.87 | 1.60 | 1.90 | 1.49 |

### PnL attribution by role (USD, BTC 5m fills; rebates from the Data API)

| role | gross_edge | fees | rebates | net |
|---|---|---|---|---|
| maker | 124,663 | 0 | 35,889 | 160,551 |
| taker | 516,603 | -471,117 | 206,346 | 251,833 |

### Market share and counterparties (sampled full tapes)

| sampled tapes | A | B |
|---|---|---|
| windows | 150 | 150 |
| live notional / window | $24,087 | $21,956 |
| pre-open share of volume | 5.9% | 5.6% |
| bot share of taker volume | 2.43% | 0.35% |
| bot share of maker volume | 0.73% | 1.00% |
| wallets hitting its quotes | 2,044 | 1,749 |
| top-5 share of that flow | 17% | 15% |
| wallets it lifts from | 1,439 | 470 |
| top-5 share of those makers | 20% | 12% |

### Model-free markouts: other traders' token VWAP minus the bot's price (cents, mean)

| regime / role | -30..-6 s | -5..-1 s | +1..+5 s | +6..+30 s |
|---|---|---|---|---|
| A / maker | 1.50 | 0.35 | 1.16 | 1.26 |
| A / taker | 0.90 | 0.04 | 2.02 | 1.95 |
| B / maker | -0.69 | 0.06 | 0.12 | -0.09 |
| B / taker | -0.61 | -1.37 | 2.85 | 2.38 |
