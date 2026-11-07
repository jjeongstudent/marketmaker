"""Regenerate the figures and headline numbers used in REPORT.md from data/.

    python -m whiskas.report

Figures follow one visual system: validated categorical slots (maker = blue, taker = orange), one
y-axis per chart, solid hairline grids, 2 px lines, direct labels on line ends.
"""

from __future__ import annotations

import json

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import DATA, FIGURES  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3de"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
ROLE = {"maker": BLUE, "taker": ORANGE}
SPLIT = pd.Timestamp("2026-10-27", tz="UTC")


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
        "axes.titlesize": 11, "axes.titlelocation": "left", "axes.labelsize": 9, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-", "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 8, "font.size": 9,
        "font.family": "DejaVu Sans",
    })


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def end_label(ax, x, y, text, color=INK2, dy=0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.plot([x], [y], marker="o", ms=4, color=color)


# ------------------------------------------------------------------------------------ figures

def fig_pnl_decomposition(stats):
    p = pd.read_parquet(DATA / "meta" / "user_pnl_1d.parquet")
    p["dt"] = pd.to_datetime(p["timestamp"], unit="s", utc=True)
    p = p[p["dt"] >= "2026-06-12"]
    series = [
        ("Gross trading PnL (before fees)", p["trade_pnl"], INK2),
        ("Rebates + rewards", p["taker_rebate"] + p["maker_rebate"] + p["reward_income"], AQUA),
        ("All-in PnL", p["economic_pnl"], BLUE),
        ("Taker fees paid", p["fees"], ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for lab, y, c in series:
        ax.plot(p["dt"], y / 1e3, color=c, label=lab)
        v = y.iloc[-1] / 1e3
        end_label(ax, p["dt"].iloc[-1], v, f"{lab}  {'-' if v < 0 else ''}${abs(v):,.0f}k", c)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("cumulative USD (thousands)")
    ax.set_title("x-MoneyForWhiskas: where the money comes from (Jun 12 - Nov 7, 2026)")
    ax.legend(loc="upper left")
    ax.set_xlim(p["dt"].iloc[0], p["dt"].iloc[-1] + pd.Timedelta(days=62))
    ax.set_xticks(pd.date_range("2026-07-01", p["dt"].iloc[-1].tz_convert(None), freq="MS").tz_localize("UTC"))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.spines["bottom"].set_bounds(matplotlib.dates.date2num(p["dt"].iloc[0]), matplotlib.dates.date2num(p["dt"].iloc[-1]))
    save(fig, "01_pnl_decomposition.png")
    last = p.iloc[-1]
    stats["pnl"] = {k: float(last[k]) for k in ("trade_pnl", "fees", "taker_rebate", "maker_rebate",
                                                 "reward_income", "economic_pnl", "volume_usdc", "trade_count")}


def fig_flow(fills, stats):
    b = fills[fills["asset"] == "btc"]
    d = b.groupby([b["dt"].dt.floor("D"), "role"])["notional"].sum().unstack(fill_value=0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax1.bar(d.index, d["taker"] / 1e3, width=0.8, color=ORANGE, label="taker (crosses the spread)")
    ax1.bar(d.index, d["maker"] / 1e3, width=0.8, bottom=d["taker"] / 1e3, color=BLUE, label="maker (resting order hit)",
            edgecolor=SURFACE, linewidth=0.4)
    ax1.set_ylabel("daily notional, USD k")
    ax1.set_title("Daily notional by role: a taker-heavy bot that turned maker-heavy on Oct 27")
    ax1.legend(loc="upper left")
    share = (d["maker"] / d.sum(axis=1)).rolling(3, min_periods=1).mean() * 100
    ax2.plot(share.index, share, color=BLUE)
    ax2.axvline(SPLIT, color=INK2, lw=0.8)
    ax2.annotate("regime B", (SPLIT, 85), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("maker share, %")
    save(fig, "02_flow_by_role.png")
    stats["flow"] = b.groupby([np.where(b["dt"] < SPLIT, "A", "B"), "role"])["notional"].sum().unstack().round(0).to_dict()


def fig_timing(fills):
    b = fills[(fills["asset"] == "btc") & fills["tau"].between(0, 299)]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    bins = np.arange(0, 301, 10)
    for ax, (lab, sel) in zip(axes, (("Regime A (Jun 16 - Oct 26)", b["dt"] < SPLIT), ("Regime B (Oct 27 - Nov 7)", b["dt"] >= SPLIT))):
        for role in ("maker", "taker"):
            x = b[sel & (b["role"] == role)]
            h, _ = np.histogram(x["tau"], bins=bins, weights=x["notional"])
            ax.step(bins[:-1], 100 * h / h.sum(), where="post", color=ROLE[role], label=role)
        ax.axvspan(240, 300, color=GRID, alpha=0.6, lw=0)
        ax.set_title(lab)
        ax.set_xlabel("seconds since the window opened")
    axes[0].set_ylabel("% of role's notional per 10 s")
    axes[0].legend(loc="upper center")
    axes[0].annotate("settlement\naveraging\nminute", (270, 0.97), xycoords=("data", "axes fraction"), ha="center",
                     va="top", fontsize=7, color=INK2)
    save(fig, "03_timing_in_window.png")


def fig_price_cap(fills):
    b = fills[fills["asset"] == "btc"].copy()
    b["week"] = b["dt"].dt.tz_convert(None).dt.to_period("W").dt.start_time
    q = b.groupby("week")["price"].agg(hi=lambda s: s.quantile(0.999), lo=lambda s: s.quantile(0.001))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
    ax1.step(q.index, q["hi"], where="post", color=BLUE, label="99.9th pct fill price")
    ax1.step(q.index, q["lo"], where="post", color=ORANGE, label="0.1th pct fill price")
    ax1.set_ylim(0, 1)
    ax1.set_title("Price band: never buys above a cap")
    ax1.legend(loc="center left")
    ax1.tick_params(axis="x", rotation=30)
    for role in ("maker", "taker"):
        x = b[b["role"] == role]
        h, e = np.histogram(x["price"], bins=np.arange(0.005, 1.0, 0.02), weights=x["notional"])
        ax2.step(e[:-1], 100 * h / h.sum(), where="post", color=ROLE[role], label=role)
    ax2.set_title("Fill price distribution (all history)")
    ax2.set_xlabel("token price")
    ax2.set_ylabel("% of notional per 2c bin")
    ax2.legend()
    save(fig, "04_price_band.png")


def fig_pairs(stats):
    pm = pd.read_parquet(DATA / "per_market.parquet")
    pm = pm[pm["pair_q"] > 0]
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    h, e = np.histogram(pm["pair_cost"].clip(0.6, 1.4), bins=np.arange(0.6, 1.4001, 0.01))
    ax.bar(e[:-1], h, width=0.0085, align="edge", color=BLUE)
    ax.axvline(1.0, color=INK, lw=1)
    med = pm["pair_cost"].median()
    ax.annotate(f"median pair cost ${med:.3f}\n(redeems for $1.000)", (med, h.max() * 0.9), xytext=(-150, 0),
                textcoords="offset points", fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.set_title("Per window: average Up price + average Down price paid")
    ax.set_xlabel("pair cost (USD per Up+Down pair; tails clipped at 0.6 and 1.4)")
    ax.set_ylabel("windows")
    save(fig, "05_pair_cost.png")
    stats["pairs"] = {"median_pair_cost": float(med), "share_windows_both_sides": float((pd.read_parquet(DATA / "per_market.parquet")["pair_q"] > 0).mean())}


def fig_markouts(stats):
    e = pd.read_parquet(DATA / "enriched.parquet", columns=["regime", "role", "size", "price", "mo_1s", "mo_2s", "mo_5s",
                                                            "mo_10s", "mo_30s", "mo_60s", "edge", "payoff_edge"])
    hs = [0, 1, 2, 5, 10, 30, 60]
    cols = ["edge", "mo_1s", "mo_2s", "mo_5s", "mo_10s", "mo_30s", "mo_60s"]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    out = {}
    for (reg, role), d in e.groupby(["regime", "role"]):
        y = [100 * np.average(d[c], weights=d["size"]) for c in cols]
        ls = "-" if reg == "A" else (0, (1, 1.5))
        ax.plot(hs, y, color=ROLE[role], ls=ls, marker="o", ms=4, label=f"{role}, regime {reg}")
        out[f"{reg}_{role}"] = {"markouts_c": dict(zip(map(str, hs), np.round(y, 3))),
                                "realised_c": float(100 * np.average(d["payoff_edge"], weights=d["size"]))}
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xscale("symlog", linthresh=2)
    ax.set_xticks(hs)
    ax.set_xticklabels([str(h) for h in hs])
    ax.set_xlim(-0.15, 66)
    ax.set_xlabel("seconds after the fill")
    ax.set_ylabel("model value - price paid, cents/share")
    ax.set_title("Markouts: neither role is adversely selected; the edge keeps accruing after the fill")
    ax.legend(ncol=2, loc="upper left")
    save(fig, "06_markouts.png")
    stats["markouts"] = out


def fig_vol(stats):
    w = pd.read_parquet(DATA / "per_window_vol.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for ax, col, lab in ((axes[0], "rvq", "realised BTC volatility in the window"), (axes[1], "mq", "|BTC net move| in the window")):
        g = w.groupby(col, observed=True)[["pair_pnl", "res_pnl", "fees", "pnl"]].mean()
        x = np.arange(len(g))
        ax.bar(x - 0.2, g["pair_pnl"], width=0.38, color=BLUE, label="locked pair profit")
        ax.bar(x + 0.2, g["pnl"], width=0.38, color=AQUA, label="net PnL after fees")
        ax.set_xticks(x)
        ax.set_xticklabels(["calm" if col == "rvq" else "flat", "", "", "", "wild" if col == "rvq" else "trend"])
        ax.set_xlabel(f"quintile of {lab}")
        ax.set_title("Per-window PnL by " + ("volatility" if col == "rvq" else "trendiness"))
    axes[0].set_ylabel("USD per window (before rebates)")
    axes[1].legend(loc="upper right")
    save(fig, "07_pnl_vs_regime.png")
    stats["vol"] = w.groupby("rvq", observed=True)[["pnl", "pair_pnl", "cost"]].mean().round(2).to_dict()


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
                 for v in row.to_numpy()]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def tables(fills: pd.DataFrame) -> str:
    """Headline tables (markdown) quoted in REPORT.md."""
    out = []
    b = fills[fills["asset"] == "btc"].copy()
    b["regime"] = np.where(b["dt"] < SPLIT, "A (Jun 16 - Oct 26)", "B (Oct 27 - Nov 7)")
    tot = pd.DataFrame({
        "fills": [f"{len(fills):,}"], "notional": [f"${fills['notional'].sum():,.0f}"],
        "shares": [f"{fills['size'].sum():,.0f}"], "BTC 5m share of notional": [f"{b['notional'].sum() / fills['notional'].sum():.1%}"],
        "BTC windows traded": [f"{b['slug'].nunique():,}"], "BUY share of fills": [f"{(fills['side'] == 'BUY').mean():.4%}"],
    }, index=pd.Index(["all history"], name=""))
    out += ["### Scale", md_table(tot)]
    g = b.groupby(["regime", "role"]).agg(fills=("size", "size"), notional=("notional", "sum"), fees=("fee", "sum"),
                                          median_size=("size", "median"), median_price=("price", "median"),
                                          median_tau_s=("tau", "median"))
    g["notional share"] = g["notional"] / g.groupby(level=0)["notional"].transform("sum")
    g.index = [f"{a} / {r}" for a, r in g.index]
    g.index.name = "regime / role"
    out += ["### Role split", md_table(g)]
    e = pd.read_parquet(DATA / "enriched.parquet", columns=["regime", "role", "size", "price", "edge", "payoff_edge", "fee",
                                                            "mo_60s"])
    r = e.groupby(["regime", "role"]).apply(lambda d: pd.Series({
        "shares": d["size"].sum(),
        "model edge at fill, c": 100 * np.average(d["edge"], weights=d["size"]),
        "60 s markout, c": 100 * np.average(d["mo_60s"], weights=d["size"]),
        "realised (won - price), c": 100 * np.average(d["payoff_edge"], weights=d["size"]),
        "taker fee, c": 100 * d["fee"].sum() / d["size"].sum(),
    }))
    r.index = [f"{a} / {b_}" for a, b_ in r.index]
    r.index.name = "regime / role"
    out += ["### Edge per share (size-weighted, cents)", md_table(r)]
    # PnL attribution by role (all BTC history)
    e["gross"] = e["size"] * e["payoff_edge"]
    att = pd.DataFrame({"gross_edge": e.groupby("role")["gross"].sum(), "fees": 0.0 - e.groupby("role")["fee"].sum()})
    att["fees"] = att["fees"].where(att["fees"] != 0, 0.0)
    pnl = pd.read_parquet(DATA / "meta" / "user_pnl_1d.parquet").iloc[-1]
    att.loc["taker", "rebates"] = pnl["taker_rebate"]
    att.loc["maker", "rebates"] = pnl["maker_rebate"] + pnl["reward_income"]
    att["net"] = att.sum(axis=1)
    att.index.name = "role"
    out += ["### PnL attribution by role (USD, BTC 5m fills; rebates from the Data API)", md_table(att, "{:,.0f}")]
    return "\n\n".join(out) + "\n"


def tape_stats() -> str:
    """Market share and counterparties from the sampled full-market tapes (data/tapes)."""
    from .config import WALLET
    files = sorted((DATA / "tapes").glob("*.parquet"))
    if not files:
        return ""
    t = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    t["ws"] = t["slug"].str.rsplit("-", n=1).str[1].astype(int)
    t["tau"] = t["timestamp"] - t["ws"]
    t["regime"] = np.where(t["ws"] < SPLIT.timestamp(), "A", "B")
    t["notional"] = t["size"] * t["price"]
    rows = {}
    for reg, d in t.groupby("regime"):
        live = d[(d["tau"] >= 0) & (d["tau"] < 300)]
        tk, mk = live[live["role"] == "taker"], live[live["role"] == "maker"]
        bot = live[live["proxy_wallet"] == WALLET]
        bt = set(bot.loc[bot["role"] == "taker", "transaction_hash"])
        bm = set(bot.loc[bot["role"] == "maker", "transaction_hash"])
        others = live[live["proxy_wallet"] != WALLET]
        makers_vs = others[others["transaction_hash"].isin(bt) & (others["role"] == "maker")].groupby("proxy_wallet")["notional"].sum()
        takers_vs = others[others["transaction_hash"].isin(bm) & (others["role"] == "taker")].groupby("proxy_wallet")["notional"].sum()
        rows[reg] = {
            "windows": d["slug"].nunique(),
            "live notional / window": f"${tk['notional'].sum() / d['slug'].nunique():,.0f}",
            "pre-open share of volume": f"{d.loc[(d['role'] == 'taker') & (d['tau'] < 0), 'notional'].sum() / d.loc[d['role'] == 'taker', 'notional'].sum():.1%}",
            "bot share of taker volume": f"{tk.loc[tk['proxy_wallet'] == WALLET, 'notional'].sum() / tk['notional'].sum():.2%}",
            "bot share of maker volume": f"{mk.loc[mk['proxy_wallet'] == WALLET, 'notional'].sum() / mk['notional'].sum():.2%}",
            "wallets hitting its quotes": f"{len(takers_vs):,}",
            "top-5 share of that flow": f"{takers_vs.nlargest(5).sum() / takers_vs.sum():.0%}",
            "wallets it lifts from": f"{len(makers_vs):,}",
            "top-5 share of those makers": f"{makers_vs.nlargest(5).sum() / makers_vs.sum():.0%}",
        }
    df = pd.DataFrame(rows)
    df.index.name = "sampled tapes"
    return ("### Market share and counterparties (sampled full tapes)\n\n" + md_table(df) + "\n\n" +
            "### Model-free markouts: other traders' token VWAP minus the bot's price (cents, mean)\n\n" +
            md_table(tape_markouts(t)) + "\n")


def tape_markouts(t: pd.DataFrame) -> pd.DataFrame:
    """Where everyone else trades the same token before/after each bot fill (no fair-value model)."""
    from .config import WALLET
    prints = t[(t["role"] == "taker") & (t["proxy_wallet"] != WALLET)].copy()
    prints["up_px"] = np.where(prints["outcome_index"] == 0, prints["price"], 1 - prints["price"])
    spans = {"-30..-6 s": (-30, -6), "-5..-1 s": (-5, -1), "+1..+5 s": (1, 5), "+6..+30 s": (6, 30)}
    acc: dict = {}
    for slug, d in t[t["proxy_wallet"] == WALLET].groupby("slug"):
        o = prints[prints["slug"] == slug].sort_values("timestamp")
        if o.empty:
            continue
        ts, px, sz = o["timestamp"].to_numpy(), o["up_px"].to_numpy(), o["size"].to_numpy()
        for r in d.itertuples():
            for lab, (a, b) in spans.items():
                m = (ts >= r.timestamp + a) & (ts <= r.timestamp + b)
                if m.any():
                    v = np.average(px[m], weights=sz[m])
                    v = v if r.outcome_index == 0 else 1 - v
                    acc.setdefault((r.regime, r.role, lab), []).append(100 * (v - r.price))
    out = pd.Series({k: float(np.mean(v)) for k, v in acc.items()}).unstack()[list(spans)]
    out.index = [f"{a} / {b}" for a, b in out.index]
    out.index.name = "regime / role"
    return out


def main():
    style()
    stats: dict = {}
    fills = pd.read_parquet(DATA / "fills.parquet")
    (FIGURES.parent / "tables.md").write_text(tables(fills) + "\n" + tape_stats())
    fig_pnl_decomposition(stats)
    fig_flow(fills, stats)
    fig_timing(fills)
    fig_price_cap(fills)
    fig_pairs(stats)
    fig_markouts(stats)
    fig_vol(stats)
    (FIGURES.parent / "stats.json").write_text(json.dumps(stats, indent=1, default=str))
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
