#!/usr/bin/env python3
"""
SPY / QQQ support & resistance zone analyzer.

  * Daily chart   -> last 1 year of daily candles
  * Weekly chart  -> last 5 years of weekly candles
  * Monthly chart -> last 10 years of monthly candles
  * Zones are found from swing highs/lows (pivots), clustered by price,
    and drawn as light bands:
        green = support    (zone below the current price)
        red   = resistance (zone above the current price)
  * EMA(20) blue, EMA(50) black, EMA(200) orange
  * One PNG per ticker (daily on top, weekly in the middle, monthly below).
    No text output.

Usage:
    pip install yfinance pandas numpy matplotlib
    python spy_qqq_levels.py                     # SPY and QQQ
    python spy_qqq_levels.py SPY QQQ IWM DIA     # up to 4 tickers
    python spy_qqq_levels.py AAPL --no-show      # just save PNGs
    python spy_qqq_levels.py --demo              # offline test with synthetic data
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --------------------------------------------------------------------------
# Settings per timeframe
# --------------------------------------------------------------------------
TIMEFRAMES = {
    "Daily": {
        "display_years": 1,   # what is shown on the chart (and used for zones)
        "fetch_period": "2y", # extra history so EMA(200) is valid across the chart
        "interval": "1d",
        "pivot_window": 5,    # bars on each side that must be lower/higher
        "max_zones": 6,
        "emas": [20, 50, 200],
    },
    "Weekly": {
        "display_years": 5,
        "fetch_period": "6y", # extra history so EMA(50) is valid across the chart
        "interval": "1wk",
        "pivot_window": 3,
        "max_zones": 6,
        "emas": [20, 50, 200],
    },
    "Monthly": {
        "display_years": 10,
        "fetch_period": "15y", # extra history so EMA(50) is valid across the chart
        "interval": "1mo",
        "pivot_window": 3,
        "max_zones": 6,
        "emas": [20, 50, 200],
    },
}

EMA_COLORS = {20: "#1F5BD8", 50: "#000000", 200: "#FF8C00"}   # blue, black, orange
MAX_TICKERS = 4

SUPPORT_COLOR = "#2E9E4F"     # medium green (support)
RESIST_COLOR = "#FF6B6B"      # light red (resistance)
ZONE_ALPHA = 0.35             # band fill opacity (raise for stronger bands)
EDGE_ALPHA = 0.9              # opacity of the band's top/bottom border lines


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} ({interval}).")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def synthetic_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Random-walk OHLC data so the script can be tested without internet."""
    import zlib
    weekly = interval == "1wk"
    years = int(period.rstrip("y"))
    n = years * (52 if weekly else 252)
    freq = "W-FRI" if weekly else "B"
    rng = np.random.default_rng(zlib.crc32(f"{ticker}{interval}".encode()))
    start = 560.0 if ticker == "SPY" else 480.0
    vol = 0.022 if weekly else 0.010
    close = start * np.exp(np.cumsum(rng.normal(0.0006 * (5 if weekly else 1), vol, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, vol / 4, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq=freq)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": 0}, index=idx)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def atr(df: pd.DataFrame, length: int = 14) -> float:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_close).abs(),
                    (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    return float(tr.rolling(length).mean().iloc[-1])


def find_pivots(df: pd.DataFrame, window: int):
    """Return list of (date, price, 'H'|'L') swing points."""
    highs, lows = df["High"].values, df["Low"].values
    pivots = []
    for i in range(window, len(df) - window):
        if highs[i] == highs[i - window:i + window + 1].max():
            pivots.append((df.index[i], highs[i], "H"))
        if lows[i] == lows[i - window:i + window + 1].min():
            pivots.append((df.index[i], lows[i], "L"))
    return pivots


def find_rbr_dbr_zones(df, max_zones):
    """Demand zones formed by a tight base + a long green rally candle.

    Rule (applied on every timeframe):
      * Base: 1 to 3 consecutive candles, tight — each candle's range
        <= 1.0 * ATR and the whole base's high-low span <= 1.5 * ATR.
      * Rally: the very next candle is long and green — body >= 0.5 * ATR
        and its close breaks above the base high. That candle is the rally.
      * Formation: the move into the base decides the label —
        RBR (rally-base-rally) when the 1-3 candles before the base
        moved up >= 0.5 * ATR, DBR (drop-base-rally) when they moved
        down <= -0.5 * ATR. Anything else is not an RBR/DBR zone.
    The zone is the base's low-to-high range (padded by 0.25 * ATR).
    Touches count later bars that entered the zone after the rally.
    """
    a = atr(df)
    pad = 0.25 * a
    n = len(df)
    last_close = float(df["Close"].iloc[-1])
    last_date = df.index[-1]
    span_days = max((last_date - df.index[0]).days, 1)
    found = []
    for base_start in range(2, n - 1):
        for base_len in (1, 2, 3):
            base_end = base_start + base_len - 1
            rally_idx = base_end + 1
            if rally_idx >= n:
                continue
            base = df.iloc[base_start:base_end + 1]
            base_low = float(base["Low"].min())
            base_high = float(base["High"].max())
            if base_high - base_low > 1.5 * a:
                continue
            if (base["High"] - base["Low"]).max() > 1.0 * a:
                continue
            r = df.iloc[rally_idx]
            body = float(r["Close"] - r["Open"])
            if body < 0.5 * a or float(r["Close"]) <= base_high:
                continue
            pre = df.iloc[max(0, base_start - 3):base_start]
            if len(pre) == 0:
                continue
            move = float(pre["Close"].iloc[-1] - pre["Open"].iloc[0])
            if move >= 0.5 * a:
                formation = "RBR"
            elif move <= -0.5 * a:
                formation = "DBR"
            else:
                continue
            mid = (base_low + base_high) / 2
            low = min(base_low, mid - pad)
            high = max(base_high, mid + pad)
            touches = int(((df["Low"].iloc[rally_idx + 1:] <= high) &
                           (df["High"].iloc[rally_idx + 1:] >= low)).sum())
            recency = 1 - (last_date - df.index[rally_idx]).days / span_days
            score = body / a + 0.5 * touches + recency
            found.append({
                "low": low, "high": high, "mid": mid, "touches": touches,
                "score": score, "type": "Support", "note": "",
                "distance_pct": (mid - last_close) / last_close * 100,
                "formation": formation,
            })
    # De-duplicate overlapping bases: keep the strongest per price area.
    found.sort(key=lambda z: z["score"], reverse=True)
    kept = []
    for z in found:
        if all(abs(z["mid"] - k["mid"]) > 0.6 * a for k in kept):
            kept.append(z)
        if len(kept) >= max_zones:
            break
    kept.sort(key=lambda z: z["mid"], reverse=True)
    return kept


def build_zones(df: pd.DataFrame, window: int, max_zones: int):
    """Cluster pivot prices into zones and rank them."""
    pivots = find_pivots(df, window)
    if not pivots:
        return []

    a = atr(df)
    tolerance = 0.6 * a          # pivots this close belong to the same zone
    pad = 0.25 * a               # minimum half-thickness of a zone
    last_close = float(df["Close"].iloc[-1])
    last_date = df.index[-1]
    span_days = max((last_date - df.index[0]).days, 1)

    pivots.sort(key=lambda p: p[1])
    clusters, current = [], [pivots[0]]
    for p in pivots[1:]:
        if p[1] - np.mean([q[1] for q in current]) <= tolerance:
            current.append(p)
        else:
            clusters.append(current)
            current = [p]
    clusters.append(current)

    # Merge clusters whose padded bands would overlap
    merged = [clusters[0]]
    for c in clusters[1:]:
        prev_top = max(max(p[1] for p in merged[-1]), np.mean([p[1] for p in merged[-1]]) + pad)
        this_bot = min(min(p[1] for p in c), np.mean([p[1] for p in c]) - pad)
        if this_bot <= prev_top:
            merged[-1] = merged[-1] + c
        else:
            merged.append(c)
    clusters = merged

    zones = []
    for c in clusters:
        prices = [p[1] for p in c]
        mid = float(np.mean(prices))
        low = min(min(prices), mid - pad)
        high = max(max(prices), mid + pad)
        touches = len(c)
        # Recent touches count a bit more than old ones
        recency = np.mean([1 - (last_date - p[0]).days / span_days for p in c])
        score = touches + recency
        if low <= last_close <= high:
            kind = "Support" if mid <= last_close else "Resistance"
            note = "price inside zone"
        else:
            kind = "Support" if high < last_close else "Resistance"
            note = ""
        zones.append({
            "low": low, "high": high, "mid": mid, "touches": touches,
            "score": score, "type": kind, "note": note,
            "distance_pct": (mid - last_close) / last_close * 100,
            "formation": "",
        })

    # Demand zones follow the RBR/DBR rule: a tight base (max 3 candles)
    # then a long green rally candle. Those zones lead the Support side;
    # pivot-cluster zones fill any remaining Support slots.
    rbr_dbr = find_rbr_dbr_zones(df, max_zones)
    zones.sort(key=lambda z: z["score"], reverse=True)
    pivot_supports = [z for z in zones if z["type"] == "Support"]
    resists = [z for z in zones if z["type"] == "Resistance"]
    half = max_zones // 2
    supports = rbr_dbr + [z for z in pivot_supports
                          if all(abs(z["mid"] - k["mid"]) > 0.6 * atr(df)
                                 for k in rbr_dbr)]
    chosen = supports[:half] + resists[:half]
    leftovers = [z for z in zones if z not in chosen]
    leftovers += [z for z in rbr_dbr if z not in chosen]
    chosen += leftovers[:max_zones - len(chosen)]
    chosen.sort(key=lambda z: z["mid"], reverse=True)
    return chosen


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def draw_candles(ax, df: pd.DataFrame):
    x = np.arange(len(df))
    up = df["Close"] >= df["Open"]
    colors = np.where(up, "#26A69A", "#EF5350")
    ax.vlines(x, df["Low"], df["High"], color=colors, linewidth=0.8, zorder=3)
    body_low = df[["Open", "Close"]].min(axis=1).values
    body_h = (df["Close"] - df["Open"]).abs().values
    min_h = (df["High"].max() - df["Low"].min()) * 0.0015
    for xi, bl, bh, col in zip(x, body_low, body_h, colors):
        ax.add_patch(Rectangle((xi - 0.35, bl), 0.7, max(bh, min_h),
                               facecolor=col, edgecolor=col, zorder=4))
    # Date labels
    step = max(len(df) // 10, 1)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([d.strftime("%b %y") for d in df.index[::step]],
                       rotation=0, fontsize=8)
    ax.set_xlim(-1, len(df) + 8)


def draw_zones(ax, df: pd.DataFrame, zones):
    n = len(df)
    for z in zones:
        color = SUPPORT_COLOR if z["type"] == "Support" else RESIST_COLOR
        ax.axhspan(z["low"], z["high"], facecolor=color, alpha=ZONE_ALPHA,
                   edgecolor="none", zorder=1)
        for edge in (z["low"], z["high"]):
            ax.axhline(edge, color=color, alpha=EDGE_ALPHA, linewidth=1.0, zorder=2)
        ax.text(n + 0.5, z["mid"], f"{z['low']:.2f}–{z['high']:.2f} ({z['touches']}x)"
                + (f" {z['formation']}" if z.get("formation") else ""),
                va="center", fontsize=7, color="#333333", zorder=5)


def draw_emas(ax, df: pd.DataFrame, emas):
    x = np.arange(len(df))
    for length in emas:
        col = f"EMA{length}"
        if col in df and df[col].notna().any():
            ax.plot(x, df[col].values, color=EMA_COLORS[length], linewidth=1.4,
                    label=f"EMA({length})", zorder=6)


def plot_timeframe(ax, ticker, tf_name, df, zones, emas):
    draw_zones(ax, df, zones)
    draw_candles(ax, df)
    draw_emas(ax, df, emas)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    last = df["Close"].iloc[-1]
    ax.axhline(last, color="#555555", linestyle="--", linewidth=0.8, zorder=2)
    ax.set_title(f"{ticker} – {tf_name}  |  last close {last:.2f}", fontsize=11, loc="left")
    ax.grid(alpha=0.25)
    ax.set_ylabel("Price")


def prepare(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add EMAs on the full history, then trim to the display window."""
    df = df.copy()
    for length in cfg["emas"]:
        df[f"EMA{length}"] = df["Close"].ewm(span=length, adjust=False).mean()
    start = df.index[-1] - pd.DateOffset(years=cfg["display_years"])
    return df[df.index > start]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Daily (1Y), weekly (5Y) and monthly (10Y) support/resistance zone charts with EMAs")
    parser.add_argument("tickers", nargs="*", default=["SPY", "QQQ"],
                        help=f"1 to {MAX_TICKERS} ticker symbols (default: SPY QQQ)")
    parser.add_argument("--outdir", default="charts", help="folder for the PNG charts")
    parser.add_argument("--no-show", action="store_true",
                        help="only save the PNGs, don't open chart windows")
    parser.add_argument("--demo", action="store_true", help="use synthetic data (no internet)")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]
    if len(tickers) > MAX_TICKERS:
        parser.error(f"at most {MAX_TICKERS} tickers allowed (got {len(tickers)})")

    warnings.filterwarnings("ignore")
    if args.no_show:
        plt.switch_backend("Agg")      # render off-screen only
    os.makedirs(args.outdir, exist_ok=True)
    loader = synthetic_data if args.demo else fetch_data

    # 1) Download all data first, so nothing is drawn until everything is ready
    data = {}
    for ticker in tickers:
        for tf_name, cfg in TIMEFRAMES.items():
            try:
                raw = loader(ticker, cfg["fetch_period"], cfg["interval"])
            except Exception as exc:
                sys.exit(f"Could not download {ticker} {tf_name} data: {exc}")
            data[(ticker, tf_name)] = prepare(raw, cfg)

    # 2) Draw and save one chart per ticker
    for ticker in tickers:
        fig, axes = plt.subplots(3, 1, figsize=(15, 16), num=f"{ticker} zones")
        for ax, (tf_name, cfg) in zip(axes, TIMEFRAMES.items()):
            df = data[(ticker, tf_name)]
            zones = build_zones(df, cfg["pivot_window"], cfg["max_zones"])
            plot_timeframe(ax, ticker, tf_name, df, zones, cfg["emas"])

        handles = [Rectangle((0, 0), 1, 1, color=SUPPORT_COLOR, alpha=0.6),
                   Rectangle((0, 0), 1, 1, color=RESIST_COLOR, alpha=0.6)]
        fig.legend(handles, ["Support zone", "Resistance zone"],
                   loc="upper right", fontsize=9)
        fig.suptitle(f"{ticker} – Daily (1Y), Weekly (5Y) & Monthly (10Y) Key Zones", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(args.outdir, f"{ticker}_zones.png"), dpi=130)

    # 3) Show all chart windows; the program ends when you close them
    if not args.no_show:
        plt.show()



if __name__ == "__main__":
    main()
