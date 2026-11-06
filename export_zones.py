#!/usr/bin/env python3
"""Regenerate the GitHub Pages assets for the levels dashboard.

For each ticker: builds the combined Daily (1Y) + Weekly (5Y) + Monthly (10Y)
chart (spy_qqq_levels logic, unchanged), saves the PNG
to docs/charts/, and writes the zone data to docs/zones.json for the page.

Tickers come from tickers.json (default SPY, QQQ, IWM, DIA).

Run:  python3 export_zones.py                       (all tickers in tickers.json)
      python3 export_zones.py --tickers NVDA,META   (only those tickers;
                                                     merges into the existing
                                                     zones.json and adds them
                                                     to tickers.json so the
                                                     nightly refresh keeps them)

Results are MERGED into the existing docs/zones.json, never wiped:
tickers already in zones.json keep their data unless regenerated here.
"""
import argparse
import io
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import spy_qqq_levels as lv

BASE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "docs")
CHART_DIR = os.path.join(OUTDIR, "charts")
TICKERS_FILE = os.path.join(BASE, "tickers.json")
ZONES_FILE = os.path.join(OUTDIR, "zones.json")
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA"]


def load_ticker_list():
    try:
        with open(TICKERS_FILE) as f:
            tickers = json.load(f).get("tickers") or []
        return [str(t).upper().strip() for t in tickers if str(t).strip()]
    except FileNotFoundError:
        return list(DEFAULT_TICKERS)
    except (json.JSONDecodeError, AttributeError):
        return list(DEFAULT_TICKERS)


def save_ticker_list(tickers):
    with open(TICKERS_FILE, "w") as f:
        json.dump({"tickers": tickers}, f, indent=2)
        f.write("\n")


def make_chart(ticker: str) -> bytes:
    fig, axes = plt.subplots(3, 1, figsize=(15, 16))
    for ax, (tf_name, cfg) in zip(axes, lv.TIMEFRAMES.items()):
        df = lv.prepare(lv.fetch_data(ticker, cfg["fetch_period"], cfg["interval"]), cfg)
        zones = lv.build_zones(df, cfg["pivot_window"], cfg["max_zones"])
        lv.plot_timeframe(ax, ticker, tf_name, df, zones, cfg["smas"])
    handles = [Rectangle((0, 0), 1, 1, color=lv.SUPPORT_COLOR, alpha=0.6),
               Rectangle((0, 0), 1, 1, color=lv.RESIST_COLOR, alpha=0.6)]
    fig.legend(handles, ["Support zone", "Resistance zone"], loc="upper right", fontsize=9)
    fig.suptitle(f"{ticker} – Daily (1Y), Weekly (5Y) & Monthly (10Y) Key Zones", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue()


def zone_json(ticker: str) -> dict:
    out = {}
    for tf_name, cfg in lv.TIMEFRAMES.items():
        df = lv.prepare(lv.fetch_data(ticker, cfg["fetch_period"], cfg["interval"]), cfg)
        zones = lv.build_zones(df, cfg["pivot_window"], cfg["max_zones"])
        out[tf_name] = {
            "last_close": round(float(df["Close"].iloc[-1]), 2),
            "as_of": str(df.index[-1].date()),
            "zones": [
                {
                    "type": z["type"],
                    "low": round(z["low"], 2),
                    "high": round(z["high"], 2),
                    "touches": z["touches"],
                    "distance_pct": round(z["distance_pct"], 2),
                    "note": z["note"],
                }
                for z in zones
            ],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Regenerate levels charts and zones")
    parser.add_argument("--tickers", default="",
                        help="Comma-separated tickers to generate (e.g. NVDA,META). "
                             "If omitted, all tickers in tickers.json are generated.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    os.makedirs(CHART_DIR, exist_ok=True)

    if args.tickers.strip():
        tickers = [t.upper().strip() for t in args.tickers.split(",") if t.strip()]
        # New tickers join the standing list so nightly refresh keeps them.
        standing = load_ticker_list()
        changed = False
        for t in tickers:
            if t not in standing:
                standing.append(t)
                changed = True
        if changed:
            save_ticker_list(standing)
            print(f"tickers.json updated: {standing}")
    else:
        tickers = load_ticker_list()

    # Merge into the existing zones.json; never drop tickers we are not
    # regenerating in this run.
    try:
        with open(ZONES_FILE) as f:
            data = json.load(f)
        if not isinstance(data.get("tickers"), dict):
            data["tickers"] = {}
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"tickers": {}}

    ok = 0
    for ticker in tickers:
        try:
            png = make_chart(ticker)
            with open(os.path.join(CHART_DIR, f"{ticker}_zones.png"), "wb") as f:
                f.write(png)
            data["tickers"][ticker] = zone_json(ticker)
            ok += 1
            print(f"{ticker}: chart + zones written")
        except Exception as exc:
            print(f"{ticker}: skipped — {exc}", file=sys.stderr)

    if not ok:
        print("No tickers generated — check the ticker symbols.", file=sys.stderr)
        sys.exit(1)

    data["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(ZONES_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote docs/zones.json ({data['generated_at_utc']})")


if __name__ == "__main__":
    main()
