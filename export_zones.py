#!/usr/bin/env python3
"""Regenerate the GitHub Pages assets for the levels dashboard.

For each ticker in TICKERS: builds the combined Daily (1Y) + Weekly (2Y)
chart (spy_qqq_levels logic, unchanged), saves the PNG
to docs/charts/, and writes the zone data to docs/zones.json for the page.

Run:  python3 export_zones.py            (from the repo root)
"""
import io
import json
import os
import warnings
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import spy_qqq_levels as lv

TICKERS = ["SPY", "QQQ", "IWM", "DIA"]
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
CHART_DIR = os.path.join(OUTDIR, "charts")


def make_chart(ticker: str) -> bytes:
    fig, axes = plt.subplots(2, 1, figsize=(15, 11))
    for ax, (tf_name, cfg) in zip(axes, lv.TIMEFRAMES.items()):
        df = lv.prepare(lv.fetch_data(ticker, cfg["fetch_period"], cfg["interval"]), cfg)
        zones = lv.build_zones(df, cfg["pivot_window"], cfg["max_zones"])
        lv.plot_timeframe(ax, ticker, tf_name, df, zones, cfg["smas"])
    handles = [Rectangle((0, 0), 1, 1, color=lv.SUPPORT_COLOR, alpha=0.6),
               Rectangle((0, 0), 1, 1, color=lv.RESIST_COLOR, alpha=0.6)]
    fig.legend(handles, ["Support zone", "Resistance zone"], loc="upper right", fontsize=9)
    fig.suptitle(f"{ticker} – Daily (1Y) & Weekly (2Y) Key Zones", fontsize=14)
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
    warnings.filterwarnings("ignore")
    os.makedirs(CHART_DIR, exist_ok=True)
    data = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tickers": {}}
    for ticker in TICKERS:
        png = make_chart(ticker)
        with open(os.path.join(CHART_DIR, f"{ticker}_zones.png"), "wb") as f:
            f.write(png)
        data["tickers"][ticker] = zone_json(ticker)
        print(f"{ticker}: chart + zones written")
    with open(os.path.join(OUTDIR, "zones.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote docs/zones.json ({data['generated_at_utc']})")


if __name__ == "__main__":
    main()
