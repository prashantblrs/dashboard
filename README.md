# Demand / supply levels

Daily (1Y) and weekly (2Y) support/resistance charts with SMA 21 / 50 / 200.
`spy_qqq_levels.py` is the zone engine: zones are found from swing highs/lows
(pivots), clustered by price, and drawn as light bands — green is support,
red is resistance.

## GitHub Pages dashboard

The charts are published as a static site from this repo:
`docs/index.html` (tickers SPY, QQQ, IWM, DIA) reads `docs/zones.json` and shows
the PNGs in `docs/charts/`. `export_zones.py` regenerates the charts and the
zone data; `.github/workflows/update-charts.yml` runs it every weekday after
the US close and commits the refresh.

To host it: repo Settings -> Pages -> Build and deployment -> Deploy from a
branch -> branch `main`, folder `/docs`. The site is then live at
https://prashantblrs.github.io/dashboard/ (note: on a free GitHub account,
Pages requires the repo to be public; a private repo needs a paid plan).

Regenerate locally: `pip install -r requirements.txt && python3 export_zones.py`
