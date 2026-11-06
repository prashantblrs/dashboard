"""Streamlit page: enter one ticker, get the daily + weekly support/resistance chart.
Uses spy_qqq_levels.py exactly as supplied; this file only adds the input box and shows the picture."""
import io
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

import spy_qqq_levels as lv

st.set_page_config(page_title="Demand / Supply Levels", layout="wide")
st.title("Demand / supply levels")
st.caption("Daily (1Y) and weekly (2Y) support and resistance zones with SMA 21 / 50 / 200.")


@st.cache_data(ttl=900, show_spinner=False)
def make_chart(ticker: str) -> bytes:
    warnings.filterwarnings("ignore")
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


with st.form("f"):
    ticker = st.text_input("Ticker", value="SPY", max_chars=10).strip().upper()
    go = st.form_submit_button("Show chart")

if go or ticker:
    if not ticker.replace("-", "").replace(".", "").replace("^", "").isalnum():
        st.error("Enter a valid ticker symbol, for example SPY or AAPL.")
    else:
        try:
            with st.spinner(f"Loading {ticker}…"):
                png = make_chart(ticker)
            st.image(png, width="stretch")
            st.download_button("Download PNG", png, file_name=f"{ticker}_zones.png", mime="image/png")
        except Exception as exc:
            st.error(f"Could not build the chart for {ticker}: {exc}")
