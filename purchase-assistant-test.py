# invest_score_demo.py
# Multi-factor LEGO investment score demo (uses BrickEconomy + BrickLink + Brickset metrics)
#
# Quick start:
#   pip install streamlit pandas numpy
#   streamlit run invest_score_demo.py
#
# By default it loads 'combined_sets.csv' (from the sample I generated).
# You can also upload your own CSV with the same columns.

from __future__ import annotations
import math
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="LEGO Investment Score — Demo", page_icon="🧱", layout="wide")
st.title("🧱 LEGO Investment Score — Demo")
st.caption("Deterministic, explainable scoring from BrickEconomy + BrickLink + Brickset style metrics (no LLM).")

DEFAULT_FILE = "combined_sets.csv"  # change if you move the sample

# ----------------------- Helpers -----------------------

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def minmax_0_100(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.full(len(s), 50.0), index=s.index)
    return (s - lo) / (hi - lo) * 100.0

def safe_pct(numer, denom) -> float:
    try:
        n = float(numer)
        d = float(denom)
        return n / d if d > 0 else 0.0
    except Exception:
        return 0.0

# ----------------------- Sidebar -----------------------

with st.sidebar:
    st.subheader("Load data")
    file = st.file_uploader("Upload a combined CSV", type=["csv"])
    st.caption("If omitted, the app tries to load 'combined_sets.csv' next to this script.")

    st.markdown("---")
    st.subheader("Weights")
    w_growth = st.slider("Growth", 0.0, 1.0, 0.35, 0.01)
    w_liq    = st.slider("Liquidity", 0.0, 1.0, 0.25, 0.01)
    w_margin = st.slider("Margin", 0.0, 1.0, 0.20, 0.01)
    w_sent   = st.slider("Sentiment", 0.0, 1.0, 0.10, 0.01)
    w_risk   = st.slider("Risk penalty", 0.0, 1.0, 0.10, 0.01)

    st.markdown("---")
    st.subheader("Buy assumptions")
    desired_discount_pct = st.slider("Desired discount off retail for buys", 0, 50, 20, 1)
    st.caption("Used only in ROI displays; core score uses normalized metrics.")

    show_debug = st.checkbox("Debug columns", value=False)

# ----------------------- Load -----------------------

def load_dataset(upload):
    if upload:
        return pd.read_csv(upload)
    try:
        return pd.read_csv(DEFAULT_FILE)
    except Exception:
        st.error("Couldn't load data. Upload a CSV with the expected columns.")
        st.stop()

df = load_dataset(file)

required_cols = [
    "set_num","name","theme","year","pieces","minifigs","retail_price",
    "current_value_new","current_value_used",
    "forecast_new_2y","forecast_new_5y","forecast_growth_2y_pct","forecast_growth_5y_pct",
    "growth_last_year_pct","growth_12m_pct",
    "bl_current_avg_new","bl_qty_new","bl_lots_new","bl_current_avg_used","bl_qty_used","bl_lots_used",
    "bl_6mo_avg_new","bl_sold_qty_new","bl_times_sold_new","bl_6mo_avg_used","bl_sold_qty_used","bl_times_sold_used",
    "brickset_rating","users_owned","users_wanted"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ----------------------- Feature Engineering -----------------------

# Age
today_year = datetime.utcnow().year
df["age_years"] = (today_year - pd.to_numeric(df["year"], errors="coerce")).clip(lower=0)

# Sell-through rate (rough): sold over available inventory
df["available_qty"] = pd.to_numeric(df["bl_qty_new"], errors="coerce").fillna(0) + \
                      pd.to_numeric(df["bl_qty_used"], errors="coerce").fillna(0)
df["sold_qty_total"] = pd.to_numeric(df["bl_sold_qty_new"], errors="coerce").fillna(0) + \
                       pd.to_numeric(df["bl_sold_qty_used"], errors="coerce").fillna(0)
df["sell_through"] = df.apply(lambda r: safe_pct(r["sold_qty_total"], r["sold_qty_total"] + r["available_qty"]), axis=1)

# Margin proxies
df["margin_new_vs_retail"] = (pd.to_numeric(df["bl_current_avg_new"], errors="coerce") - pd.to_numeric(df["retail_price"], errors="coerce")) / pd.to_numeric(df["retail_price"], errors="coerce")
df["margin_used_vs_retail"] = (pd.to_numeric(df["bl_current_avg_used"], errors="coerce") - pd.to_numeric(df["retail_price"], errors="coerce")) / pd.to_numeric(df["retail_price"], errors="coerce")

# ROI vs target buy (for display only)
df["target_buy_price"] = pd.to_numeric(df["retail_price"], errors="coerce") * (1 - desired_discount_pct/100.0)
df["roi_new_vs_target"] = (pd.to_numeric(df["bl_current_avg_new"], errors="coerce") - df["target_buy_price"]) / df["target_buy_price"]
df["roi_used_vs_target"] = (pd.to_numeric(df["bl_current_avg_used"], errors="coerce") - df["target_buy_price"]) / df["target_buy_price"]

# Competition & volatility proxies (risk)
df["competition_lots"] = pd.to_numeric(df["bl_lots_new"], errors="coerce").fillna(0) + pd.to_numeric(df["bl_lots_used"], errors="coerce").fillna(0)
df["volatility_proxy"] = (pd.to_numeric(df["growth_last_year_pct"], errors="coerce") - pd.to_numeric(df["growth_12m_pct"], errors="coerce")).abs()

# Sentiment proxies
df["wanted_owned_ratio"] = (pd.to_numeric(df["users_wanted"], errors="coerce")) / (pd.to_numeric(df["users_owned"], errors="coerce") + 1)
df["brickset_rating"] = pd.to_numeric(df["brickset_rating"], errors="coerce")

# ----------------------- Subscores -----------------------

# Growth: long-term + short-term + age tailwind
growth_sub = (
    0.4 * zscore(pd.to_numeric(df["forecast_growth_2y_pct"], errors="coerce")) +
    0.4 * zscore(pd.to_numeric(df["forecast_growth_5y_pct"], errors="coerce")) +
    0.2 * zscore(pd.to_numeric(df["growth_12m_pct"], errors="coerce")) +
    0.1 * zscore(pd.to_numeric(df["age_years"], errors="coerce"))  # small weight for EOL proximity
)

# Liquidity: sell-through + sold qty + times sold
liquidity_sub = (
    0.5 * zscore(df["sell_through"]) +
    0.3 * zscore(pd.to_numeric(df["sold_qty_total"], errors="coerce")) +
    0.2 * zscore(pd.to_numeric(df["bl_times_sold_new"], errors="coerce") + pd.to_numeric(df["bl_times_sold_used"], errors="coerce"))
)

# Margin: current market premium vs retail (favor new a bit more for sealed investors)
margin_sub = (
    0.6 * zscore(df["margin_new_vs_retail"]) +
    0.4 * zscore(df["margin_used_vs_retail"])
)

# Sentiment: rating + demand > ownership
sentiment_sub = (
    0.6 * zscore(df["brickset_rating"]) +
    0.4 * zscore(df["wanted_owned_ratio"])
)

# Risk: competition + volatility (higher is worse)
risk_sub = (
    0.6 * zscore(df["competition_lots"]) +
    0.4 * zscore(df["volatility_proxy"])
)

# ----------------------- Final Score -----------------------

raw = (
    w_growth * growth_sub +
    w_liq    * liquidity_sub +
    w_margin * margin_sub +
    w_sent   * sentiment_sub -
    w_risk   * risk_sub
)

df["score"] = minmax_0_100(raw)

# ----------------------- Explanations (rule-based) -----------------------

med = {
    "forecast_growth_5y_pct": pd.to_numeric(df["forecast_growth_5y_pct"], errors="coerce").median(),
    "sell_through": df["sell_through"].median(),
    "margin_new_vs_retail": df["margin_new_vs_retail"].median(),
    "brickset_rating": df["brickset_rating"].median(),
    "wanted_owned_ratio": df["wanted_owned_ratio"].median(),
    "competition_lots": df["competition_lots"].median(),
    "volatility_proxy": df["volatility_proxy"].median(),
}

def reasons(row: pd.Series) -> List[str]:
    msgs = []
    if row["forecast_growth_5y_pct"] >= med["forecast_growth_5y_pct"]:
        msgs.append("Strong long-term appreciation forecast (5y).")
    if row["sell_through"] >= med["sell_through"]:
        msgs.append("Healthy sell-through (demand > supply).")
    if row["margin_new_vs_retail"] >= med["margin_new_vs_retail"]:
        msgs.append("Attractive market premium vs retail (new).")
    if row["brickset_rating"] >= med["brickset_rating"]:
        msgs.append("Well-reviewed by the community.")
    if row["wanted_owned_ratio"] >= med["wanted_owned_ratio"]:
        msgs.append("High wanted/owned ratio (latent demand).")
    if row["competition_lots"] <= med["competition_lots"]:
        msgs.append("Lower competition (fewer active lots).")
    if row["volatility_proxy"] <= med["volatility_proxy"]:
        msgs.append("Stable recent growth (lower volatility).")
    if not msgs:
        msgs.append("Balanced profile across growth, liquidity, and margin.")
    return msgs

# ----------------------- Display -----------------------

st.subheader("Ranked sets")
view_cols = [
    "score","set_num","name","theme","year","retail_price",
    "bl_current_avg_new","bl_current_avg_used",
    "forecast_growth_2y_pct","forecast_growth_5y_pct",
    "growth_last_year_pct","growth_12m_pct",
    "sell_through","margin_new_vs_retail","brickset_rating","wanted_owned_ratio"
]
show = df[view_cols].sort_values("score", ascending=False).copy()
show["score"] = show["score"].round(1)
for c in ["retail_price","bl_current_avg_new","bl_current_avg_used"]:
    show[c] = show[c].map(lambda x: f"${x:,.2f}")
for p in ["forecast_growth_2y_pct","forecast_growth_5y_pct","growth_last_year_pct","growth_12m_pct","sell_through","margin_new_vs_retail","wanted_owned_ratio"]:
    show[p] = show[p].map(lambda x: f"{x*100:.1f}%" if abs(x) < 10 else f"{x:.2f}") if p in ["sell_through","margin_new_vs_retail"] else show[p].map(lambda x: f"{x:.2f}")

st.dataframe(show.reset_index(drop=True), use_container_width=True, height=420)

st.subheader("Why these scored well")
for _, row in df.sort_values("score", ascending=False).head(3).iterrows():
    with st.expander(f"#{row['set_num']} — {row['name']} (Score {row['score']:.1f})"):
        for m in reasons(row):
            st.write(f"• {m}")

if show_debug:
    st.markdown("---")
    st.write("Debug columns")
    st.dataframe(df, use_container_width=True)
