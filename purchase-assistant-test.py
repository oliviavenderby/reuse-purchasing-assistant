# find_next_buy.py
# ReUseBricks LEGO Investment Assistant — Optimized Basket Version
# Full code — ready to run in Streamlit

from __future__ import annotations
import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ReUseBricks Investment Assistant", page_icon="🧱", layout="wide")
st.title("🧱 ReUseBricks — Investment Assistant")

# ---------------------- Utility Functions ----------------------

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

def dollar_round(x: float) -> int:
    try:
        return int(max(0, round(float(x))))
    except Exception:
        return 0

def hybrid_utility(row, a_score=0.6, b_roi=0.3, c_profit=0.1):
    score = float(row.get("score", 0.0)) / 100.0
    roi   = float(row.get("sealed_roi_pct", 0.0))
    profit = float(row.get("current_new_price", 0.0) - row.get("target_buy_price", 0.0))
    return max(0.0, a_score*score + b_roi*roi + c_profit*(profit / 100.0))

# ---------------------- Sidebar ----------------------

with st.sidebar:
    st.subheader("Load Data")
    file = st.file_uploader("Upload CSV", type=["csv"])
    st.caption("Must include all required columns for scoring.")

    st.markdown("---")
    st.subheader("Weights")
    w_growth = st.slider("Growth", 0.0, 1.0, 0.35, 0.01)
    w_liq    = st.slider("Liquidity", 0.0, 1.0, 0.25, 0.01)
    w_margin = st.slider("Margin", 0.0, 1.0, 0.20, 0.01)
    w_sent   = st.slider("Sentiment", 0.0, 1.0, 0.10, 0.01)
    w_risk   = st.slider("Risk penalty", 0.0, 1.0, 0.10, 0.01)

    desired_discount_pct = st.slider("Target discount off retail (%)", 0, 50, 20, 1)
    show_debug = st.checkbox("Debug columns", value=False)

# ---------------------- Load ----------------------

def load_dataset(upload):
    if upload:
        return pd.read_csv(upload)
    st.error("Please upload a CSV file.")
    st.stop()

df = load_dataset(file)

# Required columns
required_cols = [
    "set_num","name","theme","year","pieces","minifigs","retail_price",
    "current_value_new","current_value_used",
    "forecast_growth_2y_pct","forecast_growth_5y_pct",
    "growth_last_year_pct","growth_12m_pct",
    "bl_current_avg_new","bl_qty_new","bl_lots_new","bl_current_avg_used","bl_qty_used","bl_lots_used",
    "bl_sold_qty_new","bl_times_sold_new","bl_sold_qty_used","bl_times_sold_used",
    "brickset_rating","users_owned","users_wanted",
    "sealed_roi_pct","part_out_roi_pct","rerelease_risk","units_sold_30d","part_out_value"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ---------------------- Feature Engineering ----------------------

today_year = datetime.utcnow().year
df["age_years"] = (today_year - pd.to_numeric(df["year"], errors="coerce")).clip(lower=0)

df["available_qty"] = df["bl_qty_new"].fillna(0) + df["bl_qty_used"].fillna(0)
df["sold_qty_total"] = df["bl_sold_qty_new"].fillna(0) + df["bl_sold_qty_used"].fillna(0)
df["sell_through"] = df.apply(lambda r: safe_pct(r["sold_qty_total"], r["sold_qty_total"] + r["available_qty"]), axis=1)

df["margin_new_vs_retail"] = (df["bl_current_avg_new"] - df["retail_price"]) / df["retail_price"]
df["margin_used_vs_retail"] = (df["bl_current_avg_used"] - df["retail_price"]) / df["retail_price"]

df["target_buy_price"] = df["retail_price"] * (1 - desired_discount_pct/100.0)
df["roi_new_vs_target"] = (df["bl_current_avg_new"] - df["target_buy_price"]) / df["target_buy_price"]

df["competition_lots"] = df["bl_lots_new"].fillna(0) + df["bl_lots_used"].fillna(0)
df["volatility_proxy"] = (df["growth_last_year_pct"] - df["growth_12m_pct"]).abs()

df["wanted_owned_ratio"] = df["users_wanted"] / (df["users_owned"] + 1)

# ---------------------- Subscores ----------------------

growth_sub = (
    0.4 * zscore(df["forecast_growth_2y_pct"]) +
    0.4 * zscore(df["forecast_growth_5y_pct"]) +
    0.2 * zscore(df["growth_12m_pct"]) +
    0.1 * zscore(df["age_years"])
)

liquidity_sub = (
    0.5 * zscore(df["sell_through"]) +
    0.3 * zscore(df["sold_qty_total"]) +
    0.2 * zscore(df["bl_times_sold_new"] + df["bl_times_sold_used"])
)

margin_sub = (
    0.6 * zscore(df["margin_new_vs_retail"]) +
    0.4 * zscore(df["margin_used_vs_retail"])
)

sentiment_sub = (
    0.6 * zscore(df["brickset_rating"]) +
    0.4 * zscore(df["wanted_owned_ratio"])
)

risk_sub = (
    0.6 * zscore(df["competition_lots"]) +
    0.4 * zscore(df["volatility_proxy"])
)

raw = (
    w_growth * growth_sub +
    w_liq    * liquidity_sub +
    w_margin * margin_sub +
    w_sent   * sentiment_sub -
    w_risk   * risk_sub
)

df["score"] = minmax_0_100(raw)

# ---------------------- Basket Optimizer ----------------------

def suggest_basket_optimized(
    df: pd.DataFrame,
    budget_total: float,
    *,
    objective_weights=(0.6, 0.3, 0.1),
    theme_cap: int = 2,
    min_liquidity_30d: int = 0,
    max_avg_risk: float = 0.30,
    prefer_sealed: bool = True,
    require_roi_floor: float = 0.0
) -> Tuple[pd.DataFrame, float]:
    if df.empty:
        return pd.DataFrame(columns=df.columns), 0.0

    roi_col = "sealed_roi_pct" if prefer_sealed else "part_out_roi_pct"

    mask = (
        df["target_buy_price"].fillna(0) > 0
    ) & (
        df["units_sold_30d"].fillna(0) >= min_liquidity_30d
    ) & (
        df[roi_col].fillna(-9e9) >= require_roi_floor
    )
    items = df.loc[mask].copy()
    if items.empty:
        return pd.DataFrame(columns=df.columns), 0.0

    prices = items["target_buy_price"].astype(float).map(dollar_round).tolist()
    themes  = items["theme"].fillna("Unknown").tolist()
    risks   = items["rerelease_risk"].fillna(0.15).astype(float).tolist()

    util = []
    a,b,c = objective_weights
    for _, r in items.iterrows():
        util.append(hybrid_utility(r, a_score=a, b_roi=b, c_profit=c))
    values = util

    W = dollar_round(budget_total)
    n = len(values)
    if W <= 0:
        return pd.DataFrame(columns=df.columns), 0.0

    dp = [0.0] * (W + 1)
    choose = [[False]*n for _ in range(W + 1)]

    for i in range(n):
        w_i = prices[i]
        v_i = values[i]
        if w_i <= 0:
            continue
        for w in range(W, w_i-1, -1):
            cand = dp[w - w_i] + v_i
            if cand > dp[w]:
                dp[w] = cand
                choose[w] = choose[w - w_i].copy()
                choose[w][i] = True

    w_star = max(range(W+1), key=lambda w: dp[w])
    chosen_idx = [i for i,flag in enumerate(choose[w_star]) if flag]

    sel = items.iloc[chosen_idx].copy()

    from collections import Counter
    themed = Counter(sel["theme"])
    if any(c > theme_cap for c in themed.values()):
        sel["price_$"] = sel["target_buy_price"].astype(float)
        sel["util"] = sel.apply(lambda r: hybrid_utility(r, a_score=a, b_roi=b, c_profit=c), axis=1)
        sel["u_per_$"] = sel["util"] / sel["price_$"].replace(0, np.nan)
        keep = []
        from collections import defaultdict
        by_theme = defaultdict(list)
        for idx, r in sel.iterrows():
            by_theme[r["theme"]].append((idx, r["u_per_$"]))
        for t, lst in by_theme.items():
            lst_sorted = sorted(lst, key=lambda x: x[1], reverse=True)
            keep.extend([idx for idx,_ in lst_sorted[:theme_cap]])
        sel = sel.loc[sorted(set(keep))].copy()

    if not sel.empty and float(sel["rerelease_risk"].mean()) > max_avg_risk:
        sel["u_per_$"] = sel.apply(lambda r: hybrid_utility(r, a_score=a, b_roi=b, c_profit=c), axis=1) / sel["target_buy_price"]
        while not sel.empty and float(sel["rerelease_risk"].mean()) > max_avg_risk:
            drop_idx = sel.sort_values("u_per_$", ascending=True).index[0]
            sel = sel.drop(index=drop_idx)

    spend = float(sel["target_buy_price"].sum())
    return sel.sort_values("score", ascending=False), spend

# ---------------------- Basket UI ----------------------

st.subheader("Suggested basket under your budget")

budget_total = st.number_input("Total budget", min_value=0.0, value=500.0, step=50.0)

col1, col2, col3 = st.columns(3)
with col1:
    theme_cap = st.number_input("Max sets per theme", min_value=1, value=2, step=1)
with col2:
    min_liq = st.number_input("Min 30-day units sold", min_value=0, value=50, step=10)
with col3:
    max_avg_risk = st.slider("Max average risk", 0.0, 1.0, 0.30, 0.01)

col4, col5, col6 = st.columns(3)
with col4:
    prefer_path = st.radio("ROI path", ["Sealed", "Part-out"], horizontal=True, index=0)
with col5:
    roi_floor = st.slider("Per-set ROI floor", 0.0, 1.0, 0.00, 0.01)
with col6:
    st.write("Objective mix")
    wS = st.slider("Score", 0.0, 1.0, 0.60, 0.05)
    wR = st.slider("ROI",   0.0, 1.0, 0.30, 0.05)
    wP = st.slider("Profit",0.0, 1.0, 0.10, 0.05)

basket, spend = suggest_basket_optimized(
    df,
    budget_total=budget_total,
    objective_weights=(wS, wR, wP),
    theme_cap=int(theme_cap),
    min_liquidity_30d=int(min_liq),
    max_avg_risk=float(max_avg_risk),
    prefer_sealed=(prefer_path == "Sealed"),
    require_roi_floor=float(roi_floor)
)

if basket.empty:
    st.info("No combination met the constraints.")
else:
    if prefer_path == "Sealed":
        basket["est_profit"] = basket["current_new_price"] - basket["target_buy_price"]
    else:
        basket["est_profit"] = 0.7 * basket["part_out_value"] - basket["target_buy_price"]

    total_profit = float(basket["est_profit"].sum())
    est_roi_pct = total_profit / spend if spend > 0 else 0.0

    view = basket[["score","set_num","name","theme","target_buy_price","est_profit","rerelease_risk"]].copy()
    view["target_buy_price"] = view["target_buy_price"].map(lambda x: f"${x:,.2f}")
    view["est_profit"] = view["est_profit"].map(lambda x: f"${x:,.2f}")
    st.dataframe(view.reset_index(drop=True), use_container_width=True, height=280)

    c1, c2, c3 = st.columns(3)
    c1.metric("Planned spend", f"${spend:,.2f}")
    c2.metric("Estimated profit", f"${total_profit:,.2f}")
    c3.metric("Estimated ROI (basket)", f"{est_roi_pct*100:.1f}%")
