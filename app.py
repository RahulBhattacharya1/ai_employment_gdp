import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Economic Crisis Detection", layout="wide")
st.title("Economic Crisis Detection Dashboard")

REQUIRED_COLS = [
    "Country Name", "Year",
    "Employment Sector: Agriculture",
    "Employment Sector: Industry",
    "Employment Sector: Services",
    "Unemployment Rate",
    "GDP (in USD)"
]

contamination = st.sidebar.slider("Anomaly share (contamination)", 0.01, 0.20, 0.05, 0.01)
st.sidebar.caption("Tip: lower values = fewer years flagged as crisis.")

uploaded_file = st.file_uploader("Upload dataset (CSV with required columns)", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV to begin. Required columns: " + ", ".join(REQUIRED_COLS))
    st.stop()

# Load
df = pd.read_csv(uploaded_file)

# Validate columns
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Basic cleanup
df = df.copy()
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year"]).sort_values(["Country Name", "Year"])
feature_cols = [
    "Employment Sector: Agriculture",
    "Employment Sector: Industry",
    "Employment Sector: Services",
    "Unemployment Rate",
    "GDP (in USD)",
]

# Fill numeric gaps conservatively
df[feature_cols] = df[feature_cols].astype(float).fillna(0.0)

# Fit and score
model = IsolationForest(contamination=contamination, random_state=42)
df["Anomaly"] = model.fit_predict(df[feature_cols])

df["AnomalyFlag"] = df["Anomaly"].map({1: "Normal", -1: "Crisis"})

# Country filter
countries = ["All"] + sorted(df["Country Name"].unique().tolist())
sel = st.selectbox("Country", countries, index=0)
plot_df = df if sel == "All" else df[df["Country Name"] == sel]

# Plot
fig = px.scatter(
    plot_df,
    x="Year",
    y="GDP (in USD)",
    color="AnomalyFlag",
    hover_data=["Country Name", "Unemployment Rate"],
    title="GDP vs Year with Anomaly Flags"
)
st.plotly_chart(fig, use_container_width=True)

# Optional: show table of flagged rows
with st.expander("Show detected crisis years"):
    st.dataframe(
        plot_df[plot_df["AnomalyFlag"] == "Crisis"]
        .sort_values(["Country Name", "Year"])
        [["Country Name", "Year", "Unemployment Rate", "GDP (in USD)"]],
        use_container_width=True
    )
