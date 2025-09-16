import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.title("Economic Crisis Detection Dashboard")

uploaded_file = st.file_uploader("Upload dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    features = ["Employment Sector: Agriculture","Employment Sector: Industry",
                "Employment Sector: Services","Unemployment Rate","GDP (in USD)"]

    model = IsolationForest(contamination=0.05, random_state=42)
    df["Anomaly"] = model.fit_predict(df[features].fillna(0))

    fig = px.scatter(df, x="Year", y="GDP (in USD)", 
                     color=df["Anomaly"].map({1:"Normal",-1:"Crisis"}),
                     hover_data=["Country Name","Unemployment Rate"])
    st.plotly_chart(fig)
