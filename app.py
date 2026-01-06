import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Electricity Peak Hour Predictor - Aishi De",
    layout="wide"
)

st.markdown("""
<h1 style='color:#c73d28'>
⚡ AI-Based Peak Hour Electricity Prediction Dashboard</h1>
<h5>- Aishi De <br>
<h5/>
""", unsafe_allow_html=True)

df = pd.read_csv("./data/electricity_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.date

st.sidebar.markdown("<h2 style='color:#eba50e; font-weight:bold'>Dashboard Controls 🎮</h2>", unsafe_allow_html=True)

selected_dorm = st.sidebar.selectbox("Select Dorm", df["dorm"].unique())
threshold = st.sidebar.slider("Peak Alert Threshold (kWh)", 20, 50, 27)

tab1, tab2 = st.tabs(["Predictions & Alerts", "Weekly Comparison"])

with tab1:
    filtered_df = df[df["dorm"] == selected_dorm].copy()
    
    filtered_df["smoothed_consumption"] = filtered_df["consumption_kwh"].rolling(window=3).mean()
    filtered_df = filtered_df.dropna()

    X = filtered_df[["hour"]]
    y = filtered_df["smoothed_consumption"]
    model = LinearRegression()
    model.fit(X, y)

    future_hours = pd.DataFrame({"hour": [18, 19, 20, 21, 22]})
    predicted = model.predict(future_hours)
    future_time = pd.date_range(start=filtered_df["timestamp"].max(), periods=5, freq="h")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df["timestamp"],
        y=filtered_df["consumption_kwh"],
        mode="lines",
        name="Actual Consumption"
    ))
    fig.add_trace(go.Scatter(
        x=filtered_df["timestamp"],
        y=filtered_df["smoothed_consumption"],
        mode="lines",
        name="Smoothed Consumption"
    ))
    fig.add_trace(go.Scatter(
        x=future_time,
        y=predicted,
        mode="lines+markers",
        name="Predicted Evening Peak",
        line=dict(dash="dash")
    ))
    fig.update_layout(
        xaxis_title="Time ->",
        yaxis_title="Electricity Consumption (kWh) ->",
        legend_title="Legend"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<h3 style='color:#eba50e'>🔮 Predicted Evening Peak Consumption for {selected_dorm} ㄟ(≧◇≦)ㄏ</h3>", unsafe_allow_html=True)
    for h, val in zip(future_hours["hour"], predicted):
        st.write(f"**Hour {h}:00 → {val:.2f} kWh**")

    st.markdown("<h3 style='color:#eba50e'>🚨 Peak Hour Alert System!!! ::>_<::</h3>", unsafe_allow_html=True)
    for h, val in zip(future_hours["hour"], predicted):
        if val >= threshold:
            st.error(f"⚠️ High electricity load expected at {h}:00 ({val:.2f} kWh)!!")
        else:
            st.success(f"✅ Normal load at {h}:00 ({val:.2f} kWh)")

with tab2:
    st.markdown("<h3 style='color:#eba50e'>📊 Average Hourly Consumption per Dorm (Last 7 Days)</h3>",  unsafe_allow_html=True)

    last_week = df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(days=7))]

    weekly_avg = last_week.groupby(["hour", "dorm"])["consumption_kwh"].mean().reset_index()

    fig2 = go.Figure()
    for dorm in weekly_avg["dorm"].unique():
        dorm_data = weekly_avg[weekly_avg["dorm"] == dorm]
        fig2.add_trace(go.Scatter(
            x=dorm_data["hour"],
            y=dorm_data["consumption_kwh"],
            mode="lines+markers",
            name=dorm
        ))

    fig2.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Average Consumption (kWh)",
        legend_title="Dorm",
        xaxis=dict(tickmode='linear', dtick=1)
    )

    st.plotly_chart(fig2, use_container_width=True)
