import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide"
)

# ==================================================
# LOAD MODEL AND DATA
# ==================================================

model = joblib.load("random_forest_deployment_compressed.pkl")
feature_names = joblib.load("feature_names.pkl")

forecast_df = pd.read_csv("energy_30_day_forecast.csv")
forecast_df["Datetime"] = pd.to_datetime(forecast_df["Datetime"])

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("⚡ Energy Forecasting")

st.sidebar.markdown("### Project")
st.sidebar.write("Hourly Energy Consumption Forecasting")

st.sidebar.markdown("### Dataset")
st.sidebar.write("Historical hourly energy consumption")

st.sidebar.markdown("### Final Model")
st.sidebar.write("Random Forest Regressor")

st.sidebar.markdown("### Forecast")
st.sidebar.write("Next 30 Days / 720 Hours")

st.sidebar.markdown("---")

st.sidebar.markdown("### Model Features")

for feature in feature_names:
    st.sidebar.write(f"• {feature}")

# ==================================================
# MAIN TITLE
# ==================================================

st.title("⚡ Energy Consumption Forecasting")

st.write(
    "An end-to-end machine learning system for forecasting "
    "hourly energy consumption using a Random Forest Regression model."
)

# ==================================================
# PROJECT OVERVIEW
# ==================================================

st.header("📊 Project Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Forecast Period", "30 Days")

with col2:
    st.metric("Forecast Horizon", "720 Hours")

with col3:
    st.metric("Model", "Random Forest")

with col4:
    st.metric("Target", "Energy (MW)")

# ==================================================
# FORECAST CHART
# ==================================================

st.header("🔮 30-Day Energy Consumption Forecast")

st.line_chart(
    forecast_df.set_index("Datetime")["Predicted_Consumption"]
)

# ==================================================
# FORECAST SUMMARY
# ==================================================

st.header("📈 Forecast Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Average Consumption",
        f"{forecast_df['Predicted_Consumption'].mean():,.2f} MW"
    )

with col2:
    st.metric(
        "Maximum Consumption",
        f"{forecast_df['Predicted_Consumption'].max():,.2f} MW"
    )

with col3:
    st.metric(
        "Minimum Consumption",
        f"{forecast_df['Predicted_Consumption'].min():,.2f} MW"
    )

# ==================================================
# FORECAST TABLE
# ==================================================

st.header("📋 Forecasted Energy Consumption")

st.dataframe(
    forecast_df,
    use_container_width=True,
    height=400
)

# ==================================================
# MODEL INFORMATION
# ==================================================

st.header("🤖 Final Model")

st.write("### Random Forest Regressor")

st.write(
    "The Random Forest model was selected as the final deployment "
    "model for predicting hourly energy consumption."
)

st.write("### Features Used")

feature_table = pd.DataFrame({
    "Feature": feature_names
})

st.dataframe(
    feature_table,
    use_container_width=True,
    hide_index=True
)

# ==================================================
# MODEL PERFORMANCE
# ==================================================

st.header("📊 Model Performance")

st.write(
    "Performance evaluated on the last year of historical data."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", "62.61 MW")

with col2:
    st.metric("RMSE", "82.61 MW")

with col3:
    st.metric("R² Score", "0.9931")

# ==================================================
# DOWNLOAD
# ==================================================

st.header("⬇️ Download Forecast")

csv = forecast_df.to_csv(index=False)

st.download_button(
    label="Download 30-Day Forecast CSV",
    data=csv,
    file_name="energy_30_day_forecast.csv",
    mime="text/csv"
)

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "Energy Consumption Forecasting Project | "
    "Machine Learning | Random Forest Regression"
)