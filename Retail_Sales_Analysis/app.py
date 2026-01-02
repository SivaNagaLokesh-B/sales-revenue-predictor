import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Sales Revenue Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load model artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    feature_cols = pickle.load(open("feature_columns.pkl", "rb"))
    return model, scaler, feature_cols

model, scaler, feature_cols = load_artifacts()

# -----------------------------
# Title & description
# -----------------------------
st.title("📊 Sales Revenue Intelligence Dashboard")
st.markdown(
    """
    This dashboard predicts **Sales Revenue** based on business and market factors.
    It is designed for **decision support**, not just prediction.
    """
)

# -----------------------------
# Sidebar – Business Inputs
# -----------------------------
st.sidebar.header("🔧 Business Inputs")

marketing_spend = st.sidebar.slider(
    "Marketing Spend",
    min_value=0.0,
    max_value=1000.0,
    value=500.0,
    step=25.0
)

store_count = st.sidebar.slider(
    "Store Count",
    min_value=1,
    max_value=500,
    value=100
)

customer_rating = st.sidebar.slider(
    "Customer Rating",
    min_value=1.0,
    max_value=5.0,
    value=4.0,
    step=0.1
)

seasonal_index = st.sidebar.slider(
    "Seasonal Demand Index",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.05
)

competitor_price = st.sidebar.slider(
    "Competitor Price Index",
    min_value=0.5,
    max_value=1.5,
    value=1.0,
    step=0.05
)

promotion = st.sidebar.selectbox(
    "Promotion Applied",
    ["No", "Yes"]
)

# -----------------------------
# Build input dataframe
# -----------------------------
input_data = {
    "MarketingSpend": marketing_spend,
    "StoreCount": store_count,
    "CustomerRating": customer_rating,
    "SeasonalDemandIndex": seasonal_index,
    "CompetitorPrice": competitor_price,
    "IsPromotionApplied_Yes": 1 if promotion == "Yes" else 0
}

input_df = pd.DataFrame([input_data])

# Add missing columns (one-hot encoded categories etc.)
for col in feature_cols:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_cols]

# Scale
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)[0]

# -----------------------------
# MAIN KPI SECTION
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="📈 Predicted Sales Revenue",
        value=f"{prediction:,.2f}"
    )

with col2:
    st.metric(
        label="🏪 Store Count",
        value=store_count
    )

with col3:
    st.metric(
        label="⭐ Customer Rating",
        value=customer_rating
    )

st.divider()

# -----------------------------
# Driver Explanation Section
# -----------------------------
st.subheader("🔍 Key Revenue Drivers (Model Interpretation)")

coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Impact": model.coef_
}).sort_values(by="Impact", key=abs, ascending=False)

st.write(
    "The table below shows the **relative impact** of each feature on sales revenue. "
    "Positive values increase revenue, negative values decrease it."
)

st.dataframe(
    coef_df.head(10),
    use_container_width=True
)

st.divider()

# -----------------------------
# Scenario Analysis
# -----------------------------
st.subheader("🔮 Scenario Analysis (What-If Simulation)")

scenario_col1, scenario_col2 = st.columns(2)

with scenario_col1:
    marketing_increase = st.slider(
        "Increase Marketing Spend (%)",
        min_value=0,
        max_value=50,
        value=10,
        step=5
    )

with scenario_col2:
    store_increase = st.slider(
        "Increase Store Count (%)",
        min_value=0,
        max_value=30,
        value=5,
        step=5
    )

scenario_df = input_df.copy()
scenario_df["MarketingSpend"] *= (1 + marketing_increase / 100)
scenario_df["StoreCount"] *= (1 + store_increase / 100)

scenario_scaled = scaler.transform(scenario_df)
scenario_prediction = model.predict(scenario_scaled)[0]

delta = scenario_prediction - prediction

st.metric(
    label="📊 Scenario Revenue",
    value=f"{scenario_prediction:,.2f}",
    delta=f"{delta:,.2f}"
)

st.divider()

# -----------------------------
# Business Interpretation
# -----------------------------
st.subheader("💡 Business Interpretation")

st.markdown(
    f"""
    - Increasing **marketing spend by {marketing_increase}%** and **store count by {store_increase}%**
      results in an estimated revenue change of **{delta:,.2f}**.
    - Store expansion and customer satisfaction remain the strongest revenue drivers.
    - Promotional strategies should be evaluated carefully, as their impact on net revenue may be limited.
    """
)

st.divider()

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    **Disclaimer:**  
    This model is based on historical data and linear assumptions.
    Predictions should be used for **strategic guidance**, not as exact forecasts.
    """
)
