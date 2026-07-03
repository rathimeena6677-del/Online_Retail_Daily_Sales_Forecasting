"""
Streamlit app — Online Retail Daily Sales Forecasting
------------------------------------------------------
1. Loads the cleaned online retail dataset.
2. Lets the user pick a date. If that date exists in the data, all metrics
   are auto-calculated from the raw transactions. The user can also
   override any value manually before predicting.
3. Calculates: Orders, Unique Customers, Cancelled Orders, Total Quantity,
   Average Price (plus the calendar features the model was trained on:
   Month, DayOfWeek, Weekend).
4. Feeds those values into the saved Linear Regression model (model.pkl).
5. Displays the predicted Daily Sales.
 
Run with:  streamlit run app.py
"""
 
import pickle
from datetime import date
 
import numpy as np
import pandas as pd
import streamlit as st
 
# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Retail Daily Sales Forecast",
    page_icon="🛒",
    layout="centered",
)
 
DATA_PATH = "online_retail_cleaned.csv"   # place the CSV next to app.py
MODEL_PATH = "model.pkl"                  # trained model from the notebook
 
FEATURE_ORDER = [
    "Orders",
    "Total_Quantity",
    "Average_Price",
    "Unique_Customers",
    "Cancelled_Orders",
    "Month",
    "DayOfWeek",
    "Weekend",
]
 
 
# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["invoicedate"] = pd.to_datetime(df["invoicedate"])
    df["Date"] = df["invoicedate"].dt.date
    return df
 
 
@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
 
 
def compute_daily_metrics(df: pd.DataFrame, selected_date: date) -> dict | None:
    """Aggregate raw transactions for a single date into model features."""
    day_df = df[df["Date"] == selected_date]
    if day_df.empty:
        return None
 
    return {
        "Orders": int(day_df["invoice"].nunique()),
        "Total_Quantity": int(day_df["quantity"].sum()),
        "Average_Price": float(day_df["price"].mean()),
        "Unique_Customers": int(day_df["customer_id"].nunique()),
        "Cancelled_Orders": int(day_df["is_cancelled"].sum()),
    }
 
 
def calendar_features(selected_date: date) -> dict:
    dow = pd.Timestamp(selected_date).dayofweek  # Monday=0 ... Sunday=6
    return {
        "Month": pd.Timestamp(selected_date).month,
        "DayOfWeek": dow,
        "Weekend": 1 if dow >= 5 else 0,
    }
 
 
# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
st.title("🛒 Online Retail — Daily Sales Forecast")
st.write(
    "Pick a date to auto-calculate that day's metrics from the transaction "
    "log, adjust them if needed, and get a predicted total sales figure "
    "from the trained regression model."
)
 
# Load data & model up front so we can fail fast with a clear message.
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Couldn't find `{DATA_PATH}`. Place the cleaned retail CSV in the "
        "same folder as app.py (or update DATA_PATH)."
    )
    st.stop()
 
try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"Couldn't find `{MODEL_PATH}`. Run the notebook's final cell to "
        "export the trained model, and place it next to app.py."
    )
    st.stop()
 
min_date, max_date = df["Date"].min(), df["Date"].max()
 
# --- Step 1: choose a date -------------------------------------------------
st.subheader("1. Choose a date")
selected_date = st.date_input(
    "Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date,
    help=f"Dataset covers {min_date} to {max_date}.",
)
 
auto_metrics = compute_daily_metrics(df, selected_date)
 
if auto_metrics is not None:
    st.success(f"Found {int(df[df['Date'] == selected_date]['invoice'].shape[0])} "
               f"line items for {selected_date} — metrics auto-calculated below.")
else:
    st.warning(
        "No transactions found for this date in the dataset. "
        "Enter the metrics manually below to still get a prediction."
    )
    auto_metrics = {
        "Orders": 0,
        "Total_Quantity": 0,
        "Average_Price": 0.0,
        "Unique_Customers": 0,
        "Cancelled_Orders": 0,
    }
 
# --- Step 2: review / edit the metrics -------------------------------------
st.subheader("2. Review or adjust the metrics")
col1, col2 = st.columns(2)
 
with col1:
    orders = st.number_input("Orders", min_value=0, value=auto_metrics["Orders"], step=1)
    total_quantity = st.number_input(
        "Total Quantity", min_value=0, value=auto_metrics["Total_Quantity"], step=1
    )
    unique_customers = st.number_input(
        "Unique Customers", min_value=0, value=auto_metrics["Unique_Customers"], step=1
    )
 
with col2:
    average_price = st.number_input(
        "Average Price", min_value=0.0, value=float(auto_metrics["Average_Price"]), step=0.1, format="%.2f"
    )
    cancelled_orders = st.number_input(
        "Cancelled Orders", min_value=0, value=auto_metrics["Cancelled_Orders"], step=1
    )
 
cal = calendar_features(selected_date)
st.caption(
    f"Calendar features derived from the date — "
    f"Month: {cal['Month']}, DayOfWeek: {cal['DayOfWeek']} (0=Mon), "
    f"Weekend: {'Yes' if cal['Weekend'] else 'No'}"
)
 
# --- Step 3: predict ---------------------------------------------------------
st.subheader("3. Predict")
if st.button("Predict Daily Sales", type="primary"):
    input_row = pd.DataFrame(
        [{
            "Orders": orders,
            "Total_Quantity": total_quantity,
            "Average_Price": average_price,
            "Unique_Customers": unique_customers,
            "Cancelled_Orders": cancelled_orders,
            "Month": cal["Month"],
            "DayOfWeek": cal["DayOfWeek"],
            "Weekend": cal["Weekend"],
        }]
    )[FEATURE_ORDER]
 
    prediction = model.predict(input_row)[0]
 
    st.metric(
        label=f"Predicted Daily Sales for {selected_date}",
        value=f"£{prediction:,.2f}",
    )
 
    with st.expander("Show model input"):
        st.dataframe(input_row, use_container_width=True)
 
st.divider()
st.caption(
    "Model: scikit-learn LinearRegression trained on Orders, Total_Quantity, "
    "Average_Price, Unique_Customers, Cancelled_Orders, Month, DayOfWeek, "
    "and Weekend to predict Daily_Sales."
)
