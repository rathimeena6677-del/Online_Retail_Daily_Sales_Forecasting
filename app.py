import streamlit as st
import pickle
import pandas as pd

# -------------------------------
# Load Trained Model
# -------------------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Online Retail Daily Sales Forecasting",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Online Retail Daily Sales Forecasting")
st.write("Predict the daily sales of an online retail business using Machine Learning.")

st.markdown("---")

# -------------------------------
# User Inputs
# -------------------------------
orders = st.number_input("Number of Orders", min_value=0, value=100)

quantity = st.number_input("Total Quantity Sold", min_value=0, value=500)

avg_price = st.number_input("Average Product Price", min_value=0.0, value=20.0)

customers = st.number_input("Unique Customers", min_value=0, value=80)

cancelled = st.number_input("Cancelled Orders", min_value=0, value=2)

month = st.selectbox(
    "Month",
    [1,2,3,4,5,6,7,8,9,10,11,12]
)

dayofweek = st.selectbox(
    "Day of Week",
    [0,1,2,3,4,5,6],
    format_func=lambda x: [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ][x]
)

weekend = st.selectbox(
    "Weekend",
    [0,1],
    format_func=lambda x: "Yes" if x==1 else "No"
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Daily Sales"):

    input_data = pd.DataFrame({
        "Orders":[orders],
        "Total_Quantity":[quantity],
        "Average_Price":[avg_price],
        "Unique_Customers":[customers],
        "Cancelled_Orders":[cancelled],
        "Month":[month],
        "DayOfWeek":[dayofweek],
        "Weekend":[weekend]
    })

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Daily Sales: ₹ {prediction[0]:,.2f}")

st.markdown("---")
st.caption("Developed using Python, Scikit-learn and Streamlit")
