import streamlit as st
import pandas as pd
import pickle


model = pickle.load(
    open(
        "models/isolation_model.pkl",
        "rb"
    )
)


scaler = pickle.load(
    open(
        "models/scaler.pkl",
        "rb"
    )
)


st.title(
    "Customer Anomaly Detection"
)


age = st.number_input(
    "Customer Age",
    value=40
)


gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)


income = st.number_input(
    "Income Category",
    value=2
)


credit = st.number_input(
    "Credit Limit",
    value=5000.0
)


balance = st.number_input(
    "Total Revolving Balance",
    value=1000
)


amount = st.number_input(
    "Transaction Amount",
    value=3000
)


count = st.number_input(
    "Transaction Count",
    value=50
)



# =====================
# Encoding
# =====================


if gender == "Male":

    gender_value = 1

else:

    gender_value = 0



# =====================
# Feature Engineering
# =====================


Credit_Usage_Ratio = (
    balance /
    credit
)


Avg_Transaction = (
    amount /
    count
)


Transaction_Per_Month = (
    count /
    12
)



# VERY IMPORTANT
# columns same as train.py


data = pd.DataFrame(
    [[
        age,
        gender_value,
        income,
        credit,
        balance,
        amount,
        count,
        Credit_Usage_Ratio,
        Avg_Transaction,
        Transaction_Per_Month
    ]],

    columns=[
        "Customer_Age",
        "Gender",
        "Income_Category",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Credit_Usage_Ratio",
        "Avg_Transaction",
        "Transaction_Per_Month"
    ]
)



if st.button(
    "Detect Anomaly"
):


    scaled = scaler.transform(
        data
    )


    prediction = model.predict(
        scaled
    )


    if prediction[0] == -1:

        st.error(
            "Anomaly Customer Detected"
        )


    else:

        st.success(
            "Normal Customer"
        )