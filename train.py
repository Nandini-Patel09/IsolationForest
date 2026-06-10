import pandas as pd
import pickle
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest


os.makedirs(
    "models",
    exist_ok=True
)


# ======================
# Load Dataset
# ======================

df = pd.read_csv(
    "dataset/BankChurners.csv"
)


# ======================
# Remove unnecessary columns
# ======================


df = df[
    [
        "Customer_Age",
        "Gender",
        "Income_Category",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Total_Trans_Amt",
        "Total_Trans_Ct"
    ]
]


# ======================
# Encoding
# ======================

encoder = LabelEncoder()


df["Gender"] = encoder.fit_transform(
    df["Gender"]
)


df["Income_Category"] = encoder.fit_transform(
    df["Income_Category"]
)



# ======================
# Feature Engineering
# ======================


df["Credit_Usage_Ratio"] = (
    df["Total_Revolving_Bal"]
    /
    df["Credit_Limit"]
)


df["Avg_Transaction"] = (
    df["Total_Trans_Amt"]
    /
    df["Total_Trans_Ct"]
)


df["Transaction_Per_Month"] = (
    df["Total_Trans_Ct"]
    /
    12
)


df.replace(
    [float("inf")],
    0,
    inplace=True
)



print(
    df.columns
)


# ======================
# Scaling
# ======================


scaler = StandardScaler()


scaled = scaler.fit_transform(
    df
)



# ======================
# First Isolation Forest
# Remove anomalies
# ======================


detector = IsolationForest(
    contamination=0.05,
    random_state=42
)


labels = detector.fit_predict(
    scaled
)


clean_data = scaled[
    labels == 1
]


print(
    "Before removing anomalies:",
    scaled.shape
)


print(
    "After removing anomalies:",
    clean_data.shape
)



# ======================
# Final Model
# ======================


model = IsolationForest(
    contamination=0.05,
    random_state=42
)


model.fit(
    clean_data
)



# ======================
# Save
# ======================


with open(
    "models/isolation_model.pkl",
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )



with open(
    "models/scaler.pkl",
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )


print(
    "Training completed successfully"
)