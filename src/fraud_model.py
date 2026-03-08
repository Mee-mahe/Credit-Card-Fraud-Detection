import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("data/creditcard.csv")

print("Dataset Shape:", df.shape)

# Check fraud distribution
print(df['Class'].value_counts())

# Feature Scaling
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Drop Time column
df = df.drop(['Time'], axis=1)

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE:", np.bincount(y_train_smote))

# -------------------------
# Logistic Regression
# -------------------------

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_smote, y_train_smote)

log_pred = log_model.predict(X_test)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, log_pred))
print(classification_report(y_test, log_pred))


# -------------------------
# Random Forest
# -------------------------

rf_model = RandomForestClassifier()

rf_model.fit(X_train_smote, y_train_smote)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))


# -------------------------
# XGBoost
# -------------------------

xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train_smote, y_train_smote)

xgb_pred = xgb_model.predict(X_test)

print("\nXGBoost Results")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))
import streamlit as st
import pandas as pd

st.title("Credit Card Fraud Detection App")

st.write("Demo interface for fraud prediction")

amount = st.number_input("Transaction Amount")

v1 = st.number_input("Feature V1")
v2 = st.number_input("Feature V2")
v3 = st.number_input("Feature V3")

if st.button("Check Transaction"):

    st.write("Transaction Amount:", amount)
    st.write("V1:", v1)
    st.write("V2:", v2)
    st.write("V3:", v3)

    st.success("Model prediction will appear here (demo interface)")
