import streamlit as st
import pandas as pd
import joblib

# Load saved model and scaler
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")

# Define cluster names (op1tional)
cluster_labels = {
    0: "Budget Customers",
    1: "Standard Shoppers",
    2: "Target Customers (High Income & Spending)",
    3: "Potential Customers (High Income, Low Spending)",
    4: "Low Income, High Spending"
}

# Streamlit UI
st.title("Customer Segmentation using K-Means")
st.markdown("Enter new customer details to predict their segment.")

# User inputs
income = st.number_input("Annual Income (k$)", min_value=10, max_value=150, value=50)
spending = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# Predict cluster
if st.button("Predict Cluster"):
    new_data = pd.DataFrame([[income, spending]], columns=['Annual Income (k$)', 'Spending Score (1-100)'])
    new_scaled = scaler.transform(new_data)
    cluster = kmeans.predict(new_scaled)[0]
    st.success(f"Predicted Cluster: {cluster} - {cluster_labels.get(cluster, 'Unknown')}")
