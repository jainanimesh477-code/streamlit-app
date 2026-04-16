import streamlit as st

st.title("Student Financial Risk Predictor")

income = st.slider("Income", 0, 100000, 50000)
spending = st.slider("Spending", 0, 100000, 30000)
savings = st.slider("Savings", 0, 100000, 20000)

risk_score = (spending - savings) / (income + 1)

if risk_score < 0.2:
    category = "Low Risk"
elif risk_score < 0.5:
    category = "Medium Risk"
else:
    category = "High Risk"

st.write("Risk Score:", round(risk_score,2))
st.write("Category:", category)
st.markdown("---")
st.success("Thank you for your time!")
