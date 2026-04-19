'''import streamlit as st

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
st.success("Thank you for your time!")'''

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- 1. PAGE SETUP & LIGHT THEME CSS ---
st.set_page_config(page_title="Financial Risk Profiler", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #333333; }
    .main-title { font-size: 50px !important; font-weight: 900; color: #2C3E50; text-align: center; margin-top: 30px; margin-bottom: 10px; }
    .sub-title { font-size: 20px; text-align: center; color: #7F8C8D; margin-bottom: 40px; }
    .project-idea { background-color: #FFFFFF; padding: 25px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0,0,0,0.05); margin-bottom: 30px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. TOP SECTION ---
st.markdown('<p class="main-title">Student Financial Behavior & Risk Profiler</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Discover your financial zone through AI</p>', unsafe_allow_html=True)

st.markdown("""
<div class="project-idea">
    <h3>About This Project</h3>
    <p>This machine learning model analyzes student financial habits to predict financial risk zones. 
    By evaluating your inputs against our baseline dataset of 123 student responses, the AI identifies your unique financial profile 
    and highlights exactly where you stand on the clustering map.</p>
</div>
""", unsafe_allow_html=True)

# --- 3. REVEAL BUTTON ---
if 'show_form' not in st.session_state:
    st.session_state.show_form = False

def reveal_form():
    st.session_state.show_form = True

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.button("🎯 Let's Get Your Financial Score!", on_click=reveal_form, use_container_width=True)
st.write("---")

# --- 4. THE QUESTIONNAIRE ---
if st.session_state.show_form:
    st.markdown("### Enter Your Details:")
    
    with st.form("survey_form"):
        # Demographics & Scale Questions
        place = st.selectbox("What best describes the place you grew up in?", 
                             ["🏙️ Big metro city", "🏢 Medium-sized city", "🏘️ Small town", "🌾 Rural area"])
        
        st.markdown("**When considering a purchase, how important are the following factors?**")
        col_a, col_b = st.columns(2)
        with col_a:
            price_imp = st.selectbox("Price/Cost", ["Not important", "Slightly important", "Very important"])
            brand_imp = st.selectbox("Brand reputation", ["Not important", "Slightly important", "Very important"])
        with col_b:
            peer_imp = st.selectbox("Peer recommendation", ["Not important", "Slightly important", "Very important"])
            utility_imp = st.selectbox("Long-term utility/value", ["Not important", "Slightly important", "Very important"])

        # Tracking & Spending Habits
        track = st.selectbox("How do you track your monthly expenditures?", 
                             ["I check my bank balance occasionally.", "I do not keep the track", 
                              "I review my history within payment apps (e.g., UPI, Paytm).", 
                              "I use a dedicated expense-tracking app or spreadsheet."])
        
        graph = st.selectbox("What is the expected graph for your expenditure for the months?",
                             ["Irregular and Random Spending", "Spend a lot once and then low spending for rest", 
                              "Steady Weekdays with High Weekends", "Uniform Daily Expenses", "None"])

        # Multi-selects (Binary features)
        justify = st.multiselect("In which scenarios would you justify an unexpected expense of ₹1,500+?", 
                                 ["Emergencies (e.g., phone/laptop repair).", "A 50% discount on a brand I highly value.", 
                                  "Social celebrations or parties.", "Skill development (workshops, certifications, technical kits).", 
                                  "A planned trip with friends."])
        
        budget = st.multiselect("In which categories do you spend the majority of your budget?",
                                ["Food & Dining", "Travel", "Fashion", "Subscriptions (Netflix, Spotify, etc.)", "Fun & Entertainment"])

        # Numerical & Scaled Inputs (Assuming 1-5 scales for the min/max scaled ones based on typical forms)
        unplanned = st.number_input("How often do you make purchases that you hadn’t planned for? (Scale 1-5)", 1, 5, 3)
        monthly_spend = st.number_input("On average, how much do you spend per month?", min_value=0, step=500)
        peer_inf = st.number_input("How much do social events/peer pressure influence spending? (Scale 1-5)", 1, 5, 3)
        confidence = st.number_input("How confident are you in managing personal finances? (Scale 1-5)", 1, 5, 3)

        submitted = st.form_submit_button("Submit & Predict")
        
        if submitted:
            # --- 5. DATA PREPROCESSING (Matching Cleaning.ipynb) ---
            
            # Ordinal Mappings
            ord_map_place = {"🏙️ Big metro city":0, "🏢 Medium-sized city":1, "🏘️ Small town":2, "🌾 Rural area":3}
            ord_map_imp = {"Not important":0, "Slightly important":1, "Very important":2}
            
            # 1-5. Ordinal Features
            f1 = ord_map_place[place]
            f2 = ord_map_imp[price_imp]
            f3 = ord_map_imp[brand_imp]
            f4 = ord_map_imp[peer_imp]
            f5 = ord_map_imp[utility_imp]
            
            # 6-9. Track Expenditures One-Hot
            f6 = 1 if track == "I check my bank balance occasionally." else 0
            f7 = 1 if track == "I do not keep the track" else 0
            f8 = 1 if track == "I review my history within payment apps (e.g., UPI, Paytm)." else 0
            f9 = 1 if track == "I use a dedicated expense-tracking app or spreadsheet." else 0
            
            # 10-14. Expenditure Graph One-Hot
            f10 = 1 if graph == "Irregular and Random Spending" else 0
            f11 = 1 if graph == "Spend a lot once and then low spending for rest" else 0
            f12 = 1 if graph == "Steady Weekdays with High Weekends" else 0
            f13 = 1 if graph == "Uniform Daily Expenses" else 0
            f14 = 1 if graph == "None" else 0
            
            # 15-19. Justify Unexpected Expense Binary
            f15 = 1 if "Emergencies (e.g., phone/laptop repair)." in justify else 0
            f16 = 1 if "A 50% discount on a brand I highly value." in justify else 0
            f17 = 1 if "Social celebrations or parties." in justify else 0
            f18 = 1 if "Skill development (workshops, certifications, technical kits)." in justify else 0
            f19 = 1 if "A planned trip with friends." in justify else 0
            
            # 20, 22, 23. Scaled Features (We will scale them using the loaded scaler below)
            # 21. Monthly Spend
            f21 = monthly_spend
            
            # 24-28. Budget Binary
            f24 = 1 if "Food & Dining" in budget else 0
            f25 = 1 if "Travel" in budget else 0
            f26 = 1 if "Fashion" in budget else 0
            f27 = 1 if "Subscriptions (Netflix, Spotify, etc.)" in budget else 0
            f28 = 1 if "Fun & Entertainment" in budget else 0

            # --- 6. ML PREDICTION & PLOTTING ---
            try:
                model = joblib.load('financial_model.pkl')
                scaler = joblib.load('scaler.pkl')
                df_bg = pd.read_csv('background_data.csv')
                
                # Apply MinMaxScaler to the specific 3 columns just like in the notebook
                scaled_vals = scaler.transform([[unplanned, peer_inf, confidence]])
                f20 = scaled_vals[0][0] # Unplanned
                f22 = scaled_vals[0][1] # Peer Influence
                f23 = scaled_vals[0][2] # Finance Confidence
                
                # Combine all 28 features in the EXACT order of Data.info()
                input_array = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, 
                                         f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28]])
                
                # Predict
                predicted_zone = model.predict(input_array)[0]
                
                st.success("Analysis Complete!")
                st.markdown(f"## 📊 Your Risk Profile is: **Zone {predicted_zone}**")
                
                # Plotting the Graph
                fig, ax = plt.subplots(figsize=(9, 5))
                
                # Plot background points (Plotting Monthly Spend vs. Finance Confidence as an example)
                # Note: Adjust column names if you want to plot different features from your dataset!
                ax.scatter(df_bg['Monthly_Spend'], df_bg['Finance_Confidence'], c='#BDC3C7', label='Other Students', alpha=0.6)
                
                # Plot the current user's data point
                ax.scatter(f21, f23, c='#E74C3C', marker='*', s=400, edgecolor='black', label='YOU ARE HERE')
                
                ax.set_title("Your Position in the Dataset", fontsize=14, fontweight='bold')
                ax.set_xlabel("Monthly Spend")
                ax.set_ylabel("Finance Confidence (Scaled)")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error loading model files: {e}. Ensure financial_model.pkl, scaler.pkl, and background_data.csv are uploaded.")
