import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(page_title="Financial Risk Profiler", layout="centered")

# --- CUSTOM GOOGLE FORMS CSS ---
st.markdown("""
    <style>
    /* Light background for the whole page */
    .stApp {
        background-color: #F0EBF8; 
    }
    
    /* Main Title Styling */
    .main-title { 
        font-size: 45px !important; 
        font-weight: 900; 
        color: #202124; 
        text-align: center; 
        margin-top: 20px; 
        margin-bottom: 5px; 
    }
    .sub-title { 
        font-size: 18px; 
        text-align: center; 
        color: #5F6368; 
        margin-bottom: 30px; 
    }
    
    /* Google Form Card Styling */
    div[data-testid="stForm"] {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 30px;
        border-top: 10px solid #673AB7; /* Google Form Purple */
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown('<p class="main-title">Student Financial Behavior & Risk Profiler</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Discover your financial zone through AI</p>', unsafe_allow_html=True)

if 'show_form' not in st.session_state:
    st.session_state.show_form = False

def reveal_form():
    st.session_state.show_form = True

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.button("🎯 Let's Get Your Financial Score!", on_click=reveal_form, use_container_width=True)
st.write("---")

# --- THE GOOGLE FORM UI ---
if st.session_state.show_form:
    with st.form("survey_form"):
        st.markdown("### 📋 Student Financial Survey")
        st.write("* Indicates required question")
        st.write("---")
        
        # Q1
        place = st.radio("1. What best describes the place you grew up in? *", 
                         ["Big metro city", "Medium-sized city", "Small town", "Rural area"])
        
        # Q2
        st.write("2. How often do you make purchases that you hadn't planned for? *")
        unplanned = st.radio("1 = Never, 5 = Very often", [1, 2, 3, 4, 5], horizontal=True)
        
        # Q3
        st.write("3. When considering a purchase, how important are the following factors to you? *")
        price_imp = st.radio("Price/Cost", ["Not important", "Slightly important", "Very important"], horizontal=True)
        brand_imp = st.radio("Brand reputation", ["Not important", "Slightly important", "Very important"], horizontal=True)
        peer_imp = st.radio("Peer recommendation", ["Not important", "Slightly important", "Very important"], horizontal=True)
        utility_imp = st.radio("Long-term utility/value", ["Not important", "Slightly important", "Very important"], horizontal=True)

        # Q4
        monthly_spend = st.slider("4. On average, how much do you spend per month (excluding tuition fees)? *", 
                                  min_value=1, max_value=10, value=5, help="1 = ₹1,000 | 10 = ₹10,000 or more")
        
        # Q5
        st.write("5. How confident do you feel in your ability to manage your personal finances? *")
        confidence = st.radio("1 = Very low, 5 = Very high", [1, 2, 3, 4, 5], horizontal=True)

        # Q6
        track = st.radio("6. How do you track your monthly expenditures? *", 
                         ["I check my bank balance occasionally.", 
                          "I review my history within payment apps (e.g., UPI, Paytm).", 
                          "I do not keep the track", 
                          "I use a dedicated expense-tracking app or spreadsheet."])
        
        # Q7
        st.write("7. On a scale of 1 to 5, how much do social events or peer pressure influence your spending?")
        peer_inf = st.radio("1 = Not Influenced, 5 = Highly Influenced", [1, 2, 3, 4, 5], horizontal=True)

        # Q8 (Checkboxes for "Check all that apply")
        st.write("8. In which categories do you spend the majority of your budget?")
        st.write("Check all that apply.")
        b_food = st.checkbox("Food & Dining")
        b_travel = st.checkbox("Travel")
        b_fashion = st.checkbox("Fashion")
        b_subs = st.checkbox("Subscriptions (Netflix, Spotify, etc.)")
        b_ent = st.checkbox("Fun & Entertainment")

        # Q9
        st.write("9. In which of the following scenarios would you justify an unexpected expense of 1,500 or more?")
        st.write("Check all that apply.")
        j_discount = st.checkbox("A 50% discount on a brand I highly value.")
        j_party = st.checkbox("Social celebrations or parties.")
        j_skill = st.checkbox("Skill development (workshops, certifications, technical kits).")
        j_emerg = st.checkbox("Emergencies (e.g., phone/laptop repair).")
        j_trip = st.checkbox("A planned trip with friends.")

        # Q10
        graph = st.radio("10. What is the expected graph for your expenditure for the months *",
                         ["Uniform Daily Expenses", 
                          "Irregular and Random Spending", 
                          "Spend a lot once and then low spending for rest", 
                          "Steady Weekdays with High Weekends", 
                          "None"])

        st.write("---")
        submitted = st.form_submit_button("Submit & Predict")
        
        # --- ML PREDICTION LOGIC ---
        if submitted:
            # 1. Map Ordinal Text
            ord_map_place = {"Big metro city":0, "Medium-sized city":1, "Small town":2, "Rural area":3}
            ord_map_imp = {"Not important":0, "Slightly important":1, "Very important":2}
            
            f1 = ord_map_place[place]
            f2 = ord_map_imp[price_imp]
            f3 = ord_map_imp[brand_imp]
            f4 = ord_map_imp[peer_imp]
            f5 = ord_map_imp[utility_imp]
            
            # 2. Map Track Expenditures
            f6 = 1 if track == "I check my bank balance occasionally." else 0
            f7 = 1 if track == "I do not keep the track" else 0
            f8 = 1 if track == "I review my history within payment apps (e.g., UPI, Paytm)." else 0
            f9 = 1 if track == "I use a dedicated expense-tracking app or spreadsheet." else 0
            
            # 3. Map Expenditure Graph
            f10 = 1 if graph == "Irregular and Random Spending" else 0
            f11 = 1 if graph == "Spend a lot once and then low spending for rest" else 0
            f12 = 1 if graph == "Steady Weekdays with High Weekends" else 0
            f13 = 1 if graph == "Uniform Daily Expenses" else 0
            f14 = 1 if graph == "None" else 0
            
            # 4. Map Justify Unexpected Expense
            f15 = 1 if j_emerg else 0
            f16 = 1 if j_discount else 0
            f17 = 1 if j_party else 0
            f18 = 1 if j_skill else 0
            f19 = 1 if j_trip else 0
            
            # 5. Scaled Features (Placeholder for applying your scaler)
            f21 = monthly_spend
            
            # 6. Map Budget
            f24 = 1 if b_food else 0
            f25 = 1 if b_travel else 0
            f26 = 1 if b_fashion else 0
            f27 = 1 if b_subs else 0
            f28 = 1 if b_ent else 0

            try:
                # Load models
                model = joblib.load('financial_model.pkl')
                scaler = joblib.load('scaler.pkl')
                df_bg = pd.read_csv('background_data.csv')
                
                # Apply scaler to the 3 specific columns
                scaled_vals = scaler.transform([[unplanned, peer_inf, confidence]])
                f20 = scaled_vals[0][0]
                f22 = scaled_vals[0][1]
                f23 = scaled_vals[0][2]
                
                # Combine array
                input_array = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, 
                                         f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28]])
                
                # Predict
                predicted_zone = model.predict(input_array)[0]
                
                st.success("Analysis Complete!")
                st.markdown(f"## 📊 Your Risk Profile is: **Zone {predicted_zone}**")
                
                # Plot Graph
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.scatter(df_bg['Monthly_Spend'], df_bg['Finance_Confidence'], c='#BDC3C7', label='Other Students', alpha=0.6)
                ax.scatter(f21, f23, c='#673AB7', marker='*', s=400, edgecolor='black', label='YOU ARE HERE')
                ax.set_title("Your Position in the Dataset", fontsize=14, fontweight='bold')
                ax.set_xlabel("Monthly Spend (1-10 Scale)")
                ax.set_ylabel("Finance Confidence (Scaled)")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error loading model files: {e}.")
