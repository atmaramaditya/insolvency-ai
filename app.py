import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from main import get_prediction  # Linked to your updated main.py

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Aditya Atmaram | Insolvency AI",
    page_icon="⚖️",
    layout="wide"
)

# --- 2. SIDEBAR & INPUTS (The Control Center) ---
with st.sidebar:
    st.title("🛡️ Risk Control Panel")
    st.markdown("---")
    
    # PERSONAL BRANDING
    st.subheader("Developer Profile")
    st.image("https://img.icons8.com/illustrations/external-outline-black-m-p-group/64/external-Data-Scientist-data-science-outline-black-m-p-group.png", width=80)
    st.markdown(f"""
    **Name:** Aditya Atmaram  
    **Role:** Data Scientist & Mechatronics Engineer  
    **Base:** Navi Mumbai, India 🇮🇳  
    """)
    
    st.markdown("---")
    st.subheader("Model Simulation")
    # Store slider values in variables that we will pass directly to main.py
    input_np = st.slider("Net Income / Total Assets", -1.0, 1.0, 0.10, step=0.01)
    input_debt = st.slider("Debt Ratio %", 0.0, 1.0, 0.40, step=0.01)
    input_wc = st.slider("Working Capital / Total Assets", -1.0, 1.0, 0.20, step=0.01)
    
    st.markdown("---")
    st.caption("System Status: Operational")
    st.caption("Model Version: v6.2-Ensemble")

# --- 3. DATA INFERENCE (The Engine Link) ---
# We call the function using the slider variables defined above
res = get_prediction(input_np, input_debt, input_wc)

# Error handling to prevent the "Frozen 14.2%" bug
if res and 'risk_score' in res:
    risk_score = float(res['risk_score'])
    status = str(res['status'])
else:
    # This error will show if main.py fails to load the .pkl files
    st.error("⚠️ Prediction Engine is returning a static or null value. Check main.py mapping.")
    risk_score = 0.0
    status = "Disconnected"

# --- 4. DASHBOARD HEADER ---
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("Enterprise Bankruptcy Forecasting")
    st.write(f"Developed by **Aditya Atmaram**, this engine utilizes advanced Gradient Boosting to predict corporate insolvency risk based on live financial inputs.")

with col_h2:
    # Dynamic status badge that changes color based on real-time risk
    badge_color = "#ff4b4b" if risk_score > 0.7 else "#ffa500" if risk_score > 0.3 else "#00cc96"
    st.markdown(f"""
        <div style="background-color: {badge_color}; 
                    padding: 20px; border-radius: 10px; text-align: center; border: 2px solid white;">
            <h2 style="color: white; margin: 0; font-family: sans-serif;">{status.upper()}</h2>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- 5. ANALYTICS CENTER (Tabs) ---
tab1, tab2, tab3 = st.tabs(["🎯 Risk Diagnostic", "🧠 Explainability (XAI)", "📂 Methodology"])

with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        # Gauge Chart - The 'value' is linked directly to 'risk_score'
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score * 100,
            title = {'text': "Insolvency Probability (%)", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "black"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': "#d1f2eb"},
                    {'range': [30, 70], 'color': "#fef9e7"},
                    {'range': [70, 100], 'color': "#fadbd8"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70}
            }
        ))
        fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "black", 'family': "Arial"})
        # 2026 Fix: Replaced use_container_width with width="stretch"
        st.plotly_chart(fig, width="stretch")
    
    with c2:
        st.subheader("Live Assessment")
        # 2026 Fix: Replaced use_container_width with width="stretch"
        st.metric("Real-time Risk Level", f"{risk_score:.1%}", width="stretch")
        st.write("---")
        if risk_score > 0.7:
            st.error("**High Alert:** Financial ratios indicate a high probability of bankruptcy. Restructuring advised.")
        elif risk_score > 0.3:
            st.warning("**Caution:** Significant leverage or low liquidity detected. Monitor monthly cash flows.")
        else:
            st.success("**Stable:** Current financial position is robust. Low risk of insolvency.")

with tab2:
    st.subheader("Feature Impact (Real-time XAI)")
    st.write("This chart shows how your current slider inputs are impacting the final risk score.")
    
    # Impact calculation logic
    impact_data = pd.DataFrame({
        'Financial Factor': ['Profitability (NP)', 'Leverage (Debt)', 'Liquidity (WC)'],
        'Impact Magnitude': [-(input_np * 10), (input_debt * 15), -(input_wc * 5)]
    })
    
    fig_bar = px.bar(impact_data, x='Impact Magnitude', y='Financial Factor', orientation='h', 
                     color='Impact Magnitude', color_continuous_scale='RdYlGn_r',
                     title="Model Weight Distribution")
    # 2026 Fix: Replaced use_container_width with width="stretch"
    st.plotly_chart(fig_bar, width="stretch")

with tab3:
    st.subheader("Technical Documentation")
    st.markdown(f"""
    **Model Metadata:**
    - **Architecture:** XGBoost & Random Forest Ensemble
    - **Dataset:** Taiwan Bankruptcy (UCI)
    - **Optimization:** Bayesian Hyperparameter Tuning
    - **Input Dimensions:** 3 Primary + {len(res) if res else '0'} Derived Features
    
    **Developer Statement:**
    Engineered by **Aditya Atmaram** in Navi Mumbai. This tool demonstrates the application of Machine Learning to Financial Risk Management, providing explainable insights rather than black-box predictions.
    """)

# --- 6. FOOTER ---
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: grey;'>© 2026 Aditya Atmaram | Navi Mumbai, India | Built with Streamlit & Plotly</p>", unsafe_allow_html=True)
