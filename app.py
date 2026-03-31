import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from main import get_prediction

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Aditya Atmaram | Insolvency AI",
    page_icon="⚖️",
    layout="wide"
)

# --- 2. SIDEBAR (The Control Center) ---
with st.sidebar:
    st.title("🛡️ Risk Control Panel")
    
    # Sensitivity Controller for Imbalanced Taiwan Data
    st.subheader("Decision Threshold")
    threshold = st.slider("Risk Sensitivity", 0.1, 0.9, 0.5, 
                          help="Lowering this catches more bankruptcies but increases false alarms.")
    
    st.markdown("---")
    st.subheader("Manual Override")
    input_np = st.slider("Net Income / Total Assets", -1.0, 1.0, 0.10)
    input_debt = st.slider("Debt Ratio %", 0.0, 1.0, 0.40)
    input_wc = st.slider("Working Capital / Total Assets", -1.0, 1.0, 0.20)
    
    st.markdown("---")
    st.subheader("Developer Profile")
    st.markdown(f"""
    **Name:** Aditya Atmaram  
    **Role:** Data Scientist & Mechatronics Engineer  
    **Base:** Navi Mumbai, India 🇮🇳  
    """)

# --- 3. DASHBOARD HEADER ---
st.title("Enterprise Bankruptcy Forecasting")
st.write(f"Developed by **Aditya Atmaram**, this engine utilizes advanced Gradient Boosting to predict corporate insolvency risk.")

# --- 4. DATA INFERENCE & BATCH LOGIC ---
uploaded_file = st.file_uploader("📂 Upload Company Financials (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    res = get_prediction(0, 0, 0, custom_data=df)
    
    if res:
        # FIX 1: Ensure Risk Score is numeric to prevent styling engine crash
        df['Risk Score'] = pd.to_numeric(res['all_scores'], errors='coerce').fillna(0.0)
        df['Prediction'] = ["Bankrupt" if r > threshold else "Healthy" for r in df['Risk Score']]
        
        st.write("### Batch Analysis Results (Preview)")
        
        # FIX 2: Limit data serialization to prevent Arrow Marshalling error
        # We show the top 50 high-risk entities first
        try:
            preview_df = df.sort_values(by='Risk Score', ascending=False).head(50)
            styled_df = preview_df.style.background_gradient(subset=['Risk Score'], cmap='RdYlGn_r')
            st.dataframe(styled_df, use_container_width=True)
            
            # Add Download Button for full dataset
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Assessment Report", data=csv, file_name="insolvency_analysis.csv")
        except Exception as e:
            st.warning("Styling failed due to data scale. Showing raw preview.")
            st.dataframe(df.head(50), use_container_width=True)
    else:
        st.error("Prediction failed for the uploaded file.")

else:
    # --- SINGLE PREDICTION LOGIC ---
    res = get_prediction(input_np, input_debt, input_wc)
    
    if res:
        risk_score = float(res['risk_score'])
        adj_status = "Bankrupt" if risk_score > threshold else "Healthy"
        
        # --- 5. ANALYTICS CENTER (Tabs) ---
        tab1, tab2, tab3 = st.tabs(["🎯 Risk Diagnostic", "🧠 Explainability (XAI)", "📂 Methodology"])

        with tab1:
            c1, c2 = st.columns([2, 1])
            with c1:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_score * 100,
                    title = {'text': "Insolvency Probability (%)", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "black"},
                        'steps': [
                            {'range': [0, 30], 'color': "#d1f2eb"},
                            {'range': [30, 70], 'color': "#fef9e7"},
                            {'range': [70, 100], 'color': "#fadbd8"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'value': threshold * 100}
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                st.subheader("Live Assessment")
                st.metric("Status", adj_status)
                st.metric("Risk Probability", f"{risk_score:.1%}")
                
                if risk_score > threshold:
                    st.error(f"**High Alert:** Above the {threshold} threshold.")
                else:
                    st.success("**Stable:** Risk within acceptable bounds.")

        with tab2:
            st.subheader("Feature Impact (Real-time XAI)")
            if 'explanations' in res:
                impact_df = pd.DataFrame({
                    "Financial Factor": list(res['explanations'].keys()),
                    "Input Value": list(res['explanations'].values())
                })
                fig_bar = px.bar(impact_df, x='Input Value', y='Financial Factor', orientation='h', 
                                 color='Input Value', color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab3:
            st.subheader("Technical Documentation")
            st.markdown(f"""
            **Methodology:**
            - **Control Theory Integration:** Applying mechatronics principles to monitor financial system stability.
            - **Imbalance Strategy:** Utilizing a dynamic **Decision Threshold** (currently set at {threshold}) to optimize Recall for minority-class bankruptcy events.
            - **Architecture:** Decoupled FastAPI Backend + Streamlit Frontend.
            """)
    else:
        st.error("⚠️ Prediction Engine is offline. Check if your .pkl files exist in 'deployment_assets'.")

# --- 6. FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© 2026 Aditya Atmaram | Navi Mumbai, India</p>", unsafe_allow_html=True)
