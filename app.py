import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from main import get_prediction

st.set_page_config(page_title="Aditya Atmaram | Insolvency AI", page_icon="⚖️", layout="wide")

# --- SIDEBAR UPGRADE ---
with st.sidebar:
    st.title("🛡️ Risk Control Panel")
    
    # NEW: Sensitivity Controller (Crucial for Imbalanced Taiwan Data)
    st.subheader("Decision Threshold")
    threshold = st.slider("Risk Sensitivity", 0.1, 0.9, 0.5, help="Lowering this catches more bankruptcies but increases false alarms.")
    
    st.markdown("---")
    st.subheader("Manual Override")
    input_np = st.slider("Net Income / Total Assets", -1.0, 1.0, 0.10)
    input_debt = st.slider("Debt Ratio %", 0.0, 1.0, 0.40)
    input_wc = st.slider("Working Capital / Total Assets", -1.0, 1.0, 0.20)

# --- BATCH PROCESSING LOGIC ---
st.title("Enterprise Bankruptcy Forecasting")

uploaded_file = st.file_uploader("📂 Upload Company Financials (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    res = get_prediction(0, 0, 0, custom_data=df)
    # Add results to dataframe
    df['Risk Score'] = res['risk_score']
    df['Prediction'] = ["Bankrupt" if r > threshold else "Healthy" for r in res['risk_score']]
    st.write("### Batch Results")
    st.dataframe(df.style.background_gradient(subset=['Risk Score'], cmap='RdYlGn_r'))
else:
    # SINGLE PREDICTION LOGIC (Your existing UI)
    res = get_prediction(input_np, input_debt, input_wc)
    risk_score = res['risk_score'][0]
    
    # DYNAMIC METRIC
    adj_status = "Bankrupt" if risk_score > threshold else "Healthy"
    
    # ... (Your Gauge Chart Logic here, using 'risk_score') ...

    # UPGRADE: Local Explanation (Tab 2)
    with st.expander("🔍 Why this score?"):
        impact_df = pd.DataFrame({
            "Feature": list(res['explanations'].keys()),
            "Value": list(res['explanations'].values())
        })
        fig_exp = px.bar(impact_df, x='Value', y='Feature', orientation='h', title="Feature Contribution")
        st.plotly_chart(fig_exp)

# --- METHODOLOGY UPGRADE ---
# In Tab 3, mention the Mechatronics angle: 
# "Applying Control Theory principles to Financial Stability."
