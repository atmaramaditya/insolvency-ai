import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff  # Required for Confusion Matrix Heatmap
from sklearn.metrics import confusion_matrix
from main import get_prediction

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Aditya Atmaram | Insolvency AI",
    page_icon="⚖️",
    layout="wide"
)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Risk Control Panel")
    
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
    
    if res and 'all_scores' in res:
        try:
            # Data Cleaning & Type Formatting
            scores_raw = res.get('all_scores', [])
            df['Risk Score'] = pd.Series(scores_raw).apply(pd.to_numeric, errors='coerce').fillna(0.0).values
            df['Prediction'] = ["Bankrupt" if r > threshold else "Healthy" for r in df['Risk Score']]
            
            st.write("### Batch Analysis Results (Preview)")
            
            # --- CONFUSION MATRIX LOGIC ---
            # Check if the Taiwan dataset ground truth column 'Bankrupt?' exists
            if 'Bankrupt?' in df.columns:
                st.markdown("---")
                st.subheader("📊 Model Performance Validation (Confusion Matrix)")
                
                y_true = df['Bankrupt?']
                y_pred = [1 if r > threshold else 0 for r in df['Risk Score']]
                cm = confusion_matrix(y_true, y_pred)
                
                # Plotly Annotated Heatmap
                z = cm.tolist()
                x = ['Predicted Healthy', 'Predicted Bankrupt']
                y = ['Actual Healthy', 'Actual Bankrupt']
                
                fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues', showscale=True)
                fig_cm.update_layout(xaxis_title="Predicted Status", yaxis_title="Actual Status")
                
                c_a, c_b = st.columns([2, 1])
                with c_a:
                    st.plotly_chart(fig_cm, use_container_width=True)
                with c_b:
                    # Calculate Recall (Crucial for Bankruptcy)
                    tp, fn = cm[1, 1], cm[1, 0]
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    st.metric("Recall (Sensitivity)", f"{recall:.1%}")
                    st.info("In insolvency detection, high **Recall** is prioritized to ensure no potential bankruptcy is missed.")
                st.markdown("---")

            # Result Preview
            preview_df = df.sort_values(by='Risk Score', ascending=False).head(50)
            styled_df = preview_df.style.background_gradient(subset=['Risk Score'], cmap='RdYlGn_r')
            st.dataframe(styled_df, use_container_width=True)
            
            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Assessment Report", data=csv, file_name="insolvency_analysis.csv")
            
        except Exception as e:
            st.error(f"Logic Error: {e}")
            st.dataframe(df.head(50), use_container_width=True)
    else:
        st.error("Prediction failed. The model backend returned an empty response.")

else:
    # --- SINGLE PREDICTION LOGIC ---
    res = get_prediction(input_np, input_debt, input_wc)
    
    if res:
        risk_score = float(res.get('risk_score', 0.0))
        adj_status = "Bankrupt" if risk_score > threshold else "Healthy"
        
        tab1, tab2, tab3 = st.tabs(["🎯 Risk Diagnostic", "🧠 Explainability (XAI)", "📂 Methodology"])

        with tab1:
            col_a, col_b = st.columns([2, 1])
            with col_a:
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
            
            with col_b:
                st.subheader("Live Assessment")
                st.metric("Status", adj_status)
                st.metric("Risk Probability", f"{risk_score:.1%}")
                
                if risk_score > threshold:
                    st.error(f"High Risk Alert")
                else:
                    st.success("Stable Position")

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
            - **Model:** Gradient Boosting Ensemble (XGBoost/RF)
            - **Data Handling:** Dynamic Thresholding at {threshold:.2f} to manage class imbalance.
            - **Focus:** Minimizing Type II Errors (False Negatives).
            """)
    else:
        st.error("⚠️ Prediction Engine Offline.")

# --- 6. FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© 2026 Aditya Atmaram | Navi Mumbai, India</p>", unsafe_allow_html=True)
