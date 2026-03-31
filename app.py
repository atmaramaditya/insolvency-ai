import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
from main import get_prediction

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Aditya Atmaram | Insolvency AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL FINTECH LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    div.stButton > button:first-child {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
    }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/external-soft-fill-lineal-color-mixed-round/100/external-risk-management-business-soft-fill-lineal-color-mixed-round.png", width=80)
    st.title("🛡️ Risk Control Panel")
    
    st.subheader("⚙️ Model Configuration")
    threshold = st.slider("Risk Sensitivity Threshold", 0.1, 0.9, 0.5, 
                          help="Adjust the sensitivity of the bankruptcy trigger.")
    
    st.markdown("---")
    st.subheader("🧪 Manual Simulation")
    input_np = st.slider("Net Income / Total Assets", -1.0, 1.0, 0.10)
    input_debt = st.slider("Debt Ratio %", 0.0, 1.0, 0.40)
    input_wc = st.slider("Working Capital / Total Assets", -1.0, 1.0, 0.20)
    
    st.markdown("---")
    st.subheader("👤 Developer Profile")
    st.info(f"""
    **Aditya Atmaram** Data Scientist & Mechatronics Engineer  
    *Navi Mumbai, India* 🇮🇳
    """)

# --- 3. DASHBOARD HEADER (Hero Section) ---
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("⚖️ Enterprise Bankruptcy Forecasting")
    st.markdown("""
        *Predicting corporate insolvency using high-fidelity financial ratios and Gradient Boosting.*
        Developed by **Aditya Atmaram**, this system integrates Mechatronics precision with AI to monitor financial stability.
    """)

# --- 4. DATA INFERENCE & BATCH LOGIC ---
st.markdown("### 📂 Data Acquisition")
uploaded_file = st.file_uploader("Upload Company Financials (CSV Format)", type="csv", help="Upload the Taiwan Bankruptcy dataset or similar financial records.")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    with st.spinner("🔄 Running Neural Inference..."):
        res = get_prediction(0, 0, 0, custom_data=df)
    
    if res and 'all_scores' in res:
        try:
            scores_raw = res.get('all_scores', [])
            df['Risk Score'] = pd.Series(scores_raw).apply(pd.to_numeric, errors='coerce').fillna(0.0).values
            df['Prediction'] = ["Bankrupt" if r > threshold else "Healthy" for r in df['Risk Score']]
            
            # PERFORMANCE VALIDATION SECTION
            if 'Bankrupt?' in df.columns:
                st.markdown("---")
                st.subheader("📊 Model Performance Validation")
                
                y_true = df['Bankrupt?']
                y_pred = [1 if r > threshold else 0 for r in df['Risk Score']]
                cm = confusion_matrix(y_true, y_pred)
                
                c_a, c_b = st.columns([2, 1])
                with c_a:
                    z = cm.tolist()
                    x = ['Predicted Healthy', 'Predicted Bankrupt']
                    y = ['Actual Healthy', 'Actual Bankrupt']
                    fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues', showscale=True)
                    fig_cm.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                with c_b:
                    tp, fn = cm[1, 1], cm[1, 0]
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    st.metric("Model Recall (Sensitivity)", f"{recall:.1%}")
                    st.write("---")
                    st.caption("A high recall ensures that potential bankruptcies are identified, reducing the cost of Type II errors.")

            # RESULTS TABLE
            st.markdown("### 📋 Predictive Audit Report")
            preview_df = df.sort_values(by='Risk Score', ascending=False).head(50)
            styled_df = preview_df.style.background_gradient(subset=['Risk Score'], cmap='RdYlGn_r').format({"Risk Score": "{:.2%}"})
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Full Assessment (CSV)", data=csv, file_name="insolvency_analysis.csv")
            
        except Exception as e:
            st.error(f"Execution Error: {e}")
    else:
        st.error("Prediction Engine failed to return a valid response.")

else:
    # --- SINGLE PREDICTION LOGIC (Enhanced Visualization) ---
    res = get_prediction(input_np, input_debt, input_wc)
    
    if res:
        risk_score = float(res.get('risk_score', 0.0))
        adj_status = "Bankrupt" if risk_score > threshold else "Healthy"
        
        # Dynamic Status Color
        status_color = "#ff4b4b" if adj_status == "Bankrupt" else "#00cc96"
        st.markdown(f"""<div class="status-box" style="background-color: {status_color};">PROBABLE OUTCOME: {adj_status.upper()}</div>""", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🎯 Risk Diagnostic", "🧠 Explainability (XAI)", "📂 Methodology"])

        with tab1:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_score * 100,
                    title = {'text': "Insolvency Probability", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "rgba(0,0,0,0.6)"},
                        'steps': [
                            {'range': [0, 30], 'color': "#d1f2eb"},
                            {'range': [30, 70], 'color': "#fef9e7"},
                            {'range': [70, 100], 'color': "#fadbd8"}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'value': threshold * 100}
                    }
                ))
                fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with col_b:
                st.subheader("Metric Summary")
                st.metric("System Risk Level", f"{risk_score:.1%}")
                st.metric("Variance to Threshold", f"{risk_score - threshold:+.1%}")
                if risk_score > threshold:
                    st.warning("Immediate financial audit recommended.")
                else:
                    st.success("Financial health within safety margins.")

        with tab2:
            st.subheader("Feature Impact (Real-time Interpretation)")
            if 'explanations' in res:
                impact_df = pd.DataFrame({
                    "Factor": list(res['explanations'].keys()),
                    "Magnitude": list(res['explanations'].values())
                })
                fig_bar = px.bar(impact_df, x='Magnitude', y='Factor', orientation='h', 
                                 color='Magnitude', color_continuous_scale='RdYlGn_r',
                                 template="plotly_white")
                fig_bar.update_layout(height=300)
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab3:
            st.subheader("Technical Documentation")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                **Model Architecture**
                - **Primary:** Gradient Boosting Ensemble
                - **Optimization:** Bayesian Hyperparameter Tuning
                - **Data Origin:** Taiwan Economic Journal
                """)
            with c2:
                st.markdown(f"""
                **Deployment Parameters**
                - **Threshold Strategy:** Dynamic {threshold:.2f}
                - **Interface:** Decoupled FastAPI/Streamlit
                - **Region:** Navi Mumbai Deployment
                """)
    else:
        st.error("⚠️ Prediction Engine Offline.")

# --- 6. FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© 2026 Aditya Atmaram | Mechatronics & AI Engineering | Navi Mumbai, India</p>", unsafe_allow_html=True)
