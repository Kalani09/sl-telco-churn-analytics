import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# 1. Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sri Lanka Telco - Churn Predictor",
    layout="wide"
)

st.title("Sri Lanka Telco Customer Churn Risk & Analytics Dashboard")
st.markdown("Assess customer churn probability using trained **Bagging, Boosting, Stacking, and Hybrid** Machine Learning models.")

# ---------------------------------------------------------
# 2. Load Artifacts (Models & Scaler)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load('models/scaler.pkl')
    xgb_model = joblib.load('models/best_xgb.pkl')
    rf_model = joblib.load('models/best_rf.pkl')
    stacking_model = joblib.load('models/stacking_model.pkl')
    hybrid_model = joblib.load('models/hybrid_model.pkl')
    return scaler, xgb_model, rf_model, stacking_model, hybrid_model

try:
    scaler, xgb_model, rf_model, stacking_model, hybrid_model = load_artifacts()
    st.sidebar.success("All 4 Models & Scaler Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading models: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. Sidebar Model Selection
# ---------------------------------------------------------
st.sidebar.header("Dashboard Controls")

model_choice = st.sidebar.selectbox(
    "Choose Prediction Model",
    (
        "XGBoost (Boosting)", 
        "Random Forest (Bagging)", 
        "Stacking Classifier", 
        "Hybrid Model (RF + ANN)"
    )
)

model_dict = {
    "XGBoost (Boosting)": xgb_model,
    "Random Forest (Bagging)": rf_model,
    "Stacking Classifier": stacking_model,
    "Hybrid Model (RF + ANN)": hybrid_model
}
selected_model = model_dict[model_choice]

# ---------------------------------------------------------
# 4. Dashboard Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Live Prediction", "Model Comparison", "Churn Drivers"])

# =========================================================
# TAB 1: LIVE PREDICTION
# =========================================================
with tab1:
    st.subheader("Customer Daily Usage Inputs")
    st.caption("Adjust customer daily traffic and spend parameters in the sidebar or below to calculate churn risk.")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        yt_mb = st.number_input("YouTube Usage (Daily MB)", min_value=0.0, value=150.0)
        fb_mb = st.number_input("Facebook Usage (Daily MB)", min_value=0.0, value=80.0)
        tt_mb = st.number_input("TikTok Usage (Daily MB)", min_value=0.0, value=50.0)
        
    with col_b:
        wa_mb = st.number_input("WhatsApp Usage (Daily MB)", min_value=0.0, value=20.0)
        hel_mb = st.number_input("Helakuru Usage (Daily MB)", min_value=0.0, value=15.0)
        o2o_min = st.number_input("Outgoing Calls (Same Operator - Mins)", min_value=0.0, value=30.0)
        
    with col_c:
        op2o_min = st.number_input("Outgoing Calls (Other Operators - Mins)", min_value=0.0, value=15.0)
        pack_data = st.number_input("Data Package Spend (LKR)", min_value=0.0, value=450.0)
        pack_vas = st.number_input("Value Added Services Spend (LKR)", min_value=0.0, value=100.0)
        
    recent_activity_ratio = st.slider("Recent 30-Day Activity Trend (% of 90-Day Average)", 0.0, 100.0, 75.0) / 100.0

    # Feature Vector Construction
    feature_names = scaler.feature_names_in_
    input_dict = {col: 0.0 for col in feature_names}

    for col in feature_names:
        if 'youtube' in col:
            input_dict[col] = yt_mb * (30 if 'recent' in col else (90 if 'total' in col else 1.0)) * recent_activity_ratio
        elif 'facebook' in col:
            input_dict[col] = fb_mb * (30 if 'recent' in col else (90 if 'total' in col else 1.0)) * recent_activity_ratio
        elif 'tiktok' in col:
            input_dict[col] = tt_mb * (30 if 'recent' in col else (90 if 'total' in col else 1.0)) * recent_activity_ratio
        elif 'whatsapp' in col:
            input_dict[col] = wa_mb * (30 if 'recent' in col else (90 if 'total' in col else 1.0)) * recent_activity_ratio
        elif 'helakuru' in col:
            input_dict[col] = hel_mb * (30 if 'recent' in col else (90 if 'total' in col else 1.0)) * recent_activity_ratio
        elif 'pack_data' in col:
            input_dict[col] = pack_data
        elif 'pack_vas' in col:
            input_dict[col] = pack_vas
        elif 'o2o' in col:
            input_dict[col] = o2o_min
        elif 'op2o' in col:
            input_dict[col] = op2o_min

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    if st.button("Calculate Churn Risk", type="primary"):
        # Extract scalar prediction and churn probability (class 1 index)
        prediction = selected_model.predict(input_scaled)[0]
        probability = selected_model.predict_proba(input_scaled)[0][1] * 100

        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(label=f"Predicted Churn Probability ({model_choice})", value=f"{probability:.2f}%")
            
        with res_col2:
            if prediction == 1 or probability > 50:
                st.error("**High Risk of Churning**")
            else:
                st.success("**Low Risk (Customer Retained)**")

        st.subheader("Actionable Retention Strategy")
        if probability > 50:
            st.warning(
                "**Retention Recommendations:**\n"
                "- Push a targeted **20% discount coupon** on the next data reload.\n"
                "- Send an automated SMS for custom **YouTube / Social Media data bundles**.\n"
                "- Escalate to customer care for proactive retention inquiry."
            )
        else:
            st.info(
                "**Customer Status Normal:**\n"
                "- Customer engagement parameters are healthy.\n"
                "- Eligible for routine VAS promotions and standard loyalty rewards."
            )

# =========================================================
# TAB 2: MODEL COMPARISON
# =========================================================
with tab2:
    st.subheader("Ensemble Learning Benchmark Results")
    st.markdown("Comparison across **Bagging, Boosting, Stacking, and Hybridization** models evaluated on the test set:")
    
    benchmark_data = {
        "Model Paradigm": ["XGBoost (Boosting)", "Random Forest (Bagging)", "Stacking Classifier", "Hybrid Model (RF + ANN)"],
        "Architecture": ["Gradient Boosted Trees", "Bootstrap Aggregated Trees", "RF + SVM + KNN -> LogReg Meta", "Random Forest + Neural Net"],
        "Evaluation Metric Focus": ["F1-Score / LogLoss", "OOB Accuracy / Gini", "Cross-Validated Probabilities", "Soft-Voting Manifold"]
    }
    st.table(pd.DataFrame(benchmark_data))

# =========================================================
# TAB 3: CHURN DRIVERS
# =========================================================
with tab3:
    st.subheader("Top Churn Drivers & Behavioral Insights")
    st.markdown("""
    Based on model feature importances, the primary indicators of customer churn in Sri Lankan telecom data include:
    1. **Recent 30-Day Activity Drop**: Sharp declines in recent usage compared to historical 90-day averages.
    2. **High-Bandwidth App Traffic**: Rapid drops in **YouTube** and **TikTok** daily Megabytes.
    3. **Data Package Spend**: Decreased reload frequency (`usage_pack_data`).
    """)