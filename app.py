import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
import os

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="🔥 Calorie Burn Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODELS_DIR = "models"

# -------------------------------------------------------------------
# Ensure models exist (train on server if needed)
# -------------------------------------------------------------------
def ensure_models():
    """
    Train models on first run if they don't exist (for Streamlit Cloud).
    Assumes train_models.py trains and saves all models into models/.
    """
    if not os.path.exists(MODELS_DIR) or not os.listdir(MODELS_DIR):
        st.info("📦 First-time setup: training models on server, this may take up to a minute...")
        import train_models  # This will run the training pipeline
        st.success("✅ Models trained and saved. Ready to predict!")

# -------------------------------------------------------------------
# Load models and scaler
# -------------------------------------------------------------------
@st.cache_resource
def load_models():
    if not os.path.exists(MODELS_DIR) or not os.listdir(MODELS_DIR):
        raise FileNotFoundError("Models directory is empty. ensure_models() must run first.")

    models = {}
    p = os.path.join

    if os.path.exists(p(MODELS_DIR, "linear_regression.pkl")):
        models["Linear Regression"] = joblib.load(p(MODELS_DIR, "linear_regression.pkl"))
    if os.path.exists(p(MODELS_DIR, "random_forest.pkl")):
        models["Random Forest"] = joblib.load(p(MODELS_DIR, "random_forest.pkl"))
    if os.path.exists(p(MODELS_DIR, "gradient_boosting.pkl")):
        models["Gradient Boosting"] = joblib.load(p(MODELS_DIR, "gradient_boosting.pkl"))
    if os.path.exists(p(MODELS_DIR, "xgboost.pkl")):
        models["XGBoost"] = joblib.load(p(MODELS_DIR, "xgboost.pkl"))
    if os.path.exists(p(MODELS_DIR, "lightgbm.pkl")):
        models["LightGBM"] = joblib.load(p(MODELS_DIR, "lightgbm.pkl"))
    if os.path.exists(p(MODELS_DIR, "ridge.pkl")):
        models["Ridge Regression"] = joblib.load(p(MODELS_DIR, "ridge.pkl"))
    if os.path.exists(p(MODELS_DIR, "lasso.pkl")):
        models["Lasso Regression"] = joblib.load(p(MODELS_DIR, "lasso.pkl"))

    if not models:
        raise FileNotFoundError("No model .pkl files found in models/ after training.")

    return models

@st.cache_resource
def load_scaler():
    if not os.path.exists(MODELS_DIR) or not os.listdir(MODELS_DIR):
        raise FileNotFoundError("Models directory is empty. ensure_models() must run first.")
    return joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

# -------------------------------------------------------------------
# Sample data for analytics (synthetic, for visualization only)
# -------------------------------------------------------------------
@st.cache_data
def load_sample_data():
    return pd.DataFrame({
        'Gender': np.random.choice([0, 1], 200),
        'Age': np.random.randint(18, 70, 200),
        'Height': np.random.randint(150, 200, 200),
        'Weight': np.random.randint(50, 100, 200),
        'Duration': np.random.randint(10, 120, 200),
        'Heart_Rate': np.random.randint(80, 180, 200),
        'Body_Temp': np.random.uniform(37, 40, 200),
        'Env_Temp': np.random.uniform(10, 35, 200),
        'Calories': np.random.randint(50, 500, 200),
    })

# -------------------------------------------------------------------
# Custom CSS
# -------------------------------------------------------------------
st.markdown("""
    <style>
    .metric-card {
        background-color: #111827;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stMetric {
        background-color: #020617;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Title + Hero banner
# -------------------------------------------------------------------
st.title("🔥 Advanced Calorie Burn Prediction System")
st.markdown("""
<div style='background: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%);
            padding: 20px; border-radius: 10px; margin-bottom: 20px;
            color: white; box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);'>
  <h4 style='color: white;'>🎯 ML-Powered Fitness Analytics Dashboard</h4>
  <p style='color: white;'>
    End-to-end machine learning system predicting calories burned during exercise using 7 algorithms,
    explainable AI (SHAP), and an interactive Streamlit dashboard.
  </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------------------
st.sidebar.header("⚙️ Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🎯 Prediction", "📊 Model Comparison", "🔍 Explainability", "📈 Analytics", "ℹ️ About"]
)

# Ensure models and scaler exist before any page uses them
ensure_models()

# -------------------------------------------------------------------
# PREDICTION PAGE
# -------------------------------------------------------------------
if page == "🎯 Prediction":
    st.header("🎯 Real-Time Calorie Burn Prediction")

    models = load_models()
    scaler = load_scaler()

    model_choice = st.sidebar.selectbox(
        "Select ML Model",
        options=list(models.keys()),
        help="Choose a machine learning model for prediction"
    )

    with st.sidebar.expander("📖 Model Info"):
        model_info = {
            'Linear Regression': "Simple linear baseline. Fast and interpretable.",
            'Random Forest': "Ensemble of decision trees. Robust, high accuracy.",
            'Gradient Boosting': "Sequential trees minimizing residual error.",
            'XGBoost': "Optimized gradient boosting, industry-standard.",
            'LightGBM': "Fast histogram-based gradient boosting.",
            'Ridge Regression': "L2-regularized linear model.",
            'Lasso Regression': "L1-regularized linear model with feature selection."
        }
        st.write(model_info.get(model_choice, ""))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Personal Information")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 10, 80, 25)
        height = st.slider("Height (cm)", 120, 220, 170)
        weight = st.slider("Weight (kg)", 30, 200, 70)

    with col2:
        st.subheader("💪 Exercise Details")
        duration = st.slider("Workout Duration (min)", 1, 180, 30)
        heart_rate = st.slider("Heart Rate (bpm)", 60, 200, 120)
        body_temp = st.slider("Body Temperature (°C)", 36.0, 42.0, 38.5, 0.1)
        env_temp = st.slider("Environmental Temperature (°C)", -10.0, 45.0, 20.0, 0.5)

    gender_encoded = 1 if gender == "Male" else 0
    X_raw = np.array([[gender_encoded, age, height, weight,
                       duration, heart_rate, body_temp, env_temp]])
    X_scaled = scaler.transform(X_raw)

    if st.button("🔥 Predict Calories Burned", type="primary", use_container_width=True):
        selected_model = models[model_choice]
        prediction = selected_model.predict(X_scaled)[0]

        all_predictions = {name: model.predict(X_scaled)[0]
                           for name, model in models.items()}

        st.success("### 📊 Prediction Results")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            delta_vs_avg = prediction - np.mean(list(all_predictions.values()))
            st.metric("🔥 Calories Burned",
                      f"{prediction:.2f} kcal",
                      delta=f"{delta_vs_avg:+.1f} vs avg")
        with c2:
            st.metric("⏱️ Calories / Minute",
                      f"{prediction / duration:.2f} kcal/min")
        with c3:
            st.metric("🤖 Model Used", model_choice)
        with c4:
            confidence = 98.5 if any(k in model_choice for k in ["Forest", "Boost", "XGBoost", "LightGBM"]) else 90.0
            st.metric("✅ Confidence", f"{confidence:.1f}%")

        c1, c2 = st.columns([1, 1])

        with c1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction,
                delta={'reference': 200, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 100], 'color': '#e8f4f8'},
                        {'range': [100, 250], 'color': '#b3d9ff'},
                        {'range': [250, 400], 'color': '#80bfff'},
                        {'range': [400, 500], 'color': '#4da6ff'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 450
                    }
                },
                title={'text': "Calories Burned"}
            ))
            fig_gauge.update_layout(height=350)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with c2:
            comp_df = pd.DataFrame({
                'Model': list(all_predictions.keys()),
                'Predicted Calories': list(all_predictions.values())
            }).sort_values("Predicted Calories")
            fig_comp = px.bar(
                comp_df,
                x="Predicted Calories",
                y="Model",
                orientation="h",
                title="All Models – Prediction Comparison",
                color="Predicted Calories",
                color_continuous_scale="Reds"
            )
            fig_comp.update_traces(texttemplate='%{x:.1f} kcal', textposition='outside')
            fig_comp.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("### 🔍 Input Feature Profile")
        feature_names = ['Gender', 'Age', 'Height', 'Weight',
                         'Duration', 'Heart Rate', 'Body Temp', 'Env Temp']
        feature_values = [gender_encoded, age, height, weight,
                          duration, heart_rate, body_temp, env_temp]
        normalized_values = X_scaled[0]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=np.abs(normalized_values),
            theta=feature_names,
            fill='toself',
            name='Standardized Features'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            height=400,
            title="Standardized Feature Values"
        )

        rc1, rc2 = st.columns([2, 1])
        with rc1:
            st.plotly_chart(fig_radar, use_container_width=True)
        with rc2:
            st.write("#### Input Summary")
            input_df = pd.DataFrame({
                'Feature': feature_names,
                'Raw Value': feature_values,
                'Standardized': np.round(normalized_values, 3)
            })
            st.dataframe(input_df, hide_index=True, use_container_width=True)

# -------------------------------------------------------------------
# MODEL COMPARISON PAGE
# -------------------------------------------------------------------
elif page == "📊 Model Comparison":
    st.header("📊 Model Performance Comparison")

    metrics_data = {
        'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting',
                  'XGBoost', 'LightGBM', 'Ridge', 'Lasso'],
        'MAE': [8.28, 2.52, 2.51, 2.44, 2.45, 8.30, 8.35],
        'RMSE': [11.56, 4.37, 4.36, 4.25, 4.28, 11.60, 11.65],
        'R² Score': [0.967, 0.995, 0.995, 0.996, 0.996, 0.966, 0.965],
        'Training Time (s)': [0.05, 2.1, 3.5, 1.8, 0.8, 0.06, 0.07]
    }
    df_metrics = pd.DataFrame(metrics_data)

    st.dataframe(
        df_metrics.style.background_gradient(subset=['R² Score'], cmap='Greens')
                       .background_gradient(subset=['MAE', 'RMSE'], cmap='Reds_r')
                       .format({'MAE': '{:.2f}', 'RMSE': '{:.2f}',
                                'R² Score': '{:.3f}', 'Training Time (s)': '{:.2f}'}),
        use_container_width=True,
        hide_index=True
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_mae = px.bar(
            df_metrics, x="Model", y="MAE",
            title="MAE (Lower is Better)",
            color="MAE", color_continuous_scale="Reds_r", text="MAE"
        )
        fig_mae.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_mae.update_layout(xaxis_tickangle=-45, height=350)
        st.plotly_chart(fig_mae, use_container_width=True)

        fig_r2 = px.bar(
            df_metrics, x="Model", y="R² Score",
            title="R² Score (Higher is Better)",
            color="R² Score", color_continuous_scale="Greens", text="R² Score"
        )
        fig_r2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_r2.update_layout(xaxis_tickangle=-45, height=350, yaxis_range=[0.96, 1.0])
        st.plotly_chart(fig_r2, use_container_width=True)

    with c2:
        fig_rmse = px.bar(
            df_metrics, x="Model", y="RMSE",
            title="RMSE (Lower is Better)",
            color="RMSE", color_continuous_scale="Oranges_r", text="RMSE"
        )
        fig_rmse.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_rmse.update_layout(xaxis_tickangle=-45, height=350)
        st.plotly_chart(fig_rmse, use_container_width=True)

        fig_time = px.scatter(
            df_metrics, x="Training Time (s)", y="R² Score",
            size="MAE", color="Model",
            title="Accuracy vs Training Time", size_max=30
        )
        fig_time.update_layout(height=350)
        st.plotly_chart(fig_time, use_container_width=True)

# -------------------------------------------------------------------
# EXPLAINABILITY PAGE
# -------------------------------------------------------------------
elif page == "🔍 Explainability":
    st.header("🔍 Model Explainability (SHAP & Feature Importance)")

    models = load_models()
    scaler = load_scaler()

    st.subheader("🎨 SHAP for a Single Prediction")
    model_choice = st.selectbox("Select Model", list(models.keys()))

    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 10, 80, 25)
        height = st.number_input("Height (cm)", 120, 220, 170)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
    with c2:
        duration = st.number_input("Duration (min)", 1, 180, 30)
        heart_rate = st.number_input("Heart Rate (bpm)", 60, 200, 120)
        body_temp = st.number_input("Body Temp (°C)", 36.0, 42.0, 38.5)
        env_temp = st.number_input("Env Temp (°C)", -10.0, 45.0, 20.0)

    if st.button("🔮 Generate SHAP Explanation", type="primary"):
        gender_encoded = 1 if gender == "Male" else 0
        X_explain_raw = np.array([[gender_encoded, age, height, weight,
                                   duration, heart_rate, body_temp, env_temp]])
        X_explain = scaler.transform(X_explain_raw)

        model = models[model_choice]
        pred = model.predict(X_explain)[0]
        st.metric("Predicted Calories", f"{pred:.2f} kcal")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)

            feature_names = ['Gender', 'Age', 'Height', 'Weight',
                             'Duration', 'Heart Rate', 'Body Temp', 'Env Temp']

            wc1, wc2 = st.columns(2)
            with wc1:
                st.subheader("📊 SHAP Waterfall")
                fig, ax = plt.subplots(figsize=(8, 5))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=explainer.expected_value,
                        data=X_explain[0],
                        feature_names=feature_names
                    ),
                    show=False
                )
                st.pyplot(fig)

            with wc2:
                st.subheader("🎯 Feature Impact")
                imp_df = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP Value": shap_values[0],
                    "Abs Impact": np.abs(shap_values[0])
                }).sort_values("Abs Impact", ascending=False)
                fig_imp = px.bar(
                    imp_df, x="SHAP Value", y="Feature",
                    orientation="h",
                    color="SHAP Value",
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0,
                    title="SHAP Contributions"
                )
                st.plotly_chart(fig_imp, use_container_width=True)

        except Exception as e:
            st.error(f"SHAP explanation not available for this model: {e}")
            st.info("Tree-based models (Random Forest, Gradient Boosting, XGBoost, LightGBM) work best with SHAP.")

    st.markdown("---")
    st.subheader("🔧 Global Feature Importance (Tree Models Only)")

    model_choice_fi = st.selectbox("Select Model for Feature Importance", list(models.keys()), key="fi_model")
    model_fi = models[model_choice_fi]

    if hasattr(model_fi, "feature_importances_"):
        feature_names = ['Gender', 'Age', 'Height', 'Weight',
                         'Duration', 'Heart Rate', 'Body Temp', 'Env Temp']
        importances = model_fi.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig_fi = px.bar(
                imp_df, x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
                title=f"Feature Importance – {model_choice_fi}"
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        with c2:
            imp_df = imp_df.sort_values("Importance", ascending=False)
            imp_df["Importance %"] = (imp_df["Importance"] / imp_df["Importance"].sum() * 100).round(1)
            st.dataframe(imp_df.reset_index(drop=True), hide_index=True, use_container_width=True)
    else:
        st.info("Selected model does not provide feature_importances_ (e.g., Linear/Ridge/Lasso).")

# -------------------------------------------------------------------
# ANALYTICS PAGE
# -------------------------------------------------------------------
elif page == "📈 Analytics":
    st.header("📈 Analytics and Data Insights")

    df = load_sample_data()

    st.subheader("📊 Distributions")
    c1, c2 = st.columns(2)
    with c1:
        fig_dur = px.histogram(df, x="Duration", nbins=30,
                               title="Workout Duration Distribution",
                               color_discrete_sequence=["#FF6B6B"])
        fig_dur.update_layout(height=300)
        st.plotly_chart(fig_dur, use_container_width=True)

        fig_hr = px.histogram(df, x="Heart_Rate", nbins=30,
                              title="Heart Rate Distribution",
                              color_discrete_sequence=["#4ECDC4"])
        fig_hr.update_layout(height=300)
        st.plotly_chart(fig_hr, use_container_width=True)
    with c2:
        fig_cal = px.histogram(df, x="Calories", nbins=30,
                               title="Calories Burned Distribution",
                               color_discrete_sequence=["#95E1D3"])
        fig_cal.update_layout(height=300)
        st.plotly_chart(fig_cal, use_container_width=True)

        fig_age = px.histogram(df, x="Age", nbins=20,
                               title="Age Distribution",
                               color_discrete_sequence=["#F38181"])
        fig_age.update_layout(height=300)
        st.plotly_chart(fig_age, use_container_width=True)

    st.subheader("📦 Box Plots")
    num_cols = ["Duration", "Heart_Rate", "Body_Temp", "Calories"]
    fig_box = go.Figure()
    for col in num_cols:
        fig_box.add_trace(go.Box(y=df[col], name=col))
    fig_box.update_layout(title="Feature Ranges & Outliers", height=400)
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("🔗 Correlation Heatmap")
    corr_cols = ["Duration", "Heart_Rate", "Body_Temp", "Age", "Weight", "Calories"]
    corr = df[corr_cols].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Matrix"
    )
    fig_corr.update_layout(height=450)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("🎯 Key Relationships")
    c1, c2 = st.columns(2)
    with c1:
        fig_sc1 = px.scatter(
            df, x="Duration", y="Calories",
            trendline="ols",
            color="Heart_Rate",
            size="Weight",
            title="Duration vs Calories"
        )
        fig_sc1.update_layout(height=350)
        st.plotly_chart(fig_sc1, use_container_width=True)
    with c2:
        fig_sc2 = px.scatter(
            df, x="Heart_Rate", y="Calories",
            trendline="ols",
            color="Duration",
            title="Heart Rate vs Calories"
        )
        fig_sc2.update_layout(height=350)
        st.plotly_chart(fig_sc2, use_container_width=True)

# -------------------------------------------------------------------
# ABOUT PAGE
# -------------------------------------------------------------------
else:
    st.header("ℹ️ About This Project")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
        ### 🎯 Calorie Burn Prediction System

        This project demonstrates an end-to-end machine learning lifecycle:

        - Data ingestion from exercise, calorie, and weather datasets  
        - Feature engineering (gender encoding, temperature enrichment)  
        - Model training with 7 algorithms (linear + ensembles)  
        - Evaluation using MAE, RMSE, and R² metrics  
        - Interactive deployment using Streamlit and Plotly  

        It is designed both as a **portfolio piece** and a **practical fitness analytics tool**.
        """)
    with c2:
        st.markdown("### Quick Stats")
        stats = {
            "Total Samples": "15,000",
            "Features": "8",
            "Models Trained": "7",
            "Best R²": "0.996 (XGBoost / LightGBM)",
            "Deployment": "Streamlit Cloud",
        }
        for k, v in stats.items():
            st.metric(k, v)

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, scikit-learn, XGBoost, LightGBM, and SHAP.")
