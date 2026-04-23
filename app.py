import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="HDI Predictor", layout="centered")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# -------------------------------
# TITLE
# -------------------------------
st.title("🌍 Human Development Index Predictor")
st.markdown("### 📊 Decision Support Dashboard")

st.write("Simulate how **health, education, and income** affect development.")

st.markdown("---")

# -------------------------------
# INPUTS
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    life_exp = st.number_input("🧬 Life Expectancy", 40.0, 90.0, 70.0)
    expected_school = st.number_input("📚 Expected Schooling", 0.0, 25.0, 12.0)

with col2:
    mean_school = st.number_input("🎓 Mean Schooling", 0.0, 20.0, 8.0)
    gni = st.number_input("💰 GNI per Capita", 100.0, 150000.0, 10000.0)

st.markdown("---")

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict HDI"):

    input_data = np.array([[life_exp, expected_school, mean_school, gni]])
    prediction = model.predict(input_data)[0]

    # -------------------------------
    # RESULT
    # -------------------------------
    st.subheader("📈 Prediction Result")
    st.success(f"Predicted HDI: **{round(prediction, 3)}**")

    # Progress bar
    st.progress(min(max(prediction, 0.0), 1.0))
    st.write(f"HDI Score Level: {round(prediction*100,1)}%")

    # -------------------------------
    # CATEGORY
    # -------------------------------
    st.subheader("📊 Development Category")

    if prediction >= 0.8:
        st.success("🟢 Very High Development")
        insight = "Focus on innovation and sustainability."
    elif prediction >= 0.7:
        st.info("🔵 High Development")
        insight = "Improve income equality and education quality."
    elif prediction >= 0.55:
        st.warning("🟡 Medium Development")
        insight = "Invest in education, healthcare, and economic growth."
    else:
        st.error("🔴 Low Development")
        insight = "Urgent focus on basic development sectors."

    st.write(f"💡 {insight}")

    st.markdown("---")

    # -------------------------------
    # FEATURE VISUALIZATION
    # -------------------------------
    st.markdown("### 📊 Input Feature Comparison")

    feature_data = pd.DataFrame({
        "Feature": ["Life Expectancy", "Expected Schooling", "Mean Schooling", "GNI"],
        "Value": [life_exp, expected_school, mean_school, gni]
    })

    st.bar_chart(feature_data.set_index("Feature"))
    # -------------------------------
    # COMPARISON WITH IDEAL
    # -------------------------------
    st.markdown("### 📊 HDI Comparison")

    comparison = pd.DataFrame({
        "Type": ["Predicted HDI", "Ideal HDI"],
        "Value": [prediction, 0.85]
    })

    st.bar_chart(comparison.set_index("Type"))

    # -------------------------------
    # DASHBOARD METRICS
    # -------------------------------
    st.markdown("### 📊 Key Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted HDI", round(prediction, 3))

    with col2:
        st.metric("Target HDI", 0.85)

    # -------------------------------
    # INSIGHTS
    # -------------------------------
    st.markdown("### 🧠 Key Insights")

    st.write("""
    - 📚 Education has the strongest impact on HDI  
    - 💰 Income significantly influences development  
    - 🧬 Health improvements increase HDI  
    - ⚖️ Balanced growth leads to higher development  
    """)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("🎓 MBA ABA Project | HDI Prediction Dashboard | UNDP Data")
