import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# ----------- Settings & Load Data -----------
st.set_page_config(page_title="Credit Risk Predictor", layout="wide")
sns.set_style("whitegrid")

@st.cache_data
def load_data():
    return pd.read_csv("data/loan_data.csv")

df = load_data()
pipeline = joblib.load("model/credit_risk_pipeline.pkl")
threshold = joblib.load("model/decision_threshold.pkl")

# ----------- Sidebar Navigation -----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Loan Predictor", "Data Insights", "Model Performance"])

# ============================================
# PAGE 1: LOAN PREDICTOR
# ============================================
if page == "Loan Predictor":
    st.title("🏦 Credit Risk Prediction System")
    st.write("Enter applicant details below to assess credit risk.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 18, 80, 30)
        income = st.number_input("Annual Income", 10000, 300000, 50000)
        employment = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed"])
        loan_amt = st.number_input("Loan Amount", 1000, 500000, 100000)
        
    with col2:
        duration = st.number_input("Loan Duration (months)", 6, 120, 24)
        credit_score = st.number_input("Credit Score", 300, 850, 650)
        defaults = st.number_input("Previous Defaults", 0, 10, 0)
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        dependents = st.number_input("Dependents", 0, 10, 0)

    if st.button("Predict Credit Risk", use_container_width=True):
        input_data = pd.DataFrame({
            "Age": [age], "Income": [income], "Employment_Type": [employment],
            "Loan_Amount": [loan_amt], "Loan_Duration": [duration],
            "Credit_Score": [credit_score], "Previous_Defaults": [defaults],
            "Marital_Status": [marital], "Dependents": [dependents]
        })

        prob = pipeline.predict_proba(input_data)[0][1]
        risk = "High Risk" if prob >= threshold else "Low Risk"

        st.divider()
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Risk Level", risk)
        res_col2.metric("Risk Probability", f"{prob:.2%}")

        if risk == "High Risk":
            st.error("⚠️ Warning: This applicant exceeds the risk threshold.")
        else:
            st.success("✅ Approval Recommended: Applicant shows low risk characteristics.")

# ============================================
# PAGE 2: DATA INSIGHTS (From Step 1)
# ============================================
elif page == "Data Insights":
    st.title("📊 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Risk', data=df, palette='viridis', ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Credit Score vs. Risk")
        fig, ax = plt.subplots()
        sns.boxplot(x='Risk', y='Credit_Score', data=df, ax=ax)
        st.pyplot(fig)

    st.subheader("Income Distribution by Employment Type")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.violinplot(x='Employment_Type', y='Income', hue='Risk', data=df, split=True, ax=ax)
    st.pyplot(fig)

# ============================================
# PAGE 3: MODEL PERFORMANCE (From Step 4)
# ============================================
elif page == "Model Performance":
    st.title("📈 Model Evaluation Metrics")
    
    # Generate ROC Data for display
    # (In a real app, you'd load pre-saved test metrics)
    X = df.drop(columns=["Risk"])
    y = df["Risk"].map({"Low": 0, "High": 1})
    y_probs = pipeline.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_probs)
    roc_auc = auc(fpr, tpr)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("### Model Specs")
        st.write("- **Algorithm:** Random Forest Classifier")
        st.write(f"- **Decision Threshold:** {threshold}")
        st.write(f"- **ROC-AUC Score:** {roc_auc:.4f}")
        st.info("The threshold is tuned to 0.4 to prioritize catching High Risk applicants (Higher Recall).")

    with col2:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        st.pyplot(fig)
