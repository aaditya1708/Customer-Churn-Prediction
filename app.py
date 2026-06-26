import streamlit as st
import pandas as pd
import pickle

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

# ================= LOAD MODEL =================

import traceback

try:
    with open("preprocessing.pkl", "rb") as f:
        preprocessor = pickle.load(f)
except Exception as e:
    st.error(f"Error loading preprocessor: {e}")
    st.code(traceback.format_exc())
    st.stop()

with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# ================= CUSTOM CSS =================

st.markdown("""
<style>

.stApp{
    background-color:#f8fafc;
}

.main-header{
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0px 2px 15px rgba(0,0,0,0.08);
    margin-bottom:20px;
}

.title{
    color:#1e293b;
    font-size:40px;
    font-weight:700;
}

.subtitle{
    color:#64748b;
    font-size:16px;
}

.section-card{
    background:white;
    padding:20px;
    border-radius:18px;
    margin-bottom:15px;
    box-shadow:0px 2px 12px rgba(0,0,0,0.05);
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:20px;
    font-weight:bold;
}

.result-success{
    background:#dcfce7;
    padding:25px;
    border-radius:18px;
    text-align:center;
}

.result-danger{
    background:#fee2e2;
    padding:25px;
    border-radius:18px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================

st.markdown("""
<div class="main-header">
<div class="title">🏦 Customer Churn Prediction</div>
<div class="subtitle">
Predict the likelihood of a customer leaving the bank based on
their profile and banking behaviour.
</div>
</div>
""", unsafe_allow_html=True)

# ================= FORM =================

with st.form("customer_form"):

    st.markdown("### 👤 Customer Profile")

    c1, c2 = st.columns(2)

    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 18, 100, step=1)
        city = st.text_input("City")
        state = st.text_input("State")
        occupation = st.text_input("Occupation")

    with c2:
        account_type = st.selectbox(
            "Account Type",
            ["Current", "Savings", "NRI", "Salary"]
        )
        branch = st.text_input("Branch")
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            step=1
        )
        tenure = st.number_input(
            "Tenure Years",
            min_value=0,
            step=1
        )

    st.markdown("### 💰 Financial Information")

    c3, c4 = st.columns(2)

    with c3:
        balance = st.number_input(
            "Account Balance (₹)",
            min_value=0,
            step=1000
        )

        salary = st.number_input(
            "Monthly Salary (₹)",
            min_value=0,
            step=1000
        )

    with c4:
        loan = st.number_input(
            "Loan Amount (₹)",
            min_value=0,
            step=1000
        )

        fd = st.number_input(
            "FD Amount (₹)",
            min_value=0,
            step=1000
        )

    st.markdown("### 🏦 Banking Behaviour")

    c5, c6 = st.columns(2)

    with c5:

        products = st.number_input(
            "Number of Products",
            min_value=0,
            step=1
        )

        transactions = st.number_input(
            "Monthly Transactions",
            min_value=0,
            step=1
        )

        complaints = st.number_input(
            "Complaints Last 12 Months",
            min_value=0,
            step=1
        )

    with c6:

        has_card = st.selectbox(
            "Has Credit Card",
            [0,1]
        )

        active = st.selectbox(
            "Is Active Member",
            [0,1]
        )

        net_banking = st.selectbox(
            "Net Banking Enabled",
            [0,1]
        )

        mobile_banking = st.selectbox(
            "Mobile Banking Enabled",
            [0,1]
        )

    predict = st.form_submit_button(
        "🔍 Predict Churn Risk"
    )

# ================= PREDICTION =================

if predict:

    input_df = pd.DataFrame({

        "Gender":[gender],
        "Age":[age],
        "City":[city],
        "State":[state],
        "AccountType":[account_type],
        "Branch":[branch],
        "Occupation":[occupation],
        "CreditScore":[credit_score],
        "TenureYears":[tenure],
        "AccountBalance_INR":[balance],
        "MonthlySalary_INR":[salary],
        "NumProducts":[products],
        "LoanAmount_INR":[loan],
        "FD_Amount_INR":[fd],
        "HasCreditCard":[has_card],
        "IsActiveMember":[active],
        "MonthlyTransactions":[transactions],
        "ComplaintsLast12M":[complaints],
        "NetBankingEnabled":[net_banking],
        "MobileBankingEnabled":[mobile_banking]

    })

    processed = preprocessor.transform(input_df)

    prediction = model.predict(processed)
    probability = model.predict_proba(processed)

    churn_prob = probability[0][1] * 100

    st.markdown("---")
    st.markdown("## 📊 Prediction Result")

    if prediction[0] == 1:

        st.markdown(f"""
        <div class="result-danger">
            <h2>⚠ High Churn Risk</h2>
            <h1>{churn_prob:.2f}%</h1>
            <p>This customer shows signs of potential churn.</p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result-success">
            <h2>✅ Low Churn Risk</h2>
            <h1>{100-churn_prob:.2f}%</h1>
            <p>This customer is likely to remain with the bank.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.progress(int(churn_prob))

    st.info(
        f"Predicted churn probability: {churn_prob:.2f}%"
    )