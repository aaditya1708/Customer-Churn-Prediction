# 🏦 Customer Churn Prediction System

## 📌 Project Overview

Customer churn is one of the most critical challenges faced by banks and financial institutions. This project predicts whether a customer is likely to leave the bank based on demographic, financial, and banking behavior attributes.

The solution uses Machine Learning to identify customers at risk of churning, helping organizations take proactive retention measures.

---

## 🎯 Problem Statement

The objective of this project is to build a machine learning model capable of predicting customer churn using customer profile and banking activity data.

---

## 📊 Dataset Information

The dataset contains customer information such as:

- Gender
- Age
- City
- State
- Account Type
- Occupation
- Credit Score
- Tenure Years
- Account Balance
- Monthly Salary
- Loan Amount
- Fixed Deposit Amount
- Number of Products
- Credit Card Status
- Active Membership Status
- Banking Activity Metrics
- Churn Status (Target Variable)

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Pickle
- GitHub

---

## 🔄 Machine Learning Workflow

### 1. Data Cleaning
- Removed unnecessary columns
- Handled inconsistent categorical values
- Corrected data types
- Managed missing values

### 2. Data Preprocessing
- Stratified Train-Test Split
- Missing Value Imputation
- One-Hot Encoding for categorical variables
- Feature Scaling using StandardScaler

### 3. Model Building

The following models were trained and evaluated:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

### 4. Model Evaluation

Evaluation metrics used:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report

---

## 🏆 Final Model

After comparing multiple machine learning models, **Logistic Regression** was selected as the final model because it achieved the best churn detection performance and provided better recall for churned customers.

---

## 🚀 Streamlit Application

The project includes an interactive Streamlit web application where users can:

- Enter customer information
- Predict churn probability
- View churn risk instantly
- Analyze customer retention likelihood

---

## 📂 Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── preprocessor.pkl
├── logistic_model.pkl
├── requirements.txt
├── README.md
└── images/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
```

Move into the project folder:

```bash
cd customer-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📈 Future Improvements

- Hyperparameter Tuning
- Feature Importance Analysis
- Advanced Ensemble Models
- Cloud Deployment Enhancements
- Real-Time Customer Monitoring

---

## 👨‍💻 Author

Aaditya

Machine Learning | Data Science | Python

---

## ⭐ Project Highlights

✅ End-to-End Machine Learning Project

✅ Data Cleaning & Preprocessing

✅ Multiple Model Comparison

✅ Streamlit Deployment

✅ Real-World Banking Use Case