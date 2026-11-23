# Sri Lanka Telco Customer Churn Risk & Ensemble Analytics Dashboard

An end-to-end Machine Learning project and interactive web application built to predict customer churn in the Sri Lankan telecommunications sector using advanced Ensemble Learning paradigms.

---

## Project Overview
Customer retention is a critical objective for telecommunication providers in Sri Lanka. This project processes a 90 day time-series customer usage dataset (tracking daily app MB consumption across YouTube, Facebook, TikTok, WhatsApp, Helakuru, voice call minutes, and reload spend) to forecast churn risk. 

We systematically design, train, tune, and evaluate four distinct modeling paradigms:
1. Bagging: Random Forest Classifier evaluated with Out-of-Bag (OOB) scoring.
2. Boosting: Extreme Gradient Boosting (XGBoost) with class imbalance loss weighting.
3. Stacking: Heterogeneous Meta-Classifier combining Random Forest, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN) feeding into a Logistic Regression meta-learner.
4. Hybrid Model: Novel combination of an Ensemble Method (Random Forest) and Deep Representation Learning (Artificial Neural Network / Multi-Layer Perceptron).

---

## Dataset & Feature Engineering Pipeline
- Dataset Focus: Sri Lankan Telecom Customer Churn Profiles.
- Raw Input: 90 daily activity logs (Day_1 to Day_90) mapped across individual usage_type records.
- Aggregation & Pivoting:
  - total_usage: 90-day cumulative usage volume.
  - avg_daily_usage: Daily mean traffic.
  - recent_30d_usage: Traffic over days 61–90 to detect sudden behavioral drops prior to churn.
- Preprocessing: Missing value imputation, Label Encoding for categorical variables, Stratified Train-Test Split (80% Train / 20% Test), and Z-Score Feature Scaling (StandardScaler).

---

## Model Performance Summary

| Model Paradigm | Architecture / Configuration | Key Metric Focus |
| :--- | :--- | :--- |
| Random Forest (Bagging) | 100 Trees, Bootstrap Aggregation, OOB Scoring | Variance Reduction & OOB Accuracy |
| XGBoost (Boosting) | Gradient Boosted Trees (scale_pos_weight tuned) | Bias/Variance Minimization & LogLoss |
| Stacking Classifier | Base: RF + SVM + KNN -> Meta: Logistic Regression | Cross-Validated Probability Stacking |
| Hybrid Model (RF + ANN) | Soft Voting: Random Forest + MLP (64x32 ReLU) | Non-Linear Decision Trees + Continuous Neural Manifold |

---

## Repository Structure

sl_telco_churn_assignment/
├── data/
│   └── raw/
│       ├── main.csv           # 90-day time-series usage logs
│       ├── train.csv          # Training customer_id -> churn labels
│       └── test.csv           # Testing customer_id -> churn labels
├── models/
│   ├── scaler.pkl             # Fitted StandardScaler object
│   ├── best_rf.pkl            # Trained Random Forest model
│   ├── best_xgb.pkl           # Trained XGBoost model
│   ├── stacking_model.pkl     # Trained Stacking Classifier
│   └── hybrid_model.pkl       # Trained Hybrid (RF + ANN) model
├── app.py                     # Interactive Streamlit Web Dashboard
├── churn_analysis.ipynb       # Complete Jupyter Data & Modeling Notebook
├── requirements.txt           # Environment dependencies
└── README.md                  # Project documentation & GenAI declaration

---

## Installation & How to Run Locally

### 1. Clone the Repository
git clone <your-repository-url>
cd sl_telco_churn_assignment

### 2. Create and Activate Virtual Environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Launch the Interactive Dashboard
streamlit run app.py

---

## AI Usage Acknowledgment

## AI Usage Acknowledgment

Generative AI (Gemini) was utilized during this project for:
- **Understanding & Concepts**: Assisting with technical understanding of ensemble learning workflows, data preprocessing logic, and code debugging.
- **Frontend Development**: Structuring and styling the interactive Streamlit web application (`app.py`).
---

