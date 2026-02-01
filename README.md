# Credit Risk Prediction System 🏦

An end-to-end Machine Learning application designed to predict the likelihood of loan default. This project covers the full data science lifecycle, from synthetic data generation to a deployed Streamlit dashboard.

## 🚀 Features
- **Data Generation:** Custom script to create synthetic financial records.
- **Automated Preprocessing:** Handles scaling, encoding, and missing values using Scikit-Learn Pipelines.
- **Model Comparison:** Evaluates Logistic Regression vs. Random Forest.
- **Business Logic:** Implements a custom decision threshold (0.4) to minimize financial risk by increasing recall for high-risk cases.
- **Interactive Dashboard:** - Real-time loan risk assessment.
  - Exploratory data visualizations.
  - Model performance tracking (ROC curves).

## 📂 Project Structure
- `data/`: Contains the generated `loan_data.csv`.
- `model/`: Stores the serialized pipeline and decision thresholds.
- `step1-5_*.py`: Step-by-step scripts for data processing, training, and model selection.
- `app.py`: The Streamlit application code.



credit-risk-prediction/
├── data/                          # 📊 Raw data storage
│   └── loan_data.csv              # (Created by generate_loan_data.py)
│
├── model/                         # 🤖 Saved models & pipelines
│   ├── preprocessor.pkl           # Preprocessing logic
│   ├── credit_risk_pipeline.pkl   # Full end-to-end pipeline
│   ├── decision_threshold.pkl     # Business risk threshold (0.4)
│   └── final_model.pkl            # Individual trained model
│
├── app/                           # 🌐 Web Application folder
│   └── app.py                     # Streamlit dashboard
│
├── README.md                      # Project documentation
├── requirements.txt               # List of dependencies
│
├── generate_loan_data.py          # Step 0: Data Generation
├── step1_data_understanding.py    # Step 1: EDA
├── step2_preprocessing.py         # Step 2: Cleaning & Scaling
├── step3_model_training.py        # Step 3: Baseline Models
├── step4_model_selection.py       # Step 4: ROC-AUC & Tuning
├── step5_save_pipeline.py         # Step 5: Final Export
│
├── X_train.pkl                    # (Intermediate training data)
├── X_test.pkl                     # (Intermediate testing data)
├── y_train.pkl                    # (Intermediate training labels)
└── y_test.pkl                     # (Intermediate testing labels)

## 🛠️ How to Run
1. **Generate Data:** `python generate_loan_data.py`
2. **Train Pipeline:** Run steps 1 through 5 in order.
3. **Launch App:** `streamlit run app.py`