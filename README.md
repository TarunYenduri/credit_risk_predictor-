# 🏦 Credit Risk Prediction System

An end-to-end Machine Learning solution designed to evaluate loan applications and predict potential credit defaults. This project covers the entire data science lifecycle, featuring a robust Scikit-Learn pipeline and an interactive Streamlit dashboard.

## 📌 Project Overview

The goal of this system is to classify loan applicants as either **"High Risk"** or **"Low Risk"**. Using a synthetic dataset that mimics real-world financial distributions, the project implements automated data cleaning, feature scaling, model comparison, and business-focused threshold tuning to prioritize financial safety.

---

## 🚀 Key Features

* **Automated Pipeline**: Combines preprocessing and modeling into a single `joblib` file for seamless deployment.
* **Advanced Preprocessing**: Handles missing values and performs feature scaling (StandardScaler) and categorical encoding (OneHotEncoder).
* **Risk-Averse Modeling**: Implements a custom decision threshold of **0.4** (down from the default 0.5) to increase the recall for high-risk applicants, ensuring fewer defaults go undetected.
* **Interactive UI**: A Streamlit dashboard for real-time risk assessment and visual data insights.

---

## 📊 Dataset Description

The system uses a synthetic dataset of 1,000 samples generated with specific financial logic:

* **Numerical Features**: Age, Annual Income, Loan Amount, Loan Duration (months), Credit Score, Previous Defaults, and Dependents.
* **Categorical Features**: Employment Type (Salaried, Self-Employed, Unemployed) and Marital Status.
* **Target Variable**: `Risk` ("High" or "Low").
* *Risk Logic*: An applicant is flagged as "High" risk if their Credit Score is < 600, they have > 2 defaults, or their Income is < $40,000.



---

## 🛠️ Project Workflow

### 1. Data Generation & EDA

* `generate_loan_data.py`: Creates the raw dataset in the `data/` directory.
* `step1_data_understanding.py`: Performs Exploratory Data Analysis (EDA) to check class distributions and statistical summaries.

### 2. Preprocessing & Engineering

* `step2_preprocessing.py`: Constructs a `ColumnTransformer` to handle disparate data types and splits the data into training and testing sets.

### 3. Model Training & Selection

* `step3_model_training.py`: Trains baseline Logistic Regression and Random Forest models.
* `step4_model_selection.py`: Compares models using **ROC-AUC** scores and tunes the classification threshold to 0.4 for better risk sensitivity.

### 4. Deployment Pipeline

* `step5_save_pipeline.py`: Fits a final `Pipeline` object on the full dataset and exports it alongside the custom threshold for production use.

---

## 📂 Folder Structure

```text
credit-risk-prediction/
├── data/                          # Raw CSV storage
│   └── loan_data.csv              # Generated dataset
├── model/                         # Saved artifacts
│   ├── preprocessor.pkl           # Preprocessing logic
│   ├── credit_risk_pipeline.pkl   # Full Scikit-Learn Pipeline
│   └── decision_threshold.pkl     # Optimized threshold (0.4)
├── app/                           # Dashboard source
│   └── app.py                     # Streamlit application logic
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies
└── step1-5_*.py                   # Modular pipeline scripts

```

---

## 🚦 Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-risk-prediction.git
cd credit-risk-prediction

```


2. Install dependencies:
```bash
pip install -r requirements.txt

```



### Running the System

1. **Generate Data**: `python generate_loan_data.py`
2. **Train the Pipeline**: Run steps 1 through 5 in sequence.
3. **Start the App**:
```bash
streamlit run app/app.py

```



---

## 📈 Model Performance

* **Primary Metric**: ROC-AUC (Area Under the Receiver Operating Characteristic Curve).
* **Optimization Strategy**: By lowering the threshold to **0.4**, the system achieves higher **Recall** for the "High Risk" class, which is critical for minimizing the cost of lending to potential defaulters.

---

## 💻 Technologies Used

* **Python 3.x**
* **Scikit-Learn**: Pipeline, ColumnTransformer, RandomForest
* **Pandas & NumPy**: Data manipulation
* **Streamlit**: Web deployment
* **Matplotlib & Seaborn**: Data visualization
