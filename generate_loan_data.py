# ============================================
# GENERATE LOAN DATASET FOR CREDIT RISK PROJECT
# ============================================

import pandas as pd
import numpy as np
import os

# ----------- Create data folder if not exists -----------
os.makedirs("data", exist_ok=True)

np.random.seed(42)
n_samples = 1000

# ----------- Generate Features -----------
data = {
    "Age": np.random.randint(21, 65, n_samples),
    "Income": np.random.randint(20000, 150000, n_samples),
    "Employment_Type": np.random.choice(
        ["Salaried", "Self-Employed", "Unemployed"], n_samples
    ),
    "Loan_Amount": np.random.randint(5000, 500000, n_samples),
    "Loan_Duration": np.random.randint(6, 60, n_samples),
    "Credit_Score": np.random.randint(300, 850, n_samples),
    "Previous_Defaults": np.random.randint(0, 5, n_samples),
    "Marital_Status": np.random.choice(
        ["Single", "Married", "Divorced"], n_samples
    ),
    "Dependents": np.random.randint(0, 5, n_samples)
}

df = pd.DataFrame(data)

# ----------- Create Risk Label Logic -----------
risk_conditions = (
    (df["Credit_Score"] < 600) |
    (df["Previous_Defaults"] > 2) |
    (df["Income"] < 40000)
)

df["Risk"] = np.where(risk_conditions, "High", "Low")

# ----------- Save Dataset -----------
file_path = "data/loan_data.csv"
df.to_csv(file_path, index=False)

print("Dataset generated successfully!")
print("File saved at:", file_path)
print("\nPreview:")
print(df.head())
