# ============================================
# CREDIT RISK PREDICTION SYSTEM
# STEP 1: DATA LOADING & UNDERSTANDING (IDLE)
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------- Settings -----------
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")

# ----------- Load Dataset -----------
data_path = "data/loan_data.csv"
df = pd.read_csv(data_path)

# ----------- Preview Data -----------
print("\nFIRST 5 ROWS OF DATASET:")
print(df.head())

# ----------- Dataset Shape -----------
print("\nDATASET SHAPE (rows, columns):")
print(df.shape)

# ----------- Column Information -----------
print("\nCOLUMN INFO:")
print(df.info())

# ----------- Missing Values -----------
print("\nMISSING VALUES PER COLUMN:")
print(df.isnull().sum())

# ----------- Target Variable Check -----------
target_column = "Risk"   # CHANGE if needed

print("\nTARGET VARIABLE DISTRIBUTION:")
print(df[target_column].value_counts())

print("\nTARGET VARIABLE PERCENTAGE:")
print(df[target_column].value_counts(normalize=True) * 100)

# ----------- Statistical Summary -----------
print("\nSTATISTICAL SUMMARY (NUMERICAL FEATURES):")
print(df.describe())

# ----------- Feature Type Identification -----------
categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

print("\nCATEGORICAL COLUMNS:")
print(list(categorical_cols))

print("\nNUMERICAL COLUMNS:")
print(list(numerical_cols))

# ----------- Class Distribution Plot -----------
plt.figure(figsize=(6,4))
sns.countplot(x=target_column, data=df)
plt.title("Target Variable Distribution")
plt.tight_layout()
plt.show()

print("\nSTEP 1 COMPLETED SUCCESSFULLY")
