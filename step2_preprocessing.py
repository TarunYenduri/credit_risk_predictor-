# ============================================
# CREDIT RISK PREDICTION SYSTEM
# STEP 2: PREPROCESSING (IDLE VERSION)
# ============================================

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ----------- Load Dataset -----------
data_path = "data/loan_data.csv"
df = pd.read_csv(data_path)

# ----------- Target Column -----------
target_column = "Risk"   # CHANGE if your dataset uses a different name

X = df.drop(columns=[target_column])
y = df[target_column]

# ----------- Identify Feature Types -----------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

print("Numeric features:", list(numeric_features))
print("Categorical features:", list(categorical_features))

# ----------- Pipelines -----------
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# ----------- Train-Test Split -----------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------- Apply Preprocessing -----------
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ----------- Save Everything for STEP 3 -----------
joblib.dump(X_train_processed, "X_train.pkl")
joblib.dump(X_test_processed, "X_test.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_test, "y_test.pkl")
joblib.dump(preprocessor, "model/preprocessor.pkl")

print("\nSTEP 2 COMPLETED SUCCESSFULLY")
print("Saved files:")
print("- X_train.pkl")
print("- X_test.pkl")
print("- y_train.pkl")
print("- y_test.pkl")
print("- model/preprocessor.pkl")
