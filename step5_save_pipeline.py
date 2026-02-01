# ============================================
# CREDIT RISK PREDICTION SYSTEM
# STEP 5: SAVE FINAL DEPLOYMENT PIPELINE (FIXED)
# ============================================

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# ----------- Load Original Dataset -----------
df = pd.read_csv("data/loan_data.csv")

target_column = "Risk"
X = df.drop(columns=[target_column])
y = df[target_column]

# ----------- Load Preprocessor -----------
preprocessor = joblib.load("model/preprocessor.pkl")

# ----------- Final Model -----------
final_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

# ----------- Build Full Pipeline -----------
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", final_model)
])

# ----------- Train Pipeline on RAW DATA -----------
pipeline.fit(X, y)

# ----------- Save Pipeline -----------
joblib.dump(pipeline, "model/credit_risk_pipeline.pkl")

# ----------- Save Business Threshold -----------
threshold = 0.4
joblib.dump(threshold, "model/decision_threshold.pkl")

print("PIPELINE SAVED SUCCESSFULLY")
print("- model/credit_risk_pipeline.pkl")
print("- model/decision_threshold.pkl")
print("STEP 5 FIXED & COMPLETED")
