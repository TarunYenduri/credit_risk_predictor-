# ============================================
# CREDIT RISK PREDICTION SYSTEM
# STEP 3: BASELINE MODEL TRAINING (IDLE)
# ============================================

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ----------- Load Preprocessed Data -----------
X_train = joblib.load("X_train.pkl")
X_test = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
y_test = joblib.load("y_test.pkl")

# ----------- Logistic Regression -----------
log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

# ----------- Random Forest -----------
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# ----------- Evaluation Function -----------
def evaluate_model(name, y_true, y_pred):
    print("\n==========", name, "==========")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, pos_label="High"))
    print("Recall   :", recall_score(y_true, y_pred, pos_label="High"))
    print("F1 Score :", f1_score(y_true, y_pred, pos_label="High"))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)

# ----------- Save Best Model (Temporary) -----------
joblib.dump(rf_model, "model/credit_risk_model.pkl")

print("\nSTEP 3 COMPLETED SUCCESSFULLY")
