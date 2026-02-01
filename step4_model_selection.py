# ============================================
# CREDIT RISK PREDICTION SYSTEM
# STEP 4: MODEL SELECTION + ROC + THRESHOLD
# ============================================

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ----------- Load Data -----------
X_train = joblib.load("X_train.pkl")
X_test = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
y_test = joblib.load("y_test.pkl")

# ----------- Train Models Again (Clean) -----------
log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

log_reg.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# ----------- Predict Probabilities -----------
y_prob_lr = log_reg.predict_proba(X_test)[:, 1]
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# ----------- ROC-AUC -----------
roc_lr = roc_auc_score(y_test.map({"Low": 0, "High": 1}), y_prob_lr)
roc_rf = roc_auc_score(y_test.map({"Low": 0, "High": 1}), y_prob_rf)

print("ROC-AUC Logistic Regression:", roc_lr)
print("ROC-AUC Random Forest      :", roc_rf)

# ----------- ROC Curve Plot -----------
fpr_lr, tpr_lr, _ = roc_curve(y_test.map({"Low": 0, "High": 1}), y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test.map({"Low": 0, "High": 1}), y_prob_rf)

plt.figure(figsize=(7,5))
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression")
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# ----------- Threshold Tuning (Random Forest) -----------
threshold = 0.4  # lower threshold = higher recall
y_pred_custom = np.where(y_prob_rf >= threshold, "High", "Low")

print("\nClassification Report (RF @ threshold = 0.4):")
print(classification_report(y_test, y_pred_custom))

# ----------- Save FINAL Model -----------
joblib.dump(rf_model, "model/final_credit_risk_model.pkl")

print("\nFINAL MODEL SAVED: model/final_credit_risk_model.pkl")
print("STEP 4 COMPLETED SUCCESSFULLY")
