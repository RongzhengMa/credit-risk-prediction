import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
)
import os

# 1. Setup 
os.makedirs('visualizations', exist_ok=True)

TARGET = "target"
CASE_ID = "case_id"
RANDOM_STATE = 42

# 2. Load data 
train = pd.read_parquet("data/train_data.parquet")
test = pd.read_parquet("data/test_data.parquet")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# 3. Define features 
y = train[TARGET]
X = train.drop(columns=[TARGET, CASE_ID])
X_test = test.drop(columns=[CASE_ID])

# 4. Handle categorical features 
cat_features = list(X.select_dtypes(include=["object", "category"]).columns)
X[cat_features] = X[cat_features].astype("category")
X_test[cat_features] = X_test[cat_features].astype("category")

# 5. Train-validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# 6. Compute class weights for imbalance 
pos_ratio = np.mean(y_tr)
neg_ratio = 1 - pos_ratio
class_weight = {0: 1.0, 1: neg_ratio / pos_ratio}
print(f"Class weights: {class_weight}")

# 7. Train LightGBM model
lgbm = LGBMClassifier(
    boosting_type="gbdt",
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    n_estimators=4000,
    max_depth=6,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight=class_weight,
    random_state=RANDOM_STATE
)

lgbm.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[early_stopping(300), log_evaluation(200)],
    categorical_feature=cat_features
)

# 8. Validation evaluation 
val_pred = lgbm.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_pred)
gini = 2 * auc - 1
print(f"Validation AUC  = {auc:.5f}")
print(f"Validation Gini = {gini:.5f}")

# 9. Find optimal threshold (based on F1)
precisions, recalls, thresholds = precision_recall_curve(y_val, val_pred)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = np.argmax(f1_scores[:-1])
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold based on F1: {optimal_threshold:.4f}")

# 10. ROC curve 
fpr, tpr, _ = roc_curve(y_val, val_pred)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', lw=2.5)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("visualizations/lgbm_roc_curve.png", dpi=300)
plt.close()

# 11. Precision-Recall curve 
plt.figure(figsize=(10, 8))
plt.plot(recalls, precisions, lw=2.5)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("visualizations/lgbm_pr_curve.png", dpi=300)
plt.close()

# 12. Feature importance 
importances = lgbm.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values("Importance", ascending=False)

plt.figure(figsize=(12, 10))
top_features = importance_df.head(20)
plt.barh(range(len(top_features)), top_features['Importance'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.savefig("visualizations/lgbm_feature_importance.png", dpi=300)
plt.close()

# Predict on test set 
X_test = X_test[X.columns]  # Ensure same column order
test_pred_proba = lgbm.predict_proba(X_test)[:, 1]
test_pred_label = (test_pred_proba >= optimal_threshold).astype(int)

# Save results (with actual labels if available) 
submission = pd.DataFrame({
    CASE_ID: test[CASE_ID],
    "score": test_pred_proba,
    "predicted_label": test_pred_label
})

if TARGET in test.columns:
    submission["actual_label"] = test[TARGET]

submission.to_csv("submission_lgbm.csv", index=False)
print("Saved: submission_lgbm.csv")
