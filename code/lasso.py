# ===============================
# Imports and Setup
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)


# ===============================
# Load Model Components
# ===============================
scaler = joblib.load("../output/model/scaler.pkl")
lasso = joblib.load("../output/model/lasso.pkl")
sfm = joblib.load("../output/model/feature_selector.pkl")
svm = joblib.load("../output/model/linear_svc.pkl")

# ===============================
# Load and Prepare Test Data
# ===============================
df_test = pd.read_parquet("data/train_data.parquet")
print(f"Test data shape: {df_test.shape}")

X_test = df_test.drop(columns=["case_id", "date_decision", "target"])
y_true = df_test["target"].astype(np.float32)

# Apply scaler → feature selection
X_test_scaled = scaler.transform(X_test).astype(np.float32)
X_test_sel = sfm.transform(X_test_scaled)

# ===============================
# Make Predictions
# ===============================
y_scores = svm.decision_function(X_test_sel)
y_pred = (y_scores > 0).astype("int")  # LinearSVC 默认 threshold 为 0

# ===============================
# Print Evaluation Metrics
# ===============================
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
print(f"AUC:       {roc_auc_score(y_true, y_scores):.4f}")

# ===============================
# Confusion Matrix Visualization
# ===============================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Event', 'Event'],
            yticklabels=['No Event', 'Event'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ===============================
# ROC Curve Plot
# ===============================
fpr, tpr, _ = roc_curve(y_true, y_scores)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_scores):.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# ===============================
# Precision-Recall Curve and Best F1 Threshold
# ===============================
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
pr_auc = average_precision_score(y_true, y_scores)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"\nPR AUC = {pr_auc:.4f}")
print(f"Best threshold (by F1): {best_threshold:.4f}")

# PR Curve
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall (PR) Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# F1 Score vs Threshold Plot
plt.figure(figsize=(6, 5))
plt.plot(thresholds, f1_scores[:-1], label='F1 Score')
plt.axvline(best_threshold, color='red', linestyle='--', label='Best Threshold')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# Distribution of Positive Class Scores
# ===============================
pos_scores = y_scores[y_true == 1]

plt.figure(figsize=(7, 5))
plt.hist(pos_scores, bins=50, color='skyblue', edgecolor='black')
plt.axvline(0, color='red', linestyle='--', label='Default Threshold = 0')
plt.title("Decision Function Scores for Positive Class (Test Set)")
plt.xlabel("Decision Function Score")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()

df_preds = pd.DataFrame({
    "case_id": df_test["case_id"].values,
    "score": y_scores.flatten(),
    "predicted_label": y_pred.flatten(),
    "actual_label": y_true.values
})

# 保存完整预测结果
df_preds.to_csv("result/submission_lasso_svm.csv", index=False)


print("saved：")
print("- result/submission_lasso_svm.csv")
