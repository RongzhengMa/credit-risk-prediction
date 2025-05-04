import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline 

# ========== Step 1: Load & prepare data ==========
df = pd.read_parquet("data/train_data.parquet")
X = df.drop(columns=["case_id", "date_decision", "target"])
y = df["target"]

# ========== Step 2: Train-test split ==========
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
pipeline = Pipeline([
    ('undersample', RandomUnderSampler(sampling_strategy=0.4, random_state=42)),      
])
X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)

# ========== Step 3: Standardize ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)

# ========== Step 4: Lasso for feature selection ==========
lasso = Lasso(alpha=0.005, max_iter=500, tol=1e-2, random_state=42)
lasso.fit(X_train_scaled, y_train_resampled)

sfm = SelectFromModel(lasso, prefit=True)
X_train_sel = sfm.transform(X_train_scaled)
X_val_sel = sfm.transform(X_val_scaled)

n_features = X_train_sel.shape[1]
print(f" Lasso selected {n_features} features.")
if n_features == 0:
    print("⚠️ No features selected by Lasso. Using all features instead.")
    X_train_sel = X_train_scaled
    X_val_sel = X_val_scaled

# ========== Step 5: Train SVM ==========
svm = LinearSVC(class_weight='balanced', max_iter=1000, tol=1e-3, random_state=42)
svm.fit(X_train_sel, y_train_resampled)

# ========== Step 6: Evaluate ==========
y_scores = svm.decision_function(X_val_sel)
auc = roc_auc_score(y_val, y_scores)
print(f"🎯 SVM ROC AUC: {auc:.4f}")

# ========== Step 7: Save Models ==========
import joblib
import os

model_dir = "../output/model"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
joblib.dump(lasso, os.path.join(model_dir, "lasso.pkl"))
joblib.dump(sfm, os.path.join(model_dir, "feature_selector.pkl"))
joblib.dump(svm, os.path.join(model_dir, "linear_svc.pkl"))
np.save(os.path.join(model_dir, "selected_feature_mask.npy"), sfm.get_support())

print("✅ 所有模型保存成功：")
print("- scaler.pkl")
print("- lasso.pkl")
print("- feature_selector.pkl")
print("- linear_svc.pkl")
print("- selected_feature_mask.npy")

# ========== Step 8: Visualizations ==========

import matplotlib.pyplot as plt

# 📊 1. Lasso Coefficient Bar Chart
coeffs = lasso.coef_
features = X.columns
nonzero_idx = coeffs != 0
plt.figure(figsize=(10, 12))
plt.barh(np.array(features)[nonzero_idx], coeffs[nonzero_idx])
plt.title("Lasso Selected Feature Coefficients", fontsize=14)
plt.xlabel("Coefficient Value", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig("visualizations/lasso_features.png")
plt.close()

# 📈 2. Lasso ROC Curve
y_pred_lasso = lasso.predict(X_val_scaled)
lasso_auc = roc_auc_score(y_val, y_pred_lasso)
fpr_lasso, tpr_lasso, _ = roc_curve(y_val, y_pred_lasso)
plt.figure(figsize=(6, 4))
plt.plot(fpr_lasso, tpr_lasso, label=f"Lasso (AUC = {lasso_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Lasso")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizations/lasso_roc.png")
plt.close()

# 📉 3. Lasso Prediction Distribution
plt.figure(figsize=(6, 4))
plt.hist(y_pred_lasso, bins=50, color='orange')
plt.title("Lasso Prediction Distribution")
plt.xlabel("Predicted Value")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("visualizations/lasso_pred_dist.png")
plt.close()

# 🔍 4. Lasso Scatter: Predicted vs True
plt.figure(figsize=(6, 4))
plt.scatter(y_val, y_pred_lasso, alpha=0.3)
plt.xlabel("True TARGET")
plt.ylabel("Predicted (Lasso)")
plt.title("Lasso Predicted vs True")
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizations/lasso_scatter_truth.png")
plt.close()

# 📈 5. LinearSVC ROC Curve
try:
    y_scores_svm = svm.decision_function(X_val_sel)
    svm_auc = roc_auc_score(y_val, y_scores_svm)
    fpr_svm, tpr_svm, _ = roc_curve(y_val, y_scores_svm)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr_svm, tpr_svm, label=f"LinearSVC (AUC = {svm_auc:.2f})", color='green')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - LinearSVC (Lasso-selected)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("visualizations/svm_roc_lasso_selected.png")
    plt.close()
    print(f"✅ LinearSVC ROC AUC: {svm_auc:.4f}")
except Exception as e:
    print("⚠️ SVM ROC Curve not generated.")
    print("Reason:", str(e))

print("✅ All charts saved:")
print("visualizations/lasso_features.png")
print("visualizations/lasso_roc.png")
print("visualizations/lasso_pred_dist.png")
print("visualizations/lasso_scatter_truth.png")
print("visualizations/svm_roc_lasso_selected.png (if no error)")
