import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve

# ========== Step 1: Load data ==========
df = pd.read_parquet("data/train_data.parquet")
df.columns = df.columns.str.strip().str.upper()

# ========== Step 2: Keep only numeric features ==========
X = df.drop(columns=['TARGET']).select_dtypes(include='number')
y = df['TARGET']

# ========== Step 3: Split data ==========
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========== Step 4: Standardize ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ========== Step 5: Lasso ==========
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_val_scaled)
lasso_auc = roc_auc_score(y_val, y_pred_lasso)
print(f"Lasso ROC AUC: {lasso_auc:.4f}")

# ========== Step 6: Use Lasso-selected features in SVM ==========
selected_features = X.columns[lasso.coef_ != 0]
print(f"📌 Lasso selected {len(selected_features)} features.")

X_selected = X[selected_features]
X_train_sel, X_val_sel, y_train_sel, y_val_sel = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=42
)

scaler_sel = StandardScaler()
X_train_sel_scaled = scaler_sel.fit_transform(X_train_sel)
X_val_sel_scaled = scaler_sel.transform(X_val_sel)

try:
    print("⏳ Training SVM with Lasso-selected features...")
    svm = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    svm.fit(X_train_sel_scaled, y_train_sel)
    y_pred_svm = svm.predict_proba(X_val_sel_scaled)[:, 1]
    svm_auc = roc_auc_score(y_val_sel, y_pred_svm)
    print(f"SVM ROC AUC (after Lasso selection): {svm_auc:.4f}")
    print("Predicted probability (min, max, mean):",
          round(y_pred_svm.min(), 4), round(y_pred_svm.max(), 4), round(y_pred_svm.mean(), 4))
except Exception as e:
    print("❌ SVM training failed.")
    print("Reason:", str(e))
    svm_auc = None
    y_pred_svm = None

# ========== Step 7: Visualizations ==========

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
plt.savefig("lasso_features.png")
plt.close()

# 📈 2. Lasso ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_pred_lasso)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"Lasso (AUC = {lasso_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Lasso")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lasso_roc.png")
plt.close()

# 📉 3. Lasso Prediction Distribution
plt.figure(figsize=(6, 4))
plt.hist(y_pred_lasso, bins=50, color='orange')
plt.title("Lasso Prediction Distribution")
plt.xlabel("Predicted Value")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("lasso_pred_dist.png")
plt.close()

# 🔍 4. Lasso Scatter: Predicted vs True
plt.figure(figsize=(6, 4))
plt.scatter(y_val, y_pred_lasso, alpha=0.3)
plt.xlabel("True TARGET")
plt.ylabel("Predicted (Lasso)")
plt.title("Lasso Predicted vs True")
plt.grid(True)
plt.tight_layout()
plt.savefig("lasso_scatter_truth.png")
plt.close()

# 📈 5. SVM ROC Curve (Lasso-selected features) - only if success
if y_pred_svm is not None:
    fpr_svm, tpr_svm, _ = roc_curve(y_val_sel, y_pred_svm)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {svm_auc:.2f})", color='green')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - SVM (Lasso-selected)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("svm_roc_lasso_selected.png")
    plt.close()

print("✅ All charts saved:")
print("- lasso_features.png")
print("- lasso_roc.png")
print("- lasso_pred_dist.png")
print("- lasso_scatter_truth.png")
if y_pred_svm is not None:
    print("- svm_roc_lasso_selected.png")
else:
    print("⚠️ SVM ROC chart not generated due to training failure.")
