import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import os
import warnings
warnings.filterwarnings('ignore')

# Create directory for output
os.makedirs('visualizations', exist_ok=True)

# Constants
TARGET = "target"
CASE_ID = "case_id"
RANDOM_STATE = 42

def find_optimal_threshold(y_true, y_pred_proba):
    """Find optimal threshold based on precision-recall curve"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    # F1 score = 2 * (precision * recall) / (precision + recall)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    # Find threshold that maximizes F1
    optimal_idx = np.argmax(f1_scores[:-1])  # Last element doesn't have threshold
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

# Load data
train = pd.read_parquet("data/train_data.parquet")
test = pd.read_parquet("data/test_data.parquet")

# Data exploration and preprocessing
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Check for missing values
missing_train = train.isnull().sum()
print("Missing values in train:\n", missing_train[missing_train > 0])

# Fill missing values if any
# Numeric columns: fill with median
# Categorical columns: fill with most frequent value
numeric_cols = train.select_dtypes(include=['number']).columns
categorical_cols = train.select_dtypes(include=['object']).columns

for col in numeric_cols:
    if train[col].isnull().sum() > 0:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)

for col in categorical_cols:
    if train[col].isnull().sum() > 0:
        mode_val = train[col].mode()[0]
        train[col] = train[col].fillna(mode_val)
        test[col] = test[col].fillna(mode_val)

# Class distribution and weights
class_counts = train[TARGET].value_counts()
print("Class distribution:", class_counts)
scale = class_counts[0] / class_counts[1]
class_weights = {0: 1, 1: scale}
print(f"Using class weights: {class_weights}")

# Prepare features
y = train[TARGET]
X = train.drop(columns=[TARGET, CASE_ID])
X_test = test.drop(columns=[CASE_ID])
cat_features = [col for col in X.columns if X[col].dtype == 'object']

# Train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Ensure categorical columns are strings
X_train[cat_features] = X_train[cat_features].astype(str)
X_valid[cat_features] = X_valid[cat_features].astype(str)
X_test[cat_features] = X_test[cat_features].astype(str)

# Create data pools
train_pool = Pool(X_train, y_train, cat_features=cat_features)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)
test_pool = Pool(X_test, cat_features=cat_features)


# Model training with selected parameters
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    od_type="Iter",
    od_wait=100,
    class_weights=class_weights,
    verbose=200
)

model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

# Validation performance
val_pred = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, val_pred)
gini = 2 * auc - 1
print(f"Validation AUC = {auc:.5f}")
print(f"Gini coefficient = {gini:.5f}")

# Find optimal threshold
optimal_threshold = find_optimal_threshold(y_valid, val_pred)
print(f"Optimal threshold: {optimal_threshold:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_valid, val_pred)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', lw=2.5)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visualizations/roc_curve_catboost.png', dpi=300)
plt.close()

# Precision-Recall Curve
precisions, recalls, thresholds = precision_recall_curve(y_valid, val_pred)
plt.figure(figsize=(10, 8))
plt.plot(recalls, precisions, lw=2.5)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visualizations/pr_curve_catboost.png', dpi=300)
plt.close()

# Feature importance
feature_importance = model.get_feature_importance()
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(12, 10))
top_features = importance_df.head(20)
plt.barh(range(len(top_features)), top_features['Importance'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.savefig('visualizations/feature_importance_catboost.png')
plt.close()

# Predict test set with optimal threshold
test_pred_proba = model.predict_proba(test_pool)[:, 1]
test_pred_binary = (test_pred_proba >= optimal_threshold).astype(int)

# Create submission with probability score and predicted label
submission = pd.DataFrame({
    CASE_ID: test[CASE_ID],
    "score": test_pred_proba,
    "predicted_label": test_pred_binary,
    "actual_label": test[TARGET]
})

submission.to_csv("submission_catboost.csv", index=False)
print("Submission file created.")