import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, precision_score, recall_score, confusion_matrix,roc_auc_score,roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

file1_path = "result/stacked_lightgbm_dnn.csv"
file2_path = "result/submission_catboost.csv"
file3_path = "result/submission_lasso_svm.csv"
file4_path = "result/submission_lgbm.csv"
file5_path = "result/submission_neural_network.csv"
df_staking = pd.read_csv(file1_path)
df_catboost = pd.read_csv(file2_path)
df_lasso = pd.read_csv(file3_path)
df_lgbm = pd.read_csv(file4_path)
df_network = pd.read_csv(file5_path)
df_staking = df_staking.drop_duplicates(subset=["case_id"])
df_catboost = df_catboost.drop_duplicates(subset=["case_id"])
df_lasso = df_lasso.drop_duplicates(subset=["case_id"])
df_lgbm = df_lgbm.drop_duplicates(subset=["case_id"])
df_network = df_network.drop_duplicates(subset=["case_id"])


merge_keys = ["case_id"]

df_staking = df_staking.rename(columns={"target": "prediction_staking"})
df_catboost = df_catboost.rename(columns={"predicted_label": "prediction_catboost"})
df_lasso = df_lasso.rename(columns={"predicted_label": "prediction_lasso"})
df_lgbm = df_lgbm.rename(columns={"predicted_label": "prediction_lgbm"})
df_network = df_network.rename(columns={"predicted_label": "prediction_network"})

df_merged = df_catboost[merge_keys + ["actual_label", "prediction_catboost"]]
df_merged["actual_label"] = df_merged["actual_label"].astype(int)
df_merged = df_merged.merge(df_staking[merge_keys + ["prediction_staking"]], on=merge_keys, how="inner")
df_merged = df_merged.merge(df_lasso[merge_keys + ["prediction_lasso"]], on=merge_keys, how="inner")
df_merged = df_merged.merge(df_lgbm[merge_keys + ["prediction_lgbm"]], on=merge_keys, how="inner")
df_merged = df_merged.merge(df_network[merge_keys + ["prediction_network"]], on=merge_keys, how="inner")

accuracy = {
    "SVM": accuracy_score(df_merged["actual_label"], df_merged["prediction_lasso"]),
    "Networks": accuracy_score(df_merged["actual_label"], df_merged["prediction_network"]),
    "LightGBM": accuracy_score(df_merged["actual_label"], df_merged["prediction_lgbm"]),
    "Catboost": accuracy_score(df_merged["actual_label"], df_merged["prediction_catboost"]),
    "Staking": accuracy_score(df_merged["actual_label"], df_merged["prediction_staking"])
}

recall = {
    "SVM": recall_score(df_merged["actual_label"], df_merged["prediction_lasso"]),
    "Networks": recall_score(df_merged["actual_label"], df_merged["prediction_network"]),
    "LightGBM": recall_score(df_merged["actual_label"], df_merged["prediction_lgbm"]),
    "Catboost": recall_score(df_merged["actual_label"], df_merged["prediction_catboost"]),
    "Staking": recall_score(df_merged["actual_label"], df_merged["prediction_staking"])
}

precision = {
    "SVM": precision_score(df_merged["actual_label"], df_merged["prediction_lasso"]),
    "Networks": precision_score(df_merged["actual_label"], df_merged["prediction_network"]),
    "LightGBM": precision_score(df_merged["actual_label"], df_merged["prediction_lgbm"]),
    "Catboost": precision_score(df_merged["actual_label"], df_merged["prediction_catboost"]),
    "Staking": precision_score(df_merged["actual_label"], df_merged["prediction_staking"])
}

auc_score = {
    "SVM": roc_auc_score(df_merged["actual_label"], df_merged["prediction_lasso"]),
    "Networks": roc_auc_score(df_merged["actual_label"], df_merged["prediction_network"]),
    "LightGBM": roc_auc_score(df_merged["actual_label"], df_merged["prediction_lgbm"]),
    "Catboost": roc_auc_score(df_merged["actual_label"], df_merged["prediction_catboost"]),
    "Staking": roc_auc_score(df_merged["actual_label"], df_merged["prediction_staking"])
}

ks = {}
for model_name, prediction in zip(["SVM", "Networks", "LightGBM", "Catboost", "Staking"],
                                 ["prediction_lasso", "prediction_network", "prediction_lgbm", "prediction_catboost", "prediction_staking"]):
    df_temp = df_merged[["actual_label", prediction]].copy()
    df_temp.sort_values(by=prediction, ascending=False, inplace=True)
    df_temp['cum_good'] = (df_temp['actual_label'] == 0).cumsum() / (df_temp['actual_label'] == 0).sum()
    df_temp['cum_bad'] = (df_temp['actual_label'] == 1).cumsum() / (df_temp['actual_label'] == 1).sum()
    ks[model_name] = max(abs(df_temp['cum_bad'] - df_temp['cum_good']))

def calculate_ks(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    ks_value = max(tpr - fpr)
    return ks_value

y_true = df_merged["actual_label"]
proba_catboost = df_catboost["score"]
proba_lasso = df_lasso["score"] 
proba_lgbm = df_lgbm["score"]
proba_network = df_network["score"]

ks_catboost = calculate_ks(y_true, proba_catboost)
ks_lasso = calculate_ks(y_true, proba_lasso)
ks_lgbm = calculate_ks(y_true, proba_lgbm)
ks_network = calculate_ks(y_true, proba_network)

print(f"KS (catboost): {ks_catboost:.4f}")
print(f"KS (SVM): {ks_lasso:.4f}")
print(f"KS (LightGBM): {ks_lgbm:.4f}")
print(f"KS (Networks): {ks_network:.4f}")

for model, acc in accuracy.items():
    print(f"{model} Accuracy: {acc:.4f}")

for model, prec in precision.items():
    print(f"{model} Precision: {prec:.4f}")

for model, rec in recall.items():
    print(f"{model} Recall: {rec:.4f}")

for model, auc in auc_score.items():
    print(f"{model} AUC: {auc:.4f}")

for model, k in ks.items():
    print(f"{model} KS: {k:.4f}")


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    filename = f"output/{model_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(model_name+" finished")

plot_confusion_matrix(df_merged["actual_label"], df_merged["prediction_lasso"], "SVM")
plot_confusion_matrix(df_merged["actual_label"], df_merged["prediction_network"], "Networks")
plot_confusion_matrix(df_merged["actual_label"], df_merged["prediction_lgbm"], "LightGBM")
plot_confusion_matrix(df_merged["actual_label"], df_merged["prediction_catboost"], "Catboost")
plot_confusion_matrix(df_merged["actual_label"], df_merged["prediction_staking"], "Staking")