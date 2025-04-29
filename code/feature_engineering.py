import polars as pl
from pathlib import Path
from glob import glob
import gc
import os
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import numpy as np
import pandas as pd

def clean_base_data(df_base: pl.DataFrame) -> pl.DataFrame:

    missing_info = df_base.null_count().to_dict(as_series=False)
    drop_cols = [col for col, miss in missing_info.items() if miss[0] / df_base.height > 0.8]
    df_base = df_base.drop(drop_cols)

    df_base = df_base.with_columns([
        pl.when(pl.col(col) == "a55475b1").then(None).otherwise(pl.col(col)).alias(col)
        for col in df_base.columns if df_base[col].dtype == pl.Utf8
    ])

    df_base = df_base.fill_null(0)

    numeric_cols = df_base.select(pl.selectors.numeric()).columns
    for col in numeric_cols:
        q01 = df_base[col].quantile(0.01)
        q99 = df_base[col].quantile(0.99)
        df_base = df_base.with_columns(
            pl.when(pl.col(col) < q01).then(q01)
            .when(pl.col(col) > q99).then(q99)
            .otherwise(pl.col(col))
            .alias(col)
        )

    df_base = df_base.with_columns([
    pl.col(col).shrink_dtype() for col in df_base.select(pl.selectors.numeric()).columns
    ])

    print(f"df_base: {df_base.shape}")

    return df_base

def feature_engineering(df: pl.DataFrame) -> pl.DataFrame:

    reference_date_polars = pl.lit(datetime(2024, 12, 31)).cast(pl.Datetime)
    date_cols = [col for col in df.columns if df[col].dtype in [pl.Date, pl.Datetime]]
    for col in date_cols:
        df = df.with_columns(
            (reference_date_polars - pl.col(col)).dt.total_days().alias(f"{col}_days_diff")
        )

    formula_features = []
    if set(["loan_amount", "income_total"]).issubset(df.columns):
        df = df.with_columns(
            (pl.col("loan_amount") / (pl.col("income_total") + 1)).alias("loan_income_ratio")
        )
        formula_features.append("loan_income_ratio")
    if set(["total_payment", "total_debt"]).issubset(df.columns):
        df = df.with_columns(
            (pl.col("total_payment") / (pl.col("total_debt") + 1)).alias("payment_debt_ratio")
        )
        formula_features.append("payment_debt_ratio")

    if set(["application_status", "contract_status"]).issubset(df.columns):
        df = df.with_columns([
            (pl.col("application_status") == "approved").cast(pl.Int8).alias("is_approved"),
            (pl.col("contract_status") == "active").cast(pl.Int8).alias("is_active")
        ])

    if "event_date" in df.columns:
        df = df.with_columns([
            ((reference_date - pl.col("event_date")).dt.days() <= 180).cast(pl.Int8).alias("event_last_6m"),
            ((reference_date - pl.col("event_date")).dt.days() <= 365).cast(pl.Int8).alias("event_last_1y"),
        ])

    if set(["loan_amount", "total_assets"]).issubset(df.columns):
        df = df.with_columns(
            (pl.col("loan_amount") / (pl.col("total_assets") + 1)).alias("loan_asset_ratio")
        )
    if set(["credit_card_limit", "income_total"]).issubset(df.columns):
        df = df.with_columns(
            (pl.col("credit_card_limit") / (pl.col("income_total") + 1)).alias("credit_limit_income_ratio")
        )

    print(f"df shape: {df.shape}")
    return df

def construct_features(df: pl.DataFrame) -> pl.DataFrame:
    def safe_div(numerator: str, denominator: str, name: str):
        if numerator in df.columns and denominator in df.columns:
            return (pl.col(numerator) / (pl.col(denominator) + 1)).alias(name)
        return None

    def safe_diff(a: str, b: str, name: str):
        if a in df.columns and b in df.columns:
            return (pl.col(a) - pl.col(b)).alias(name)
        return None

    new_columns = [
        safe_div("annuitynextmonth_57A", "annuity_780A", "annuity_ratio_next_vs_current"),
        safe_div("avgpmtlast12m_4525200A", "avginstallast24m_3658937A", "payment_to_install_ratio"),
        safe_div("currdebt_22A", "maininc_215A", "debt_to_income_ratio"),
        safe_div("avgpmtlast12m_4525200A", "currdebt_22A", "payment_to_debt_ratio"),
        safe_div("numberofqueries_373L", "clientscnt3m_3712950L", "queries_per_client_3m"),
        safe_diff("maxdpdlast12m_727P", "maxdpdtolerance_374P", "dpd_tolerance_gap"),
        safe_div("currdebt_22A", "credacc_credlmt_575A_mean", "active_credit_util_ratio"),
        safe_div("residualamount_856A_mean", "totaloutstanddebtvalue_668A_mean", "residual_debt_ratio"),
        safe_div("maininc_215A", "monthlyinstlamount_674A_mean", "monthly_payment_capacity"),
        safe_div("(currdebt_22A + totaldebtoverduevalue_178A_mean)", "maininc_215A", "debt_pressure_index"),

        safe_div("numinstpaidearly_338L", "numinstls_657L", "installment_completion_rate"),
        safe_div("numinstpaidlate1d_3546852L", "numinstls_657L", "late_installment_ratio"),
        safe_div("clientscnt12m_3712952L", "applicationscnt_1086L", "active_clients_ratio_12m"),
        safe_div("actualdpd_943P_max", "credacc_credlmt_575A_mean", "dpd_relative_to_limit"),
        safe_div("annuity_780A", "credamount_770A", "annuity_to_credit_ratio"),
        safe_div("residualamount_856A_mean", "totaldebtoverduevalue_178A_mean", "residual_to_totaldebt_ratio"),
        safe_div("credacc_transactions_402L_sum", "credacc_credlmt_575A_mean", "credit_utilization_fluctuation"),
        safe_div("monthlyinstlamount_674A_mean", "maininc_215A", "monthly_burden_index"),
        safe_div("outstandingamount_362A_mean", "avginstallast24m_3658937A", "outstanding_to_installment_ratio"),
        safe_div("avgpmtlast12m_4525200A", "totaldebtoverduevalue_178A_mean", "pmt_over_debt_ratio"),
    ]

    new_columns = [expr for expr in new_columns if expr is not None]

    return df.with_columns(new_columns)

def construct_rfm_features(df: pl.DataFrame) -> pl.DataFrame:
    reference_date = pl.lit(datetime(2024, 12, 31)).cast(pl.Date)

    def safe_date_diff(colname: str, newname: str):
        if colname in df.columns:
            return (reference_date - pl.col(colname).str.strptime(pl.Date, strict=False)).dt.total_days().alias(newname)
        else:
            return None

    def safe_div(numerator: str, denominator: str, name: str):
        if numerator in df.columns and denominator in df.columns:
            return (pl.col(numerator) / (pl.col(denominator) + 1)).alias(name)
        else:
            return None

    r_features = [
        safe_date_diff("lastrejectdate_50D", "days_since_last_reject"),
        safe_date_diff("lastapprdate_640D", "days_since_last_approval"),
        safe_date_diff("dtlastpmtallstes_4499206D", "days_since_last_payment"),
        safe_date_diff("datefirstoffer_1144D", "days_since_first_offer"),
        safe_date_diff("datelastunpaid_3546854D", "days_since_last_unpaid"),
        safe_date_diff("dateofcredstart_739D", "days_since_first_credit"),
        safe_date_diff("birthdate_574D", "days_since_birthdate"),
    ]

    f_features = [
        safe_div("applications30d_658L", "30", "application_freq_30d"),
        safe_div("numrejects9m_859L", "270", "reject_freq_9m"),
        safe_div("cntpmts24_3658933L", "24", "instalment_freq"),
        safe_div("numincomingpmts_3546848L", "12", "payment_incoming_freq"),
        safe_div("numinstpaidearly5d_1087L", "numinstls_657L", "early_pmt_freq"),
        safe_div("numinstunpaidmax_3546851L", "numinstls_657L", "unpaid_install_freq"),
        safe_div("numactivecreds_622L", "opencred_647L", "active_credit_ratio"),
    ]

    m_features = [
        safe_div("credamount_770A", "maininc_215A", "loan_to_income_ratio"),
        safe_div("outstandingamount_362A_mean", "credamount_770A", "outstanding_to_loan_ratio"),
        safe_div("avgpmtlast12m_4525200A", "outstandingamount_362A_mean", "pmt_to_outstanding_ratio"),
        safe_div("numinstpaidearly5d_1087L", "amtinstpaidbefduel24m_4187115A", "early_payment_amount_ratio"),
        safe_div("currdebt_22A", "amtdepositbalance_4809441A", "debt_vs_deposit_ratio"),
        safe_div("maininc_215A", "annuity_780A", "monthly_income_to_annuity"),
    ]

    all_features = r_features + f_features + m_features

    all_features = [expr for expr in all_features if expr is not None]

    return df.with_columns(all_features)

def prepare_features_for_lgbm(df: pl.DataFrame, target_col: str = "target") -> list:
    df_pd = df.to_pandas()
    allowed_types = ['int8', 'int16', 'int32', 'int64', 
                     'uint8', 'uint16', 'uint32', 'uint64',
                     'float16', 'float32', 'float64', 'bool']
    
    feature_cols = []
    for col in df_pd.columns:
        if col in [target_col, "case_id", "date_decision", "MONTH", "WEEK_NUM"]:
            continue
        if str(df_pd[col].dtype) in allowed_types:
            feature_cols.append(col)
    return feature_cols
    
def select_features_null(df: pl.DataFrame, target_col: str, num_rounds: int = 3) -> list:
    features = [col for col in df.columns if col not in [target_col, "case_id", "date_decision", "MONTH", "WEEK_NUM"]]
    X = df.select(features).to_pandas()
    y = df[target_col].to_numpy()
    
    real_importance = np.zeros(len(features))
    null_importance = np.zeros((len(features), num_rounds))
    
    for round_idx in range(num_rounds):
        if round_idx == 0:
            lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, n_jobs=-1,verbose=-1)
            lgbm.fit(X, y)
            real_importance = lgbm.feature_importances_
        y_shuffled = np.random.permutation(y)
        lgbm_null = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, n_jobs=-1)
        lgbm_null.fit(X, y_shuffled)
        null_importance[:, round_idx] = lgbm_null.feature_importances_
    
    null_importance_mean = np.mean(null_importance, axis=1)
    imp_score = real_importance / (null_importance_mean + 1e-5)
    
    selected = [features[i] for i, score in enumerate(imp_score) if score > 1.5]
    print(f"features remaining after null importance: {len(selected)}/{len(features)}")
    return selected


def select_features_adv(df: pl.DataFrame, selected_features: list) -> tuple:
    df = df.with_columns([
        (pl.col("WEEK_NUM") >= 65).cast(pl.Int8).alias("is_covid")
    ])
    X = df.select(selected_features).to_pandas()
    y = df["is_covid"].to_numpy()

    auc_scores = []

    for feature in tqdm(selected_features, desc="Adversarial Validation Progress"):
        lgbm = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            n_jobs=-1,
            verbose=-1 
        )
        lgbm.fit(X[[feature]], y)
        preds = lgbm.predict_proba(X[[feature]])[:, 1]
        auc = roc_auc_score(y, preds)
        auc_scores.append((feature, auc))
    
    auc_df = pd.DataFrame(auc_scores, columns=["feature", "auc"])
    selected = auc_df.loc[auc_df["auc"] < 0.65, "feature"].tolist()
    print(f"features remaining after adv validation : {len(selected)}/{len(selected_features)}")

    top20_features = auc_df.sort_values("auc", ascending=False).head(20)["feature"].tolist()
    print("\n Adversarial Validation Top20 features：")
    for i, feat in enumerate(top20_features, 1):
        print(f"{i}. {feat}")

    return selected, auc_df, top20_features


def select_features_lgbm(df: pl.DataFrame, selected_features: list, target_col: str, topk: int = 300) -> tuple:
    X = df.select(selected_features).to_pandas()
    y = df[target_col].to_numpy()

    lgbm = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        n_jobs=-1,
        verbose=-1 
    )
    lgbm.fit(X, y)
    feature_importance = lgbm.feature_importances_

    imp_df = pd.DataFrame({
        "feature": selected_features,
        "importance": feature_importance
    }).sort_values("importance", ascending=False)

    final_features = imp_df.head(topk)["feature"].tolist()

    print(f"features remaining after lightGBM : {len(final_features)}")

    top20_features = imp_df.head(20)["feature"].tolist()
    print("\n LightGBM Top20 features：")
    for i, feat in enumerate(top20_features, 1):
        print(f"{i}. {feat}")

    return final_features, imp_df, top20_features

def split_train_test_with_target_balance(df: pl.DataFrame, target_col: str = "target", 
                                         total_test_size: int = 10000, min_target1_count: int = 100, seed: int = 42):
    np.random.seed(seed)

    df_pos = df.filter(pl.col(target_col) == 1)
    df_neg = df.filter(pl.col(target_col) == 0)
    df_pos_sample = df_pos.sample(n=min_target1_count, seed=seed)
    df_neg_sample = df_neg.sample(n=total_test_size - min_target1_count, seed=seed)

    df_test = pl.concat([df_pos_sample, df_neg_sample], how="vertical")
    df_test = df_test.sample(n=df_test.height, seed=seed) 

    df_train = df.filter(~pl.col("case_id").is_in(df_test["case_id"]))


    df_test.write_csv("data/test_data.csv")
    df_train.write_csv("data/train_data.csv")
    print("save test_data.csv and train_data.csv")

df_base = pl.read_parquet('data/base_data.parquet')
df_base_cleaned = clean_base_data(df_base)

df_final = feature_engineering(df_base_cleaned)
df_final = construct_features(df_final)
df_final = construct_rfm_features(df_final)
df_final_clean = clean_base_data(df_final)

selected_candidates = prepare_features_for_lgbm(df_final_clean, target_col="target")

selected_null = select_features_null(
    df_final_clean.select(selected_candidates + ["target"]), 
    target_col="target"
)

selected_adv, adv_auc_df, top20_adv_features = select_features_adv(
    df_final_clean.select(selected_candidates + ["target", "WEEK_NUM"]), 
    selected_null
)

selected_final, lgbm_imp_df, top20_lgbm_features = select_features_lgbm(
    df_final_clean.select(selected_candidates + ["target"]),
    selected_adv,
    target_col="target",
    topk=60
)

selected_top50 = selected_final[:60]
important_cols = ["case_id", "target", "date_decision"]
final_cols = important_cols + selected_top50
final_data = df_final_clean.select(final_cols)

split_train_test_with_target_balance(final_data, target_col="target", total_test_size=10000, min_target1_count=100, seed=42)