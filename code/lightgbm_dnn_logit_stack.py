# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import time
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score,
    balanced_accuracy_score
)

# Try importing sklearn models/classes
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

# Try importing RAPIDS libraries for GPU acceleration
try:
    import cudf
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier
    from cuml.linear_model import LogisticRegression
    from cuml.preprocessing import StandardScaler
    from cuml import train_test_split as cu_train_test_split
    RAPIDS_AVAILABLE = True
    print("RAPIDS libraries successfully imported for GPU acceleration.")
except ImportError:
    RAPIDS_AVAILABLE = False
    print("RAPIDS libraries not available. Using scikit-learn instead.")

# Try importing LightGBM with GPU support
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("LightGBM successfully imported.")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available.")

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42

# Utility Functions
def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Find optimal decision threshold to maximize the given metric
    """
    # Try multiple thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred)  # Default to F1

        scores.append(score)

    # Find highest score and corresponding threshold
    best_score_idx = np.argmax(scores)
    optimal_threshold = thresholds[best_score_idx]
    best_score = scores[best_score_idx]

    return optimal_threshold, best_score

def is_cupy_array(arr):
    """Check if the array is a CuPy array"""
    return hasattr(arr, '__module__') and arr.__module__.startswith('cupy')

def is_cudf_dataframe(df):
    """Check if the dataframe is a cuDF DataFrame or Series"""
    return hasattr(df, '__module__') and df.__module__.startswith('cudf')

def to_numpy(data):
    """Safely convert any array-like object to NumPy array"""
    if data is None:
        return None

    # If it's already a NumPy array
    if isinstance(data, np.ndarray):
        return data

    # CuPy array
    if is_cupy_array(data):
        return data.get()

    # cuDF DataFrame/Series
    if is_cudf_dataframe(data):
        return data.to_pandas().values if hasattr(data, 'to_pandas') else np.array(data)

    # Pandas DataFrame/Series
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.values

    # Other array-like
    return np.asarray(data)

def preprocess_for_gpu(data):
    """Preprocess data to make it compatible with GPU operations"""
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        for col in result.columns:
            if result[col].dtype == 'object':
                print(f"Converting object column {col} to category codes")
                result[col] = result[col].astype('category').cat.codes.astype('float64')
            elif not np.issubdtype(result[col].dtype, np.number):
                print(f"Converting non-numeric column {col} to float")
                try:
                    result[col] = result[col].astype('float64')
                except:
                    result[col] = result[col].astype('category').cat.codes.astype('float64')
        return result
    return data

def safe_to_gpu(data, using_gpu=True):
    """Safely convert data to GPU format, with better error handling"""
    if not using_gpu or data is None:
        return data

    # Already on GPU
    if is_cupy_array(data) or is_cudf_dataframe(data):
        return data

    try:
        # For pandas DataFrame, need to ensure all columns are numeric
        if isinstance(data, pd.DataFrame):
            # First convert all object columns to category codes
            for col in data.columns:
                if data[col].dtype == 'object':
                    try:
                        data[col] = data[col].astype('category').cat.codes.astype('float64')
                    except:
                        # If conversion fails, use a simpler approach - just convert to zeros
                        print(f"Warning: Could not convert column {col} to numeric. Using zeros instead.")
                        data[col] = 0.0

                # Convert any other non-numeric types (like boolean)
                if not np.issubdtype(data[col].dtype, np.number):
                    try:
                        data[col] = data[col].astype('float64')
                    except:
                        print(f"Warning: Could not convert column {col} to numeric. Using zeros instead.")
                        data[col] = 0.0

            # Convert to numpy array
            data_np = data.values
        else:
            # For other types, try direct conversion to numpy
            data_np = to_numpy(data)

            # If data_np contains objects, try to convert them to float
            if data_np.dtype == object:
                try:
                    data_np = data_np.astype('float64')
                except:
                    print("Warning: Failed to convert object array to numeric. Using zeros instead.")
                    data_np = np.zeros(data_np.shape, dtype='float64')

        # Convert to GPU array
        return cp.array(data_np)
    except Exception as e:
        print(f"GPU conversion failed with error: {e}")
        print("Falling back to CPU processing")
        return data

def identify_feature_types(df):
    """Identify feature types in the dataframe"""
    numeric_features = []
    categorical_features = []
    date_features = []

    for col in df.columns:
        # Check for date features first
        if df[col].dtype == 'object':
            # Try to parse as date
            try:
                pd.to_datetime(df[col].iloc[0])
                date_features.append(col)
                continue
            except:
                pass

            # Check number of unique values
            n_unique = df[col].nunique()
            if n_unique < 20:  # Arbitrary threshold for categorical
                categorical_features.append(col)
            else:
                # This could be a high cardinality categorical or text
                categorical_features.append(col)
        else:
            # Numeric features
            numeric_features.append(col)

    return {
        'numeric': numeric_features,
        'categorical': categorical_features,
        'date': date_features
    }

def analyze_dataset(X_train, y_train, using_gpu):
    """Analyze dataset to determine appropriate evaluation metrics"""
    print("\nAnalyzing dataset characteristics...")

    # Convert to pandas if using GPU
    y_train_pd = y_train.to_pandas() if hasattr(y_train, 'to_pandas') else y_train

    # Check class distribution
    class_counts = y_train_pd.value_counts()
    class_distribution = class_counts / len(y_train_pd)

    # Determine if dataset is imbalanced
    minority_class_ratio = class_distribution.min() / class_distribution.max()
    is_imbalanced = minority_class_ratio < 0.2  # Arbitrary threshold

    # Print class distribution
    print("\nClass Distribution:")
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count} samples ({class_distribution[cls]:.4f})")

    # Set primary metric to AUC-PR (average_precision)
    primary_metric = "average_precision"

    # Determine secondary metric based on dataset characteristics
    if is_imbalanced:
        print("\nDataset is imbalanced.")
        if class_distribution.idxmin() == 1:
            print("Positive class (1) is the minority class.")
            print("Primary metric: Precision-Recall AUC (PR-AUC)")
            secondary_metric = "f1"
        else:
            print("Negative class (0) is the minority class.")
            print("Primary metric: Precision-Recall AUC (PR-AUC)")
            secondary_metric = "f1"
    else:
        print("\nDataset is relatively balanced.")
        print("Primary metric: Precision-Recall AUC (PR-AUC)")
        secondary_metric = "accuracy"

    # Create metrics_info dictionary
    metrics_info = {
        "class_distribution": class_distribution,
        "is_imbalanced": is_imbalanced,
        "minority_class_ratio": minority_class_ratio,
        "primary_metric": primary_metric,
        "secondary_metric": secondary_metric
    }

    return metrics_info

def evaluate_model(model, X, y, model_name, metrics_info, feature_names=None, using_gpu=True):
    """Evaluate a model"""
    print(f"\nEvaluating {model_name}...")

    # Convert to CPU/GPU as needed
    X_np = to_numpy(X)
    y_np = to_numpy(y)

    # Get optimal threshold if available
    optimal_threshold = getattr(model, 'optimal_threshold', 0.5)

    # Initialize variables before try block
    y_pred = None
    y_pred_proba = None

    try:
        if using_gpu and not isinstance(model, lgb.LGBMClassifier):
            X_gpu = safe_to_gpu(X, using_gpu)

            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_gpu)

                # Get probabilities for positive class
                if y_pred_proba.shape[1] > 1:
                    y_pred_proba = y_pred_proba[:, 1]

                # Convert probabilities to class labels using optimal threshold
                y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            else:
                y_pred = model.predict(X_gpu)
                y_pred_proba = y_pred  # Use predictions as probabilities if no probabilities available
        else:
            # For LightGBM or CPU models
            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_np)

                # Get probabilities for positive class
                if y_pred_proba.shape[1] > 1:
                    y_pred_proba = y_pred_proba[:, 1]

                # Convert probabilities to class labels using optimal threshold
                y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            else:
                y_pred = model.predict(X_np)
                y_pred_proba = y_pred  # Use predictions as probabilities if no probabilities available
    except Exception as e:
        print(f"Error in model evaluation: {e}")
        print("Falling back to CPU prediction")

        # Make sure variables are initialized
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_np)

                # Get probabilities for positive class
                if y_pred_proba.shape[1] > 1:
                    y_pred_proba_np = y_pred_proba[:, 1]
                else:
                    y_pred_proba_np = y_pred_proba

                # Convert probabilities to class labels using optimal threshold
                y_pred_np = (y_pred_proba_np >= optimal_threshold).astype(int)

                # Set the main variables for later use
                y_pred = y_pred_np
                y_pred_proba = y_pred_proba_np
            except:
                # If all else fails, use random predictions
                print("Prediction failed. Using random predictions for evaluation.")
                np.random.seed(RANDOM_STATE)
                y_pred_proba_np = np.random.random(y_np.shape)
                y_pred_np = (y_pred_proba_np >= optimal_threshold).astype(int)

                # Set the main variables for later use
                y_pred = y_pred_np
                y_pred_proba = y_pred_proba_np
        else:
            try:
                y_pred_np = model.predict(X_np)
                y_pred_proba_np = y_pred_np.astype(float)

                # Set the main variables for later use
                y_pred = y_pred_np
                y_pred_proba = y_pred_proba_np
            except:
                # If all else fails, use random predictions
                print("Prediction failed. Using random predictions for evaluation.")
                np.random.seed(RANDOM_STATE)
                y_pred_np = np.random.randint(0, 2, y_np.shape)
                y_pred_proba_np = y_pred_np.astype(float)

                # Set the main variables for later use
                y_pred = y_pred_np
                y_pred_proba = y_pred_proba_np

    # Convert to numpy for metrics
    y_pred_np = to_numpy(y_pred)
    y_pred_proba_np = to_numpy(y_pred_proba)

    # Calculate metrics
    accuracy = accuracy_score(y_np, y_pred_np)
    precision = precision_score(y_np, y_pred_np)
    recall = recall_score(y_np, y_pred_np)
    f1 = f1_score(y_np, y_pred_np)
    roc_auc = roc_auc_score(y_np, y_pred_proba_np)
    pr_auc = average_precision_score(y_np, y_pred_proba_np)

    # Confusion matrix
    cm = confusion_matrix(y_np, y_pred_np)

    # Print metrics
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  PR AUC: {pr_auc:.4f}")

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(cm)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_np, y_pred_np))

    # Create output directory if it doesn't exist
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'outputs/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_np, y_pred_proba_np)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'outputs/{model_name.lower().replace(" ", "_")}_roc_curve.png')
    plt.close()

    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    precision_curve, recall_curve, _ = precision_recall_curve(y_np, y_pred_proba_np)
    plt.plot(recall_curve, precision_curve, lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(f'outputs/{model_name.lower().replace(" ", "_")}_pr_curve.png')
    plt.close()

    # Plot feature importance if available
    if model_name in ['Random Forest', 'LightGBM'] and feature_names is not None:
        if hasattr(model, 'feature_importances_'):
            try:
                # Get feature importances
                importances = to_numpy(model.feature_importances_)

                # Ensure feature_names is the right length
                if len(feature_names) >= len(importances):
                    # Create DataFrame for plotting
                    importance_df = pd.DataFrame({
                        'Feature': feature_names[:len(importances)],
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)

                    # Plot top 20 features
                    plt.figure(figsize=(10, 8))
                    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
                    plt.title(f'{model_name} - Top 20 Feature Importance')
                    plt.tight_layout()
                    plt.savefig(f'outputs/{model_name.lower().replace(" ", "_")}_feature_importance.png')
                    plt.close()
                else:
                    print(f"Feature names length ({len(feature_names)}) doesn't match importances length ({len(importances)}). Skipping feature importance plot.")
            except Exception as e:
                print(f"Error plotting feature importance: {e}")
                print("Skipping feature importance plot")

    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'optimal_threshold': optimal_threshold
    }

    return metrics

def load_data(train_path, test_path, feature_list):
    """Load training and testing data from parquet files using GPU acceleration if available"""
    # Load data
    print("Loading data...")
    using_gpu = RAPIDS_AVAILABLE

    try:
        if using_gpu:
            # Try loading with cuDF for GPU acceleration
            train_data = cudf.read_parquet(train_path)
            test_data = cudf.read_parquet(test_path)
            print("Successfully loaded data using GPU acceleration (cuDF)")
        else:
            # Fall back to pandas
            train_data = pd.read_parquet(train_path)
            test_data = pd.read_parquet(test_path)
            print("Loaded data using pandas")
    except:
        # Fall back to pandas if cuDF fails
        print("Could not load data with cuDF, falling back to pandas...")
        train_data = pd.read_parquet(train_path)
        test_data = pd.read_parquet(test_path)
        using_gpu = False

    # Print available columns for debugging
    print("Available columns in training data:")
    train_columns = train_data.columns.tolist()
    print(train_columns)

    # Filter to only use the specified features if the list is not empty
    if feature_list and len(feature_list) > 0:
        valid_features = []
        for feature in feature_list:
            if feature in train_data.columns:
                # Skip target and case_id
                if feature != 'target' and feature != 'case_id':
                    valid_features.append(feature)
            else:
                print(f"Warning: Feature '{feature}' not found in the data and will be skipped.")

        # If no valid features were found, use all available features
        if len(valid_features) == 0:
            print("No valid features found in the provided list. Using all available features.")
            valid_features = [col for col in train_data.columns if col != 'target' and col != 'case_id']
    else:
        # If feature_list is empty, use all available features
        print("No feature list provided. Using all available features.")
        valid_features = [col for col in train_data.columns if col != 'target' and col != 'case_id']

    print(f"Using {len(valid_features)} features: {valid_features[:10]}...")

    # Ensure we have at least one feature
    if len(valid_features) == 0:
        raise ValueError("No valid features found for training. Check your data and feature list.")

    # Store case_id for later use
    if 'case_id' in test_data.columns:
        case_id = test_data['case_id'].to_pandas() if hasattr(test_data, 'to_pandas') else test_data['case_id'].copy()
    else:
        case_id = None

    # Separate features and target
    if 'target' in train_data.columns:
        y_train = train_data['target']
        X_train = train_data[valid_features]
    else:
        raise ValueError("Target variable 'target' not found in training data")

    # If case_id is in the test data, ensure it's not used as a feature
    X_test = test_data[valid_features].copy()

    # If target is in test data, use it
    if 'target' in test_data.columns:
        y_test = test_data['target']
        print("Test data contains target variable. Will evaluate on test set.")
    else:
        y_test = None
        print("Test data does not contain target variable. Will not evaluate on test set.")

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Convert to pandas temporarily for feature type identification
    X_train_pd = X_train.to_pandas() if hasattr(X_train, 'to_pandas') else X_train

    # Identify feature types
    feature_types = identify_feature_types(X_train_pd)

    print("Feature types summary:")
    for ftype, features in feature_types.items():
        print(f"  {ftype}: {len(features)} features")

    return X_train, X_test, y_train, y_test, case_id, feature_types, using_gpu

def preprocess_data(X_train, X_test, feature_types, using_gpu):
    """Preprocess data for model training with better error handling"""
    print("\nPreprocessing data...")

    # Convert from GPU to CPU if needed for preprocessing
    X_train_pd = X_train.to_pandas() if hasattr(X_train, 'to_pandas') else X_train
    X_test_pd = X_test.to_pandas() if hasattr(X_test, 'to_pandas') else X_test

    # Create lists to store processed features and their names
    train_processed_features = []
    test_processed_features = []
    feature_names = []

    # Process numeric features - standardize
    if feature_types['numeric']:
        print(f"Processing {len(feature_types['numeric'])} numeric features...")

        try:
            if using_gpu:
                # Convert to numeric values first
                numeric_train = X_train_pd[feature_types['numeric']].copy()
                numeric_test = X_test_pd[feature_types['numeric']].copy()

                # Handle non-numeric values
                for col in numeric_train.columns:
                    if not np.issubdtype(numeric_train[col].dtype, np.number):
                        try:
                            numeric_train[col] = pd.to_numeric(numeric_train[col], errors='coerce').fillna(0)
                            numeric_test[col] = pd.to_numeric(numeric_test[col], errors='coerce').fillna(0)
                        except:
                            print(f"Warning: Could not convert column {col} to numeric. Using zeros instead.")
                            numeric_train[col] = 0.0
                            numeric_test[col] = 0.0

                # Convert to cuDF DataFrame for GPU processing
                try:
                    numeric_train_gpu = cudf.DataFrame(numeric_train)
                    numeric_test_gpu = cudf.DataFrame(numeric_test)

                    # Use cuML's StandardScaler
                    scaler = StandardScaler()
                    numeric_train_scaled = scaler.fit_transform(numeric_train_gpu)
                    numeric_test_scaled = scaler.transform(numeric_test_gpu)

                    # Convert back to numpy for later concatenation
                    train_processed_features.append(to_numpy(numeric_train_scaled))
                    test_processed_features.append(to_numpy(numeric_test_scaled))
                except Exception as e:
                    print(f"GPU scaling failed with error: {e}")
                    print("Falling back to CPU scaling...")

                    # Fall back to sklearn
                    from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
                    scaler = SklearnStandardScaler()
                    numeric_train_scaled = scaler.fit_transform(numeric_train)
                    numeric_test_scaled = scaler.transform(numeric_test)

                    train_processed_features.append(numeric_train_scaled)
                    test_processed_features.append(numeric_test_scaled)
            else:
                # Use sklearn's StandardScaler
                from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

                numeric_train = X_train_pd[feature_types['numeric']]
                numeric_test = X_test_pd[feature_types['numeric']]

                # Handle non-numeric values
                for col in numeric_train.columns:
                    if not np.issubdtype(numeric_train[col].dtype, np.number):
                        try:
                            numeric_train[col] = pd.to_numeric(numeric_train[col], errors='coerce').fillna(0)
                            numeric_test[col] = pd.to_numeric(numeric_test[col], errors='coerce').fillna(0)
                        except:
                            print(f"Warning: Could not convert column {col} to numeric. Using zeros instead.")
                            numeric_train[col] = 0.0
                            numeric_test[col] = 0.0

                scaler = SklearnStandardScaler()
                numeric_train_scaled = scaler.fit_transform(numeric_train)
                numeric_test_scaled = scaler.transform(numeric_test)

                train_processed_features.append(numeric_train_scaled)
                test_processed_features.append(numeric_test_scaled)
        except Exception as e:
            print(f"Error processing numeric features: {e}")
            print("Skipping numeric feature scaling...")

        # Add feature names
        feature_names.extend(feature_types['numeric'])

    # Process categorical features - one-hot encoding
    if feature_types['categorical']:
        print(f"Processing {len(feature_types['categorical'])} categorical features...")

        try:
            # We'll use pandas for one-hot encoding (cuML doesn't have a direct equivalent)
            from sklearn.preprocessing import OneHotEncoder

            categorical_train = X_train_pd[feature_types['categorical']]
            categorical_test = X_test_pd[feature_types['categorical']]

            # Convert object columns to category
            for col in categorical_train.columns:
                if categorical_train[col].dtype == 'object':
                    categorical_train[col] = categorical_train[col].astype('category')
                    categorical_test[col] = categorical_test[col].astype('category')

            # Handle categories not seen during training
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            categorical_train_encoded = encoder.fit_transform(categorical_train)
            categorical_test_encoded = encoder.transform(categorical_test)

            train_processed_features.append(categorical_train_encoded)
            test_processed_features.append(categorical_test_encoded)

            # Add feature names
            for i, feature in enumerate(feature_types['categorical']):
                for j, category in enumerate(encoder.categories_[i]):
                    feature_names.append(f"{feature}_{category}")
        except Exception as e:
            print(f"Error processing categorical features: {e}")
            print("Using simple label encoding for categorical features...")

            # Fall back to simpler approach - just convert to category codes
            for col in feature_types['categorical']:
                try:
                    # Create encoded columns
                    train_encoded = X_train_pd[col].astype('category').cat.codes.values.reshape(-1, 1)
                    test_encoded = X_test_pd[col].astype('category').cat.codes.values.reshape(-1, 1)

                    # Add to processed features
                    train_processed_features.append(train_encoded)
                    test_processed_features.append(test_encoded)

                    # Add feature name
                    feature_names.append(col)
                except Exception as e:
                    print(f"Could not encode {col}: {e}")

    # Process date features - extract year, month, day, etc.
    if feature_types['date']:
        print(f"Processing {len(feature_types['date'])} date features...")

        try:
            date_train = X_train_pd[feature_types['date']].copy()
            date_test = X_test_pd[feature_types['date']].copy()

            # Convert to datetime
            for col in feature_types['date']:
                try:
                    date_train[col] = pd.to_datetime(date_train[col], errors='coerce')
                    date_test[col] = pd.to_datetime(date_test[col], errors='coerce')
                except Exception as e:
                    print(f"Error converting {col} to datetime: {e}")
                    print(f"Using dummy values for {col}")
                    # Use dummy datetime
                    date_train[col] = pd.to_datetime('2020-01-01')
                    date_test[col] = pd.to_datetime('2020-01-01')

            # Create DataFrames to store extracted features
            date_train_features = pd.DataFrame()
            date_test_features = pd.DataFrame()

            # Extract features for each date column
            for col in feature_types['date']:
                # Year
                date_train_features[f"{col}_year"] = date_train[col].dt.year
                date_test_features[f"{col}_year"] = date_test[col].dt.year
                feature_names.append(f"{col}_year")

                # Month
                date_train_features[f"{col}_month"] = date_train[col].dt.month
                date_test_features[f"{col}_month"] = date_test[col].dt.month
                feature_names.append(f"{col}_month")

                # Day
                date_train_features[f"{col}_day"] = date_train[col].dt.day
                date_test_features[f"{col}_day"] = date_test[col].dt.day
                feature_names.append(f"{col}_day")

                # Day of week
                date_train_features[f"{col}_dayofweek"] = date_train[col].dt.dayofweek
                date_test_features[f"{col}_dayofweek"] = date_test[col].dt.dayofweek
                feature_names.append(f"{col}_dayofweek")

                # Quarter
                date_train_features[f"{col}_quarter"] = date_train[col].dt.quarter
                date_test_features[f"{col}_quarter"] = date_test[col].dt.quarter
                feature_names.append(f"{col}_quarter")

                # Is weekend
                date_train_features[f"{col}_is_weekend"] = (date_train[col].dt.dayofweek >= 5).astype(int)
                date_test_features[f"{col}_is_weekend"] = (date_test[col].dt.dayofweek >= 5).astype(int)
                feature_names.append(f"{col}_is_weekend")

                # Additional date features
                date_train_features[f"{col}_weekofyear"] = date_train[col].dt.isocalendar().week
                date_test_features[f"{col}_weekofyear"] = date_test[col].dt.isocalendar().week
                feature_names.append(f"{col}_weekofyear")

                date_train_features[f"{col}_is_month_start"] = date_train[col].dt.is_month_start.astype(int)
                date_test_features[f"{col}_is_month_start"] = date_test[col].dt.is_month_start.astype(int)
                feature_names.append(f"{col}_is_month_start")

                date_train_features[f"{col}_is_month_end"] = date_train[col].dt.is_month_end.astype(int)
                date_test_features[f"{col}_is_month_end"] = date_test[col].dt.is_month_end.astype(int)
                feature_names.append(f"{col}_is_month_end")

                date_train_features[f"{col}_is_quarter_start"] = date_train[col].dt.is_quarter_start.astype(int)
                date_test_features[f"{col}_is_quarter_start"] = date_test[col].dt.is_quarter_start.astype(int)
                feature_names.append(f"{col}_is_quarter_start")

                date_train_features[f"{col}_is_quarter_end"] = date_train[col].dt.is_quarter_end.astype(int)
                date_test_features[f"{col}_is_quarter_end"] = date_test[col].dt.is_quarter_end.astype(int)
                feature_names.append(f"{col}_is_quarter_end")

                date_train_features[f"{col}_days_in_month"] = date_train[col].dt.days_in_month
                date_test_features[f"{col}_days_in_month"] = date_test[col].dt.days_in_month
                feature_names.append(f"{col}_days_in_month")

            # Convert to numpy arrays
            train_processed_features.append(date_train_features.values)
            test_processed_features.append(date_test_features.values)
        except Exception as e:
            print(f"Error processing date features: {e}")
            print("Skipping date feature extraction...")

    # Combine all processed features
    try:
        X_train_processed = np.hstack(train_processed_features)
        X_test_processed = np.hstack(test_processed_features)
    except Exception as e:
        print(f"Error combining features: {e}")
        print("Falling back to simpler preprocessing...")

        # Simplify preprocessing - just convert everything to numeric
        X_train_simple = preprocess_for_gpu(X_train_pd).values
        X_test_simple = preprocess_for_gpu(X_test_pd).values

        X_train_processed = X_train_simple
        X_test_processed = X_test_simple
        feature_names = X_train_pd.columns.tolist()

    # Convert to cupy arrays if using GPU
    if using_gpu:
        try:
            X_train_processed = safe_to_gpu(X_train_processed, using_gpu)
            X_test_processed = safe_to_gpu(X_test_processed, using_gpu)
        except Exception as e:
            print(f"Error converting to GPU: {e}")
            print("Continuing with CPU arrays...")

    print(f"Processed data shapes - Training: {X_train_processed.shape}, Testing: {X_test_processed.shape}")

    return X_train_processed, X_test_processed, feature_names

# Initial data loading and preparation
def prepare_data_and_split():
    # Read feature list from Excel file if available
    try:
        feature_df = pd.read_excel('feature_explanations.xlsx', sheet_name='all_features')
        if 'Feature_Name' in feature_df.columns:
            feature_list = feature_df['Feature_Name'].tolist()
            # Make sure we don't include target or case_id in features
            feature_list = [f for f in feature_list if f != 'target' and f != 'case_id']
            print(f"Loaded {len(feature_list)} features from all_features tab")
            if len(feature_list) == 0:
                print("Warning: No valid features found in Excel file. Will use all available features.")
        else:
            print("Warning: Column 'Feature_Name' not found in Excel file. Will use all available features.")
            feature_list = []
    except Exception as e:
        print(f"Error loading feature list: {e}")
        print("Using all available features")
        # Default to empty list - all features will be used
        feature_list = []

    # Set paths to training and testing data
    train_path = 'train_data.parquet'
    test_path = 'test_data.parquet'

    # Load data with specified features
    X_train, X_test, y_train, y_test, case_id, feature_types, using_gpu = load_data(train_path, test_path, feature_list)

    # Analyze dataset to determine appropriate metrics
    metrics_info = analyze_dataset(X_train, y_train, using_gpu)

    # Preprocess data
    X_train_processed, X_test_processed, feature_names = preprocess_data(X_train, X_test, feature_types, using_gpu)

    # Split data into training and validation sets (ensuring proportional distribution of classes)
    X_train_np = to_numpy(X_train_processed)
    y_train_np = to_numpy(y_train)

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_np, y_train_np, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_np
    )

    # Convert back to GPU if needed
    if using_gpu:
        X_train_split = safe_to_gpu(X_train_split, using_gpu)
        X_val = safe_to_gpu(X_val, using_gpu)
        y_train_split = safe_to_gpu(y_train_split, using_gpu)
        y_val = safe_to_gpu(y_val, using_gpu)

    # Check for positive cases in train/val splits to ensure model training works
    y_train_split_np = to_numpy(y_train_split)
    y_val_np = to_numpy(y_val)

    # Check if we have positive cases in both train and validation
    train_positive = np.sum(y_train_split_np == 1)
    val_positive = np.sum(y_val_np == 1)

    print(f"Training split: {train_positive}/{len(y_train_split_np)} positive cases ({train_positive/len(y_train_split_np)*100:.2f}%)")
    print(f"Validation split: {val_positive}/{len(y_val_np)} positive cases ({val_positive/len(y_val_np)*100:.2f}%)")

    # If no positive cases in validation, regenerate splits with stratification
    if val_positive == 0:
        print("WARNING: No positive cases in validation set. Re-splitting data with stratification...")
        # Try a different random state to ensure proper stratification
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train_np, y_train_np, test_size=0.2, random_state=RANDOM_STATE+1, stratify=y_train_np
        )

        # Convert back to GPU if needed
        if using_gpu:
            X_train_split = safe_to_gpu(X_train_split, using_gpu)
            X_val = safe_to_gpu(X_val, using_gpu)
            y_train_split = safe_to_gpu(y_train_split, using_gpu)
            y_val = safe_to_gpu(y_val, using_gpu)

        # Check again
        y_train_split_np = to_numpy(y_train_split)
        y_val_np = to_numpy(y_val)
        train_positive = np.sum(y_train_split_np == 1)
        val_positive = np.sum(y_val_np == 1)

        print(f"After re-split - Training: {train_positive}/{len(y_train_split_np)} positive cases ({train_positive/len(y_train_split_np)*100:.2f}%)")
        print(f"After re-split - Validation: {val_positive}/{len(y_val_np)} positive cases ({val_positive/len(y_val_np)*100:.2f}%)")

    return X_train, X_test, y_train, y_test, X_train_processed, X_test_processed, X_train_split, X_val, y_train_split, y_val, case_id, feature_names, metrics_info, using_gpu

# Load and preprocess the data
X_train, X_test, y_train, y_test, X_train_processed, X_test_processed, X_train_split, X_val, y_train_split, y_val, case_id, feature_names, metrics_info, using_gpu = prepare_data_and_split()

print("\n--- Training and Optimizing Models on Split Dataset ---")

def optimize_lightgbm_random_search(X_train, y_train, metrics_info, cv=5, n_iter=15):
    """Optimize LightGBM hyperparameters with Random Search"""
    print("\nOptimizing LightGBM hyperparameters with Random Search...")

    if not LIGHTGBM_AVAILABLE:
        print("LightGBM not available. Skipping optimization.")
        return None

    # Import LightGBM
    from lightgbm import LGBMClassifier

    # Determine scoring metric for optimization
    scoring_metric = metrics_info['primary_metric']
    threshold_metric = metrics_info['secondary_metric']
    print(f"Using {scoring_metric} as the primary scoring metric for optimization")
    print(f"Using {threshold_metric} for threshold optimization")

    # Convert to NumPy for sklearn/LightGBM
    X_train_np = to_numpy(X_train)
    y_train_np = to_numpy(y_train)

    # Determine if the dataset is imbalanced
    is_imbalanced = metrics_info['is_imbalanced']
    print(f"Dataset is imbalanced: {is_imbalanced}")

    # Check if GPU is available for LightGBM
    try:
        test_model = LGBMClassifier(device='gpu')
        use_gpu_for_lgb = True
        print("GPU is available for LightGBM.")
    except:
        use_gpu_for_lgb = False
        print("GPU is not available for LightGBM. Using CPU instead.")

    # Define expanded parameter space for random search
    param_space = {
        'num_leaves': [16, 32, 64, 96, 128, 200],
        'max_depth': [3, 4, 5, 6, 7, 8, 10, 12],
        'learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1, 0.15],
        'n_estimators': [100, 200, 300, 400, 500, 700],
        'min_child_samples': [3, 5, 7, 10, 15, 20, 25],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0, 2.0],
        'reg_lambda': [0, 0.01, 0.1, 0.5, 1.0, 2.0],
        'min_split_gain': [0, 0.01, 0.1, 0.5, 1.0],
        'min_data_in_leaf': [5, 10, 15, 20, 25, 30],
        'max_bin': [128, 255, 512]
    }

    # Handle imbalanced data
    if is_imbalanced:
        param_space['is_unbalance'] = [True]
    else:
        param_space['is_unbalance'] = [False]

    # Add GPU support if available
    if use_gpu_for_lgb:
        param_space['device'] = ['gpu']

    print(f"Testing {n_iter} random combinations...")

    # Custom scoring function that finds optimal threshold
    def custom_scorer(model, X, y):
        """Custom scorer that finds optimal threshold and calculates score"""
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)[:, 1]

            # Calculate score based on whether the metric needs threshold
            if scoring_metric in ['roc_auc', 'average_precision']:
                # These metrics don't need threshold
                if scoring_metric == 'roc_auc':
                    score = roc_auc_score(y, y_pred_proba)
                else:  # average_precision
                    score = average_precision_score(y, y_pred_proba)
            else:
                # These metrics need threshold
                threshold, _ = find_optimal_threshold(y, y_pred_proba, scoring_metric)
                y_pred_labels = (y_pred_proba >= threshold).astype(int)

                if scoring_metric == 'f1':
                    score = f1_score(y, y_pred_labels)
                elif scoring_metric == 'accuracy':
                    score = accuracy_score(y, y_pred_labels)
                elif scoring_metric == 'precision':
                    score = precision_score(y, y_pred_labels)
                elif scoring_metric == 'recall':
                    score = recall_score(y, y_pred_labels)
                else:
                    score = f1_score(y, y_pred_labels)  # Default to F1

            return score
        else:
            # Fallback to default scoring if predict_proba not available
            if scoring_metric == 'f1':
                return f1_score(y, model.predict(X))
            elif scoring_metric == 'accuracy':
                return accuracy_score(y, model.predict(X))
            elif scoring_metric == 'precision':
                return precision_score(y, model.predict(X))
            elif scoring_metric == 'recall':
                return recall_score(y, model.predict(X))
            else:
                return accuracy_score(y, model.predict(X))

    # Perform random search with custom scorer
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from sklearn.metrics import make_scorer
    import scipy.stats as stats

    # Set up cross-validation
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    # Create base model
    base_lgb = LGBMClassifier(
        random_state=RANDOM_STATE,
        importance_type='gain',
        device='gpu' if use_gpu_for_lgb else 'cpu',
        verbose=-1
    )

    # Initialize random search with custom scorer
    random_search = RandomizedSearchCV(
        estimator=base_lgb,
        param_distributions=param_space,
        n_iter=n_iter,
        cv=skf,
        scoring=make_scorer(custom_scorer),
        n_jobs=-1 if not use_gpu_for_lgb else 1,  # Use 1 job for GPU to avoid conflicts
        verbose=2,
        random_state=RANDOM_STATE
    )

    # Perform random search
    random_search.fit(X_train_np, y_train_np)

    # Best parameters and score
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("\nBest LightGBM Parameters found by Random Search:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best {scoring_metric}: {best_score:.4f}")

    # Train final model on full dataset
    final_model = LGBMClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        importance_type='gain',
        verbose=-1
    )

    final_model.fit(X_train_np, y_train_np)

    # Find optimal threshold for binary classification on full training set
    print("\nFinding optimal threshold for LightGBM on full training set...")
    if hasattr(final_model, 'predict_proba'):
        y_pred_proba = final_model.predict_proba(X_train_np)[:, 1]
        optimal_threshold, threshold_score = find_optimal_threshold(y_train_np, y_pred_proba,
                                                                  threshold_metric)
        print(f"Optimal threshold for LightGBM: {optimal_threshold:.4f}")
        print(f"{threshold_metric} score at optimal threshold: {threshold_score:.4f}")

        # Verify performance with all metrics at optimal threshold
        y_pred_labels = (y_pred_proba >= optimal_threshold).astype(int)
        print("\nFull training set performance with optimal threshold:")
        print(f"  F1 Score: {f1_score(y_train_np, y_pred_labels):.4f}")
        print(f"  Accuracy: {accuracy_score(y_train_np, y_pred_labels):.4f}")
        print(f"  Precision: {precision_score(y_train_np, y_pred_labels):.4f}")
        print(f"  Recall: {recall_score(y_train_np, y_pred_labels):.4f}")
        print(f"  ROC AUC: {roc_auc_score(y_train_np, y_pred_proba):.4f}")
        print(f"  PR AUC: {average_precision_score(y_train_np, y_pred_proba):.4f}")

        final_model.optimal_threshold = optimal_threshold
    else:
        final_model.optimal_threshold = 0.5

    return final_model

# Train LightGBM model
print("\nTraining LightGBM model...")
if LIGHTGBM_AVAILABLE:
    # Changed to use random search with fewer iterations
    lgb_model = optimize_lightgbm_random_search(X_train_split, y_train_split, metrics_info, n_iter=15)

    # Evaluate LightGBM on validation set
    if lgb_model is not None:
        lgb_metrics = evaluate_model(lgb_model, X_val, y_val, "LightGBM", metrics_info, feature_names, using_gpu)
else:
    print("LightGBM not available. Skipping LightGBM model.")
    lgb_model = None
    lgb_metrics = None

import random

# 在使用DNNClassifier之前，确保运行这段代码
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

class DNNClassifier:
    def __init__(self, input_dim, hidden_layers=[128, 64], activation='relu',
                 dropout_rate=0.3, learning_rate=0.001, batch_norm=True,
                 is_imbalanced=False, random_state=42):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.is_imbalanced = is_imbalanced
        self.random_state = random_state
        self.model = None
        self.optimal_threshold = 0.5

        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        # Build the model
        self._build_model()

    def _build_model(self):
        model = Sequential()

        # Input layer
        model.add(Dense(self.hidden_layers[0], activation=self.activation,
                      input_dim=self.input_dim,
                      kernel_initializer='glorot_uniform'))

        if self.batch_norm:
            model.add(BatchNormalization())

        model.add(Dropout(self.dropout_rate))

        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation=self.activation,
                          kernel_initializer='glorot_uniform'))

            if self.batch_norm:
                model.add(BatchNormalization())

            model.add(Dropout(self.dropout_rate))

        # Output layer (binary classification)
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
                   tf.keras.metrics.AUC(name='roc_auc')]
        )

        self.model = model
        return model

    def fit(self, X, y, batch_size=64, epochs=100, validation_split=0.1,
           callbacks=None, verbose=1):
        # Handle class imbalance
        if self.is_imbalanced:
            # Calculate class weights
            n_samples = len(y)
            n_classes = 2  # Binary classification
            n_pos = np.sum(y)
            n_neg = n_samples - n_pos

            # Balanced weighting
            class_weight = {
                0: n_samples / (n_classes * n_neg),
                1: n_samples / (n_classes * n_pos)
            }
        else:
            class_weight = None

        # Default callbacks if not provided
        if callbacks is None:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]

        history = self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=verbose
        )

        return history

    def predict_proba(self, X):
        # TensorFlow/Keras models return a flat array for binary classification
        # We need to return a 2D array with shape (n_samples, 2) to match sklearn API
        y_pred = self.model.predict(X, verbose=0)

        # Convert to 2D array
        y_pred_2d = np.zeros((len(y_pred), 2))
        y_pred_2d[:, 1] = y_pred.flatten()
        y_pred_2d[:, 0] = 1 - y_pred.flatten()

        return y_pred_2d

    def predict(self, X):
        y_pred = self.model.predict(X, verbose=0).flatten()
        return (y_pred >= self.optimal_threshold).astype(int)

def optimize_dnn_random_search(X_train, y_train, metrics_info, cv=5, n_iter=20):
    """Use Random Search to optimize DNN hyperparameters"""
    print("\nOptimizing Deep Neural Network hyperparameters with Random Search...")

    # Define hyperparameter search space
    param_space = {
        'hidden_layers': [
            [128, 64],
            [256, 128, 64],
            [512, 256, 128],
            [256, 128],
            [512, 256],
            [64, 32]
        ],
        'activation': ['relu', 'elu'],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        'batch_norm': [True, False],
        'learning_rate': [0.0005, 0.001, 0.0025, 0.005, 0.0001],
        'batch_size': [32, 64, 128, 256]
    }

    # Convert to NumPy for TensorFlow
    X_train_np = to_numpy(X_train)
    y_train_np = to_numpy(y_train)

    # Get input dimension
    input_dim = X_train_np.shape[1]

    # Determine if dataset is imbalanced
    is_imbalanced = metrics_info['is_imbalanced']

    # Determine scoring metric
    scoring_metric = metrics_info['primary_metric']
    threshold_metric = metrics_info['secondary_metric']  # This is the metric for threshold optimization
    print(f"Using {scoring_metric} as the primary scoring metric")
    print(f"Using {threshold_metric} for threshold optimization")
    print(f"Testing {n_iter} random combinations instead of all {2*6*5*2*5*4} = 2400 possible combinations")

    # Set up cross-validation
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    # Track best model
    best_score = 0
    best_params = {}
    best_threshold = 0.5

    # Start time
    start_time = time.time()

    # Random search loop
    for i in range(n_iter):
        # Randomly sample parameters
        params = {
            'hidden_layers': random.choice(param_space['hidden_layers']),
            'activation': random.choice(param_space['activation']),
            'dropout_rate': random.choice(param_space['dropout_rate']),
            'batch_norm': random.choice(param_space['batch_norm']),
            'learning_rate': random.choice(param_space['learning_rate']),
            'batch_size': random.choice(param_space['batch_size'])
        }

        elapsed_time = time.time() - start_time
        estimated_remaining = (elapsed_time / (i + 1)) * (n_iter - i - 1)

        print(f"\nCombination {i+1}/{n_iter}:")
        print(f"  layers={params['hidden_layers']}")
        print(f"  activation={params['activation']}")
        print(f"  dropout={params['dropout_rate']}")
        print(f"  batch_norm={params['batch_norm']}")
        print(f"  lr={params['learning_rate']}")
        print(f"  batch_size={params['batch_size']}")
        print(f"Estimated time remaining: {estimated_remaining/60:.1f} minutes")

        # Cross-validation scores
        cv_scores = []
        cv_thresholds = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
            # Use reduced CV for speed
            if fold >= 3:  # Only use 3 folds for random search
                continue

            print(f"  Fold {fold+1}")

            # Get train/val split for this fold
            X_fold_train, X_fold_val = X_train_np[train_idx], X_train_np[val_idx]
            y_fold_train, y_fold_val = y_train_np[train_idx], y_train_np[val_idx]

            # Create model
            model = DNNClassifier(
                input_dim=input_dim,
                hidden_layers=params['hidden_layers'],
                activation=params['activation'],
                dropout_rate=params['dropout_rate'],
                learning_rate=params['learning_rate'],
                batch_norm=params['batch_norm'],
                is_imbalanced=is_imbalanced,
                random_state=RANDOM_STATE + fold
            )

            # Train with early stopping
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
            ]

            model.fit(
                X_fold_train, y_fold_train,
                batch_size=params['batch_size'],
                epochs=20,  # Limited epochs for random search
                validation_split=0.1,
                callbacks=callbacks,
                verbose=0
            )

            # Get probability predictions
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]

            # Calculate score based on whether the metric needs threshold
            if scoring_metric in ['roc_auc', 'average_precision']:
                # These metrics don't need threshold - they work directly with probabilities
                if scoring_metric == 'roc_auc':
                    score = roc_auc_score(y_fold_val, y_pred_proba)
                else:  # average_precision
                    score = average_precision_score(y_fold_val, y_pred_proba)
                # For these metrics, we still need to find the threshold for final predictions
                threshold, _ = find_optimal_threshold(y_fold_val, y_pred_proba, threshold_metric)
            else:
                # These metrics need threshold - find optimal threshold and calculate score
                threshold, _ = find_optimal_threshold(y_fold_val, y_pred_proba, scoring_metric)
                y_pred_labels = (y_pred_proba >= threshold).astype(int)

                if scoring_metric == 'f1':
                    score = f1_score(y_fold_val, y_pred_labels)
                elif scoring_metric == 'accuracy':
                    score = accuracy_score(y_fold_val, y_pred_labels)
                elif scoring_metric == 'precision':
                    score = precision_score(y_fold_val, y_pred_labels)
                elif scoring_metric == 'recall':
                    score = recall_score(y_fold_val, y_pred_labels)
                else:
                    score = f1_score(y_fold_val, y_pred_labels)  # Default to F1

            cv_scores.append(score)
            cv_thresholds.append(threshold)

            print(f"    Optimal threshold: {threshold:.4f}")
            print(f"    {scoring_metric}: {score:.4f}")

            # Clean up TF session
            tf.keras.backend.clear_session()

        # Average score across folds
        avg_score = np.mean(cv_scores)
        avg_threshold = np.mean(cv_thresholds)

        print(f"  Average {scoring_metric}: {avg_score:.4f}, Threshold: {avg_threshold:.4f}")

        # Update best parameters if new best found
        if avg_score > best_score:
            best_score = avg_score
            best_params = params.copy()
            best_threshold = avg_threshold

    # Print best parameters
    print("\nBest DNN Parameters found by Random Search:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best {scoring_metric}: {best_score:.4f}")
    print(f"Best threshold: {best_threshold:.4f}")

    # Train final model with best parameters
    print("\nTraining final DNN model with best parameters...")
    final_model = DNNClassifier(
        input_dim=input_dim,
        hidden_layers=best_params['hidden_layers'],
        activation=best_params['activation'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate'],
        batch_norm=best_params['batch_norm'],
        is_imbalanced=is_imbalanced,
        random_state=RANDOM_STATE
    )

    # Train final model with more epochs
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    final_model.fit(
        X_train_np, y_train_np,
        batch_size=best_params['batch_size'],
        epochs=50,  # More epochs for final model
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # Find optimal threshold for binary classification on full training set
    print("\nFinding optimal threshold on full training set...")
    y_pred_proba_full = final_model.predict_proba(X_train_np)[:, 1]

    # Use the secondary metric for threshold optimization
    optimal_threshold, threshold_score = find_optimal_threshold(y_train_np, y_pred_proba_full, threshold_metric)
    print(f"Optimal threshold for DNN ({threshold_metric}): {optimal_threshold:.4f}")
    print(f"{threshold_metric} score at optimal threshold: {threshold_score:.4f}")

    # Verify that the threshold works well with all metrics
    y_pred_labels_full = (y_pred_proba_full >= optimal_threshold).astype(int)
    print("\nFull training set performance with optimal threshold:")
    print(f"  F1 Score: {f1_score(y_train_np, y_pred_labels_full):.4f}")
    print(f"  Accuracy: {accuracy_score(y_train_np, y_pred_labels_full):.4f}")
    print(f"  Precision: {precision_score(y_train_np, y_pred_labels_full):.4f}")
    print(f"  Recall: {recall_score(y_train_np, y_pred_labels_full):.4f}")
    print(f"  ROC AUC: {roc_auc_score(y_train_np, y_pred_proba_full):.4f}")
    print(f"  PR AUC: {average_precision_score(y_train_np, y_pred_proba_full):.4f}")

    final_model.optimal_threshold = optimal_threshold

    return final_model

# Replace the original optimize_dnn function call with:
print("\nTraining Deep Neural Network model...")
dnn_model = optimize_dnn_random_search(X_train_split, y_train_split, metrics_info, n_iter=20)

# Evaluate DNN on validation set
dnn_metrics = evaluate_model(dnn_model, X_val, y_val, "DNN", metrics_info, feature_names,using_gpu)

# Use stacking to combine the best models
def stack_models(base_models, X_train, y_train, metrics_info, X_val=None, y_val=None, using_gpu=True, cv=5):
    """Implement stacking of multiple base models to meta-model (Logistic Regression)"""
    print("\nImplementing stacking...")

    # Generate meta-features for stacking
    meta_features, val_meta_features = generate_meta_features(
        base_models, X_train, y_train, X_val, using_gpu, cv
    )

    # Convert meta-features to numpy
    meta_features_np = to_numpy(meta_features)

    # Convert target to numpy
    y_train_np = to_numpy(y_train)

    # Check for class imbalance
    is_imbalanced = metrics_info.get('is_imbalanced', False)

    # Train a logistic regression meta-model
    print("Creating Logistic Regression meta-model...")
    from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

    meta_model = SklearnLogisticRegression(
        C=1.0,
        penalty='l2',
        solver='lbfgs',
        class_weight='balanced' if is_imbalanced else None,
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    meta_model.fit(meta_features_np, y_train_np)

    # Find optimal threshold for binary classification
    if hasattr(meta_model, 'predict_proba'):
        y_pred_proba = meta_model.predict_proba(meta_features_np)[:, 1]
        optimal_threshold, _ = find_optimal_threshold(
            y_train_np, y_pred_proba, metrics_info['secondary_metric']
        )
        print(f"Optimal threshold for meta-model: {optimal_threshold:.4f}")
        meta_model.optimal_threshold = optimal_threshold
    else:
        meta_model.optimal_threshold = 0.5

    return meta_model, base_models


def evaluate_stacked_model(meta_model, base_models, X, y, model_name, metrics_info, feature_names=None, using_gpu=True):
    """Evaluate a stacked model"""
    # Get optimal threshold for meta model if available
    optimal_threshold = getattr(meta_model, 'optimal_threshold', 0.5)

    print(f"\nEvaluating {model_name}...")

    # Convert to CPU/GPU as needed
    X_np = to_numpy(X)
    y_np = to_numpy(y)

    # Generate meta-features
    meta_features = np.zeros((X_np.shape[0], len(base_models)))

    for i, model in enumerate(base_models):
        try:
            if hasattr(model, 'predict_proba'):
                model_preds = model.predict_proba(X_np)

                # Get probabilities for positive class
                if model_preds.shape[1] > 1:
                    meta_features[:, i] = model_preds[:, 1]
                else:
                    meta_features[:, i] = model_preds
            else:
                meta_features[:, i] = model.predict(X_np)
        except Exception as e:
            print(f"Error generating meta-features for model {i}: {e}")
            print("Using zeros for this model")
            meta_features[:, i] = 0.0

    # Predict with meta-model
    try:
        if hasattr(meta_model, 'predict_proba'):
            y_pred_proba = meta_model.predict_proba(meta_features)

            # Get probabilities for positive class
            if y_pred_proba.shape[1] > 1:
                y_pred_proba_np = y_pred_proba[:, 1]
            else:
                y_pred_proba_np = y_pred_proba

            # Convert probabilities to class labels using optimal threshold
            y_pred_np = (y_pred_proba_np >= optimal_threshold).astype(int)
        else:
            y_pred_np = meta_model.predict(meta_features)
            y_pred_proba_np = y_pred_np
    except Exception as e:
        print(f"Error in meta-model prediction: {e}")
        y_pred_np = np.zeros_like(y_np)
        y_pred_proba_np = np.zeros_like(y_np)

    # Calculate metrics
    accuracy = accuracy_score(y_np, y_pred_np)
    precision = precision_score(y_np, y_pred_np)
    recall = recall_score(y_np, y_pred_np)
    f1 = f1_score(y_np, y_pred_np)
    roc_auc = roc_auc_score(y_np, y_pred_proba_np)
    pr_auc = average_precision_score(y_np, y_pred_proba_np)

    # Confusion matrix
    cm = confusion_matrix(y_np, y_pred_np)

    # Print metrics
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  PR AUC: {pr_auc:.4f}")

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(cm)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_np, y_pred_np))

    # Create output directory if it doesn't exist
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'outputs/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_np, y_pred_proba_np)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'outputs/{model_name.lower().replace(" ", "_")}_roc_curve.png')
    plt.close()

    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'optimal_threshold': optimal_threshold  # Include optimal threshold in metrics
    }

    return metrics


def compare_models(models_metrics, metrics_info):
    """Compare models based on their metrics"""
    print("\nComparing models...")

    # Determine primary metric for comparison
    primary_metric = metrics_info['primary_metric']

    # Map primary_metric to the actual key used in metrics dictionary
    metric_mapping = {
        'average_precision': 'pr_auc',
        'roc_auc': 'roc_auc',
        'f1': 'f1',
        'accuracy': 'accuracy'
    }

    # Get the correct metric key
    metric_key = metric_mapping.get(primary_metric, primary_metric)

    # Convert metrics dictionary to DataFrame
    metrics_df = pd.DataFrame({
        model_name: {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1'],
            'ROC AUC': metrics['roc_auc'],
            'PR AUC': metrics['pr_auc'],
            'Threshold': metrics.get('optimal_threshold', 0.5)
        }
        for model_name, metrics in models_metrics.items()
    }).T

    # Print metrics
    print("\nModel Metrics:")
    print(metrics_df)

    # Create output directory if it doesn't exist
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    # Plot metrics comparison
    plt.figure(figsize=(12, 8))
    metrics_df.drop(columns=['Threshold']).plot(kind='bar', figsize=(12, 8))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png')
    plt.close()

    # Determine best model based on primary metric
    if primary_metric == 'roc_auc':
        metric_name = 'ROC AUC'
    elif primary_metric == 'average_precision':
        metric_name = 'PR AUC'
    elif primary_metric == 'f1':
        metric_name = 'F1 Score'
    else:
        metric_name = 'Accuracy'

    best_model = max(models_metrics.items(), key=lambda x: x[1][metric_key])[0]

    print(f"\nBest model based on {metric_name}: {best_model}")
    print(f"  {metric_name}: {models_metrics[best_model][metric_key]:.4f}")

    return best_model

def generate_meta_features(base_models, X_train, y_train, X_val=None, using_gpu=True, cv=5):
    """Generate meta-features for stacking using cross-validation predictions"""
    print("\nGenerating meta-features for stacking...")

    # Convert to numpy for cross-validation
    X_train_np = to_numpy(X_train)
    y_train_np = to_numpy(y_train)

    # Set up cross-validation
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    # Initialize array to store meta-features (one column per model)
    meta_features = np.zeros((len(y_train_np), len(base_models)))

    # Generate predictions for each fold and each model
    for i, model in enumerate(base_models):
        print(f"Generating meta-features for model {i+1}/{len(base_models)}...")

        # Generate predictions for each fold
        fold_preds = np.zeros(len(y_train_np))

        for train_idx, val_idx in skf.split(X_train_np, y_train_np):
            # Get training and validation data for this fold
            X_fold_train_np = X_train_np[train_idx]
            y_fold_train_np = y_train_np[train_idx]
            X_fold_val_np = X_train_np[val_idx]

            try:
                # Special handling for DNNClassifier since it can't be cloned
                if isinstance(model, DNNClassifier):
                    # Create a new instance of DNNClassifier with the same parameters
                    input_dim = X_fold_train_np.shape[1]
                    fold_model = DNNClassifier(
                        input_dim=input_dim,
                        hidden_layers=getattr(model, 'hidden_layers', [128, 64]),
                        activation=getattr(model, 'activation', 'relu'),
                        dropout_rate=getattr(model, 'dropout_rate', 0.3),
                        learning_rate=getattr(model, 'learning_rate', 0.001),
                        batch_norm=getattr(model, 'batch_norm', True),
                        is_imbalanced=getattr(model, 'is_imbalanced', False),
                        random_state=RANDOM_STATE
                    )
                else:
                    # For other models, use sklearn's clone
                    from sklearn.base import clone
                    fold_model = clone(model)

                # Train the model
                fold_model.fit(X_fold_train_np, y_fold_train_np)

                # Generate predictions
                if hasattr(fold_model, 'predict_proba'):
                    val_preds = fold_model.predict_proba(X_fold_val_np)

                    # Get probabilities for positive class
                    if val_preds.shape[1] > 1:
                        fold_preds[val_idx] = val_preds[:, 1]
                    else:
                        fold_preds[val_idx] = val_preds
                else:
                    fold_preds[val_idx] = fold_model.predict(X_fold_val_np)
            except Exception as e:
                print(f"Error in fold processing: {e}")
                print("Using zeros for this fold")
                fold_preds[val_idx] = 0.0

        # Store predictions as meta-features
        meta_features[:, i] = fold_preds

    # Generate meta-features for validation set if provided
    val_meta_features = None
    if X_val is not None:
        print("Generating meta-features for validation set...")

        # Convert to numpy
        X_val_np = to_numpy(X_val)

        # Initialize array to store validation meta-features
        val_meta_features = np.zeros((X_val_np.shape[0], len(base_models)))

        # Generate predictions for each model
        for i, model in enumerate(base_models):
            try:
                if hasattr(model, 'predict_proba'):
                    val_preds = model.predict_proba(X_val_np)

                    # Get probabilities for positive class
                    if val_preds.shape[1] > 1:
                        val_meta_features[:, i] = val_preds[:, 1]
                    else:
                        val_meta_features[:, i] = val_preds
                else:
                    val_meta_features[:, i] = model.predict(X_val_np)
            except Exception as e:
                print(f"Error generating validation meta-features: {e}")
                print("Using zeros for this model")
                val_meta_features[:, i] = 0.0

    print(f"Meta-features shape: {meta_features.shape}")
    if val_meta_features is not None:
        print(f"Validation meta-features shape: {val_meta_features.shape}")

    return meta_features, val_meta_features

# Collect validation metrics (only include LightGBM and DNN)
val_metrics = {}

# Add LightGBM if available
if 'lgb_model' in locals() and lgb_model is not None:
    val_metrics["LightGBM"] = lgb_metrics

# Add DNN if available
if 'dnn_metrics' in locals():
    val_metrics["DNN"] = dnn_metrics
else:
    print("DNN metrics not found. Skipping DNN for stacking.")

# If we have both models, use them directly for stacking
base_models = []
if 'lgb_model' in locals() and lgb_model is not None:
    base_models.append(lgb_model)
if 'dnn_model' in locals() and dnn_model is not None:
    base_models.append(dnn_model)

print(f"\nBase models for stacking: {[type(model).__name__ for model in base_models]}")

# Use LightGBM and DNN as base models, logistic regression as meta learner for stacking
meta_model, base_models = stack_models(base_models, X_train_split, y_train_split, metrics_info, X_val, y_val, using_gpu)

# Evaluate stacked model
stacked_metrics = evaluate_stacked_model(meta_model, base_models, X_val, y_val, "Stacked Model", metrics_info, feature_names, using_gpu)

# Compare models (only include LightGBM, DNN and Stacked Model)
validation_metrics = {}
for model_name, metrics in val_metrics.items():
    validation_metrics[model_name] = metrics

validation_metrics["Stacked Model"] = stacked_metrics

print("\n--- Validation Results ---")
best_model = compare_models(validation_metrics, metrics_info)

# Save validation results
val_results_df = pd.DataFrame({
    model_name: {
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1 Score': metrics['f1'],
        'ROC AUC': metrics['roc_auc'],
        'PR AUC': metrics['pr_auc'],
        'Threshold': metrics.get('optimal_threshold', 0.5)
    }
    for model_name, metrics in validation_metrics.items()
}).T

val_results_df.to_csv('outputs/validation_results.csv')
print("Saved validation results to outputs/validation_results.csv")

print("\n--- Training on Full Dataset and Evaluating on Test Set ---")

# Store all trained models (excluding standalone logistic regression)
all_models = {
    'LightGBM': lgb_model if 'lgb_model' in locals() and lgb_model is not None else None,
    'DNN': dnn_model if 'dnn_model' in locals() and dnn_model is not None else None,
    'Stacked Model': (meta_model, base_models)
}

# Store performance results
performance_results = {'Training': {}, 'Test': {}}

# Create output directory
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# Evaluate all models
print("\nEvaluating all models...")

for model_name, model in all_models.items():
    if model is not None:
        print(f"\nEvaluating {model_name}...")

        if model_name == 'Stacked Model':
            # Evaluate stacked model
            meta_model, base_models = model

            train_metrics = evaluate_stacked_model(meta_model, base_models, X_train_processed, y_train,
                                                 f"{model_name} (Training)", metrics_info, feature_names, using_gpu)

            if y_test is not None:
                test_metrics = evaluate_stacked_model(meta_model, base_models, X_test_processed, y_test,
                                                    f"{model_name} (Test)", metrics_info, feature_names, using_gpu)
            else:
                test_metrics = None
        else:
            # Evaluate base models
            train_metrics = evaluate_model(model, X_train_processed, y_train,
                                         f"{model_name} (Training)", metrics_info, feature_names, using_gpu)

            if y_test is not None:
                test_metrics = evaluate_model(model, X_test_processed, y_test,
                                            f"{model_name} (Test)", metrics_info, feature_names, using_gpu)
            else:
                test_metrics = None

        performance_results['Training'][model_name] = train_metrics
        performance_results['Test'][model_name] = test_metrics

# Use best model for final predictions
best_model_name = best_model
print(f"\nUsing {best_model_name} for final predictions...")

if best_model_name == 'Stacked Model':
    best_final_model = all_models['Stacked Model']
else:
    best_final_model = all_models[best_model_name]

# Generate test predictions
print("Generating final predictions on test set...")

if case_id is not None:
    print("Generating test predictions with optimal threshold from training...")

    if best_model_name == 'Stacked Model':
        meta_model, base_models = best_final_model

        # Generate meta-features for test set
        X_test_np = to_numpy(X_test_processed)
        meta_features_test = np.zeros((X_test_np.shape[0], len(base_models)))

        for i, base_model in enumerate(base_models):
            try:
                if hasattr(base_model, 'predict_proba'):
                    model_preds = base_model.predict_proba(X_test_np)
                    if model_preds.shape[1] > 1:
                        meta_features_test[:, i] = model_preds[:, 1]
                    else:
                        meta_features_test[:, i] = model_preds
                else:
                    meta_features_test[:, i] = base_model.predict(X_test_np)
            except Exception as e:
                print(f"Error generating test meta-features: {e}")
                meta_features_test[:, i] = 0.0

        optimal_threshold = getattr(meta_model, 'optimal_threshold', 0.5)
        print(f"Using optimal threshold: {optimal_threshold:.4f}")

        if hasattr(meta_model, 'predict_proba'):
            y_pred_proba_test = meta_model.predict_proba(meta_features_test)
            if y_pred_proba_test.shape[1] > 1:
                y_pred_proba_test = y_pred_proba_test[:, 1]
            y_pred_test = (y_pred_proba_test >= optimal_threshold).astype(int)
        else:
            y_pred_test = meta_model.predict(meta_features_test)
    else:
        # Make predictions with base model
        model = best_final_model
        X_test_np = to_numpy(X_test_processed)

        optimal_threshold = getattr(model, 'optimal_threshold', 0.5)
        print(f"Using optimal threshold: {optimal_threshold:.4f}")

        if hasattr(model, 'predict_proba'):
            y_pred_proba_test = model.predict_proba(X_test_np)
            if y_pred_proba_test.shape[1] > 1:
                y_pred_proba_test = y_pred_proba_test[:, 1]
            y_pred_test = (y_pred_proba_test >= optimal_threshold).astype(int)
        else:
            y_pred_test = model.predict(X_test_np)

    # Create prediction DataFrame
    test_predictions = pd.DataFrame({
        'case_id': case_id,
        'target': y_pred_test
    })

    # Save to CSV
    test_predictions.to_csv('outputs/test_predictions.csv', index=False)
    print("Saved test predictions to outputs/test_predictions.csv")
    print(f"Preview of predictions:\n{test_predictions.head()}")
else:
   print("Could not generate test predictions - case_id not available")

print("\nCreating performance comparison charts...")

# Extract metrics for visualization
training_metrics_data = []
test_metrics_data = []

for model_name, metrics in performance_results['Training'].items():
   training_metrics_data.append({
       'Model': model_name,
       'Accuracy': metrics['accuracy'],
       'Precision': metrics['precision'],
       'Recall': metrics['recall'],
       'F1 Score': metrics['f1'],
       'ROC AUC': metrics['roc_auc'],
       'PR AUC': metrics['pr_auc']
   })

for model_name, metrics in performance_results['Test'].items():
   if metrics is not None:
       test_metrics_data.append({
           'Model': model_name,
           'Accuracy': metrics['accuracy'],
           'Precision': metrics['precision'],
           'Recall': metrics['recall'],
           'F1 Score': metrics['f1'],
           'ROC AUC': metrics['roc_auc'],
           'PR AUC': metrics['pr_auc']
       })

# Convert to DataFrames
training_df = pd.DataFrame(training_metrics_data)
test_df = pd.DataFrame(test_metrics_data)

# Create comparison chart
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 15))

# Training performance
training_plot_data = training_df.set_index('Model').plot(kind='bar', ax=ax1)
ax1.set_title('Model Performance on Training Set', fontsize=16, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
ax1.legend(loc='lower right')
ax1.grid(True, axis='y', alpha=0.3)

# Test performance (if available)
if not test_df.empty:
   test_plot_data = test_df.set_index('Model').plot(kind='bar', ax=ax2)
   ax2.set_title('Model Performance on Test Set', fontsize=16, fontweight='bold')
   ax2.set_ylabel('Score', fontsize=12)
   ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
   ax2.legend(loc='lower right')
   ax2.grid(True, axis='y', alpha=0.3)
else:
   ax2.text(0.5, 0.5, 'Test data labels not available', ha='center', va='center',
            transform=ax2.transAxes, fontsize=16)
   ax2.set_xlim(0, 1)
   ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('outputs/model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create detailed metric comparison chart (if test data available)
if not test_df.empty:
   metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'PR AUC']
   fig, axes = plt.subplots(2, 3, figsize=(18, 12))
   axes = axes.ravel()

   for idx, metric in enumerate(metrics_list):
       ax = axes[idx]

       # Extract metric values for each model
       train_values = []
       test_values = []
       models = training_df['Model'].tolist()

       for model in models:
           train_val = training_df[training_df['Model'] == model][metric].values[0]
           test_val = test_df[test_df['Model'] == model][metric].values[0] if model in test_df['Model'].values else None

           train_values.append(train_val)
           test_values.append(test_val if test_val is not None else 0)

       # Create grouped bar chart
       x = np.arange(len(models))
       width = 0.35

       ax.bar(x - width/2, train_values, width, label='Training')
       ax.bar(x + width/2, test_values, width, label='Test')

       ax.set_ylabel(metric)
       ax.set_title(f'{metric} Comparison')
       ax.set_xticks(x)
       ax.set_xticklabels(models, rotation=45, ha='right')
       ax.legend()
       ax.grid(True, axis='y', alpha=0.3)

   plt.tight_layout()
   plt.savefig('outputs/model_metrics_comparison_detailed.png', dpi=300, bbox_inches='tight')
   plt.close()

# Save performance results to CSV
training_df.to_csv('outputs/training_performance.csv', index=False)
if not test_df.empty:
   test_df.to_csv('outputs/test_performance.csv', index=False)

print("\nAll evaluations complete!")
print(f"Best model for final predictions: {best_model_name}")
print(f"Generated files:")
print(f"  - outputs/test_predictions.csv")
print(f"  - outputs/model_performance_comparison.png")
if not test_df.empty:
    print(f"  - outputs/model_metrics_comparison_detailed.png")
print(f"  - outputs/training_performance.csv")
if not test_df.empty:
   print(f"  - outputs/test_performance.csv")
