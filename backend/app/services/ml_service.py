import os
import uuid
# pyrefly: ignore [missing-import]
import joblib
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import math
from datetime import datetime
from typing import Tuple, Dict, Any, List

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from app.config import UPLOAD_DIR, MODEL_DIR, DEFAULT_RANDOM_STATE
from app.models.schemas import TrainingConfig, TrainedModel, ModelMetrics

def load_dataset(dataset_id: str) -> pd.DataFrame:
    """Loads a dataset from the upload directory."""
    for ext in ['.csv', '.json']:
        file_path = os.path.join(UPLOAD_DIR, f"{dataset_id}{ext}")
        if os.path.exists(file_path):
            if ext == '.csv':
                return pd.read_csv(file_path)
            else:
                return pd.read_json(file_path)
    raise FileNotFoundError(f"Dataset {dataset_id} not found.")

def preprocess_data(df: pd.DataFrame, target_column: str, drop_columns: list[str]) -> Tuple[pd.DataFrame, pd.Series, bool]:
    """
    Basic preprocessing: drop columns, handle missing values, encode categoricals.
    Returns X, y, and a boolean indicating if it's a classification task.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # Drop rows where target is missing (use .copy() to avoid SettingWithCopyWarning)
    df = df.dropna(subset=[target_column]).copy()
    if len(df) == 0:
        raise ValueError("Dataset is empty after dropping rows with missing target values.")

    # Drop requested columns
    df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors='ignore')

    # Separate features and target
    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()

    if X.shape[1] == 0:
        raise ValueError("No features remaining for training. Please keep at least one feature column.")

    # Determine task type based on target dtype and unique values
    is_classification = False
    if y.dtype == 'object' or y.nunique() < 20: # Heuristic for classification
        is_classification = True
        if y.dtype == 'object':
            # Encode target if it's categorical strings
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=target_column)

    # Impute missing values (simple strategy for baseline)
    # First, replace infinity with NaN so SimpleImputer can handle it
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Numeric features -> median
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        X[numeric_cols] = imputer_num.fit_transform(X[numeric_cols])

    # Categorical features -> most frequent, then one-hot encode
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y, is_classification

def _sanitize_metric(value):
    """Replace NaN/Inf with None so JSON serialization never fails."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_sanitize_metric(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_metric(v) for k, v in value.items()}
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.floating):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def train_model(config: TrainingConfig) -> TrainedModel:
    """
    Trains a model based on the configuration and saves it.
    """
    df = load_dataset(config.dataset_id)
    X, y, is_classification = preprocess_data(df, config.target_column, config.drop_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=DEFAULT_RANDOM_STATE
    )

    # Select model (model_type is already validated by Pydantic Literal)
    actual_model_type = config.model_type
    hp = config.hyperparams or {}

    # Whitelist safe hyperparameters per algorithm
    def _pick(allowed_keys: list, defaults: dict) -> dict:
        """Merge user-provided hyperparams with defaults, only allowing whitelisted keys."""
        merged = {**defaults}
        for k in allowed_keys:
            if k in hp:
                merged[k] = hp[k]
        return merged

    if is_classification:
        if config.model_type == "random_forest":
            params = _pick(
                ["n_estimators", "max_depth", "min_samples_split"],
                {"random_state": DEFAULT_RANDOM_STATE}
            )
            model = RandomForestClassifier(**params)
        elif config.model_type == "gradient_boosting":
            params = _pick(
                ["n_estimators", "max_depth", "learning_rate"],
                {"random_state": DEFAULT_RANDOM_STATE}
            )
            model = GradientBoostingClassifier(**params)
        elif config.model_type == "xgboost":
            params = _pick(
                ["n_estimators", "max_depth", "learning_rate", "reg_alpha", "reg_lambda"],
                {"random_state": DEFAULT_RANDOM_STATE}
            )
            model = xgb.XGBClassifier(**params)
        elif config.model_type == "lightgbm":
            params = _pick(
                ["n_estimators", "max_depth", "learning_rate", "reg_alpha", "reg_lambda"],
                {"random_state": DEFAULT_RANDOM_STATE}
            )
            model = lgb.LGBMClassifier(**params)
        elif config.model_type == "mlp":
            params = _pick(
                ["max_iter", "alpha", "early_stopping"],
                {"random_state": DEFAULT_RANDOM_STATE, "max_iter": 500, "hidden_layer_sizes": (100,)}
            )
            model = MLPClassifier(**params)
        else:  # logistic_regression
            params = _pick(
                ["max_iter", "C"],
                {"random_state": DEFAULT_RANDOM_STATE, "max_iter": 1000}
            )
            model = LogisticRegression(**params)
    else:
        if config.model_type == "random_forest":
            params = _pick(
                ["n_estimators", "max_depth", "min_samples_split"],
                {"random_state": DEFAULT_RANDOM_STATE}
            )
            model = RandomForestRegressor(**params)
        elif config.model_type == "gradient_boosting":
            params = _pick(
                ["n_estimators", "max_depth", "learning_rate"],
                {"random_state": DEFAULT_RANDOM_STATE}
            )
            model = GradientBoostingRegressor(**params)
        elif config.model_type == "xgboost":
            params = _pick(
                ["n_estimators", "max_depth", "learning_rate", "reg_alpha", "reg_lambda"],
                {"random_state": DEFAULT_RANDOM_STATE}
            )
            model = xgb.XGBRegressor(**params)
        elif config.model_type == "lightgbm":
            params = _pick(
                ["n_estimators", "max_depth", "learning_rate", "reg_alpha", "reg_lambda"],
                {"random_state": DEFAULT_RANDOM_STATE}
            )
            model = lgb.LGBMRegressor(**params)
        elif config.model_type == "mlp":
            params = _pick(
                ["max_iter", "alpha", "early_stopping"],
                {"random_state": DEFAULT_RANDOM_STATE, "max_iter": 500, "hidden_layer_sizes": (100,)}
            )
            model = MLPRegressor(**params)
        else:  # logistic_regression → use LinearRegression/Ridge for regression tasks
            params = _pick(["alpha"], {})
            if "alpha" in params:
                model = Ridge(**params)
            else:
                model = LinearRegression()
            actual_model_type = "linear_regression"

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    metrics = {}
    
    if is_classification:
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['train_accuracy'] = accuracy_score(y_train, y_pred_train)
        metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')
        metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        # classification_report returns a string by default, output_dict=True returns dict
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    else:
        metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_pred_train))
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['r2'] = r2_score(y_test, y_pred)
        metrics['train_r2'] = r2_score(y_train, y_pred_train)

    # Sanitize all metrics to prevent NaN/Inf from breaking JSON serialization
    metrics = {k: _sanitize_metric(v) for k, v in metrics.items()}

    # Save model artifact and feature names (critical for SHAP/LIME later)
    model_id = str(uuid.uuid4())
    artifact = {
        'model': model,
        'feature_names': X.columns.tolist(),
        'target_column': config.target_column,
        'is_classification': is_classification,
        'model_type': actual_model_type,
        'drop_columns': config.drop_columns,
    }
    
    joblib.dump(artifact, os.path.join(MODEL_DIR, f"{model_id}.joblib"))

    return TrainedModel(
        id=model_id,
        dataset_id=config.dataset_id,
        model_type=actual_model_type,
        target_column=config.target_column,
        feature_names=X.columns.tolist(),
        task_type="classification" if is_classification else "regression",
        metrics=ModelMetrics(**metrics),
        created_at=datetime.utcnow().isoformat()
    )

def smart_train_model(config: TrainingConfig) -> TrainedModel:
    """
    Trains a model. If smart_regularization is enabled, checks for overfitting
    and applies strong regularization if a gap > 5% is detected.
    """
    import copy
    
    # 1. Train baseline
    baseline_model = train_model(config)
    if not config.smart_regularization:
        return baseline_model
        
    is_classification = baseline_model.task_type == "classification"
    metrics = baseline_model.metrics
    
    # 2. Detect Overfitting
    overfit = False
    if is_classification:
        train_acc = metrics.train_accuracy or 0
        test_acc = metrics.accuracy or 0
        if (train_acc - test_acc) > 0.05:
            overfit = True
    else:
        train_r2 = metrics.train_r2 or 0
        test_r2 = metrics.r2 or 0
        # If train is much better than test, we are overfitting
        if (train_r2 - test_r2) > 0.05:
            overfit = True
            
    if not overfit:
        return baseline_model
        
    # 3. Apply Strong Regularization based on model_type
    reg_config = copy.deepcopy(config)
    if reg_config.hyperparams is None:
        reg_config.hyperparams = {}
        
    mt = reg_config.model_type
    
    if mt == "random_forest":
        reg_config.hyperparams["max_depth"] = 5
        reg_config.hyperparams["min_samples_split"] = 10
    elif mt == "xgboost":
        reg_config.hyperparams["max_depth"] = 3
        reg_config.hyperparams["reg_lambda"] = 5.0 # L2
        reg_config.hyperparams["reg_alpha"] = 1.0  # L1
    elif mt == "lightgbm":
        reg_config.hyperparams["max_depth"] = 3
        reg_config.hyperparams["reg_lambda"] = 5.0
        reg_config.hyperparams["reg_alpha"] = 1.0
    elif mt == "mlp":
        reg_config.hyperparams["alpha"] = 0.05
        reg_config.hyperparams["early_stopping"] = True
    elif mt == "logistic_regression":
        if is_classification:
            reg_config.hyperparams["C"] = 0.01  # Strong L2
        else:
            reg_config.hyperparams["alpha"] = 10.0 # Ridge penalty
            
    # 4. Retrain
    reg_model = train_model(reg_config)
    
    # 5. Compare
    if is_classification:
        reg_test = reg_model.metrics.accuracy or 0
        base_test = baseline_model.metrics.accuracy or 0
    else:
        reg_test = reg_model.metrics.r2 or -9999
        base_test = baseline_model.metrics.r2 or -9999
        
    if reg_test >= base_test:
        return reg_model
        
    # If regularization hurt test performance more than it helped generalization, return baseline
    return baseline_model
