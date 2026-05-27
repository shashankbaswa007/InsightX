import os
import uuid
import joblib
import pandas as pd
import numpy as np
import math
from datetime import datetime
from typing import Tuple, Dict, Any, List

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder, RobustScaler, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE

from app.config import UPLOAD_DIR, MODEL_DIR, DEFAULT_RANDOM_STATE
from app.models.schemas import TrainingConfig, TrainedModel, ModelMetrics

def load_dataset(dataset_id: str) -> pd.DataFrame:
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
    Cleans target NAs, separates X and y, and infers classification.
    Imputation, scaling, and encoding are handled post-split to prevent data leakage.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    df = df.dropna(subset=[target_column]).copy()
    if len(df) == 0:
        raise ValueError("Dataset is empty after dropping rows with missing target values.")

    df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors='ignore')
    
    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()

    if X.shape[1] == 0:
        raise ValueError("No features remaining for training. Please keep at least one feature column.")

    is_classification = False
    if y.dtype == 'object' or y.nunique() < 20: 
        is_classification = True
        if y.dtype == 'object':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=target_column)

    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y, is_classification

def apply_transformers(X: pd.DataFrame, transformers: dict) -> pd.DataFrame:
    """Applies the fitted preprocessing transformers to a raw DataFrame."""
    if not transformers:
        return X
    
    X_out = X.copy()
    num_cols = transformers.get('numeric_cols', [])
    cat_cols = transformers.get('cat_cols', [])
    
    if len(num_cols) > 0:
        X_out.loc[:, num_cols] = X_out[num_cols].astype(float)
        X_out_num = X_out[num_cols]
        if transformers.get('imputer_num'):
            X_out_num = transformers['imputer_num'].transform(X_out_num)
        if transformers.get('scaler'):
            X_out.loc[:, num_cols] = transformers['scaler'].transform(X_out_num)
            
    if len(cat_cols) > 0:
        if transformers.get('imputer_cat'):
            X_out.loc[:, cat_cols] = transformers['imputer_cat'].transform(X_out[cat_cols])
        if transformers.get('te'):
            X_out.loc[:, cat_cols] = transformers['te'].transform(X_out[cat_cols])
            
    return X_out

def _sanitize_metric(value):
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
    df = load_dataset(config.dataset_id)
    X, y, is_classification = preprocess_data(df, config.target_column, config.drop_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=DEFAULT_RANDOM_STATE
    )

    # 1. IMPUTATION, ENCODING & SCALING (Fit on Train, Transform on Test)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns

    transformers = {
        'numeric_cols': numeric_cols.tolist(),
        'cat_cols': cat_cols.tolist(),
        'imputer_num': None,
        'scaler': None,
        'imputer_cat': None,
        'te': None
    }

    if len(numeric_cols) > 0:
        X_train.loc[:, numeric_cols] = X_train[numeric_cols].astype(float)
        X_test.loc[:, numeric_cols] = X_test[numeric_cols].astype(float)
        
        imputer_num = IterativeImputer(random_state=DEFAULT_RANDOM_STATE)
        X_train_num = imputer_num.fit_transform(X_train[numeric_cols])
        X_test_num = imputer_num.transform(X_test[numeric_cols])
        
        scaler = RobustScaler()
        X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train_num)
        X_test.loc[:, numeric_cols] = scaler.transform(X_test_num)

        transformers['imputer_num'] = imputer_num
        transformers['scaler'] = scaler

    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X_train.loc[:, cat_cols] = imputer_cat.fit_transform(X_train[cat_cols])
        X_test.loc[:, cat_cols] = imputer_cat.transform(X_test[cat_cols])
        
        te = TargetEncoder(target_type="auto", random_state=DEFAULT_RANDOM_STATE)
        X_train.loc[:, cat_cols] = te.fit_transform(X_train[cat_cols], y_train)
        X_test.loc[:, cat_cols] = te.transform(X_test[cat_cols])

        transformers['imputer_cat'] = imputer_cat
        transformers['te'] = te

    # 2. SMOTE FOR CLASSIFICATION IMBALANCE
    if is_classification:
        class_counts = y_train.value_counts(normalize=True)
        if class_counts.min() < 0.2:
            try:
                smote = SMOTE(random_state=DEFAULT_RANDOM_STATE)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except ValueError:
                pass # Fails if a class has extremely few samples

    actual_model_type = config.model_type
    hp = config.hyperparams or {}

    def _pick(allowed_keys: list, defaults: dict) -> dict:
        merged = {**defaults}
        for k in allowed_keys:
            if k in hp:
                merged[k] = hp[k]
        return merged

    model = None
    param_grid = {}

    if is_classification:
        if config.model_type == "random_forest":
            base_model = RandomForestClassifier(random_state=DEFAULT_RANDOM_STATE)
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]}
            if hp: model = RandomForestClassifier(**_pick(["n_estimators", "max_depth", "min_samples_split"], {"random_state": DEFAULT_RANDOM_STATE}))
        elif config.model_type == "gradient_boosting":
            base_model = GradientBoostingClassifier(random_state=DEFAULT_RANDOM_STATE)
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10], "learning_rate": [0.01, 0.1, 0.2]}
            if hp: model = GradientBoostingClassifier(**_pick(["n_estimators", "max_depth", "learning_rate"], {"random_state": DEFAULT_RANDOM_STATE}))
        elif config.model_type == "xgboost":
            base_model = xgb.XGBClassifier(random_state=DEFAULT_RANDOM_STATE, use_label_encoder=False, eval_metric="logloss")
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10], "learning_rate": [0.01, 0.1, 0.2], "reg_alpha": [0, 0.1, 1], "reg_lambda": [1, 2, 5]}
            if hp: model = xgb.XGBClassifier(**_pick(["n_estimators", "max_depth", "learning_rate", "reg_alpha", "reg_lambda"], {"random_state": DEFAULT_RANDOM_STATE}))
        elif config.model_type == "lightgbm":
            base_model = lgb.LGBMClassifier(random_state=DEFAULT_RANDOM_STATE)
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [-1, 5, 10], "learning_rate": [0.01, 0.1, 0.2], "reg_alpha": [0, 0.1, 1], "reg_lambda": [0, 1, 5]}
            if hp: model = lgb.LGBMClassifier(**_pick(["n_estimators", "max_depth", "learning_rate", "reg_alpha", "reg_lambda"], {"random_state": DEFAULT_RANDOM_STATE}))
        elif config.model_type == "mlp":
            base_model = MLPClassifier(random_state=DEFAULT_RANDOM_STATE, max_iter=1000, early_stopping=True)
            param_grid = {"hidden_layer_sizes": [(50,), (100,), (100, 50)], "alpha": [0.0001, 0.001, 0.01, 0.1]}
            if hp: model = MLPClassifier(**_pick(["max_iter", "alpha", "early_stopping"], {"random_state": DEFAULT_RANDOM_STATE, "max_iter": 1000, "hidden_layer_sizes": (100,)}))
        elif config.model_type == "stacking":
            rf = RandomForestClassifier(n_estimators=100, random_state=DEFAULT_RANDOM_STATE)
            xgb_m = xgb.XGBClassifier(n_estimators=100, random_state=DEFAULT_RANDOM_STATE, use_label_encoder=False, eval_metric="logloss")
            lgb_m = lgb.LGBMClassifier(n_estimators=100, random_state=DEFAULT_RANDOM_STATE)
            base_model = StackingClassifier(estimators=[('rf', rf), ('xgb', xgb_m), ('lgb', lgb_m)], final_estimator=LogisticRegression())
            param_grid = {}
            if hp: model = base_model
        else:
            base_model = LogisticRegression(random_state=DEFAULT_RANDOM_STATE, max_iter=2000)
            param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
            if hp: model = LogisticRegression(**_pick(["max_iter", "C"], {"random_state": DEFAULT_RANDOM_STATE, "max_iter": 2000}))
    else:
        if config.model_type == "random_forest":
            base_model = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE)
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]}
            if hp: model = RandomForestRegressor(**_pick(["n_estimators", "max_depth", "min_samples_split"], {"random_state": DEFAULT_RANDOM_STATE}))
        elif config.model_type == "gradient_boosting":
            base_model = GradientBoostingRegressor(random_state=DEFAULT_RANDOM_STATE)
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10], "learning_rate": [0.01, 0.1, 0.2]}
            if hp: model = GradientBoostingRegressor(**_pick(["n_estimators", "max_depth", "learning_rate"], {"random_state": DEFAULT_RANDOM_STATE}))
        elif config.model_type == "xgboost":
            base_model = xgb.XGBRegressor(random_state=DEFAULT_RANDOM_STATE)
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 10], "learning_rate": [0.01, 0.1, 0.2], "reg_alpha": [0, 0.1, 1], "reg_lambda": [1, 2, 5]}
            if hp: model = xgb.XGBRegressor(**_pick(["n_estimators", "max_depth", "learning_rate", "reg_alpha", "reg_lambda"], {"random_state": DEFAULT_RANDOM_STATE}))
        elif config.model_type == "lightgbm":
            base_model = lgb.LGBMRegressor(random_state=DEFAULT_RANDOM_STATE)
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [-1, 5, 10], "learning_rate": [0.01, 0.1, 0.2], "reg_alpha": [0, 0.1, 1], "reg_lambda": [0, 1, 5]}
            if hp: model = lgb.LGBMRegressor(**_pick(["n_estimators", "max_depth", "learning_rate", "reg_alpha", "reg_lambda"], {"random_state": DEFAULT_RANDOM_STATE}))
        elif config.model_type == "mlp":
            base_model = MLPRegressor(random_state=DEFAULT_RANDOM_STATE, max_iter=1000, early_stopping=True)
            param_grid = {"hidden_layer_sizes": [(50,), (100,), (100, 50)], "alpha": [0.0001, 0.001, 0.01, 0.1]}
            if hp: model = MLPRegressor(**_pick(["max_iter", "alpha", "early_stopping"], {"random_state": DEFAULT_RANDOM_STATE, "max_iter": 1000, "hidden_layer_sizes": (100,)}))
        elif config.model_type == "stacking":
            rf = RandomForestRegressor(n_estimators=100, random_state=DEFAULT_RANDOM_STATE)
            xgb_m = xgb.XGBRegressor(n_estimators=100, random_state=DEFAULT_RANDOM_STATE)
            lgb_m = lgb.LGBMRegressor(n_estimators=100, random_state=DEFAULT_RANDOM_STATE)
            base_model = StackingRegressor(estimators=[('rf', rf), ('xgb', xgb_m), ('lgb', lgb_m)], final_estimator=Ridge())
            param_grid = {}
            if hp: model = base_model
        else: 
            actual_model_type = "linear_regression"
            base_model = Ridge(random_state=DEFAULT_RANDOM_STATE)
            param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
            if hp:
                params = _pick(["alpha"], {})
                model = Ridge(**params) if "alpha" in params else LinearRegression()

    if model is None:
        n_iter = 10 if len(X_train) > 1000 else 20
        search = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=n_iter, cv=3, random_state=DEFAULT_RANDOM_STATE, n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_
    else:
        model.fit(X_train, y_train)

    # 3. METRICS EVALUATION & CROSS-VALIDATION
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    metrics = {}
    
    cv_score = None
    if len(X_train) >= 20: # Ensure enough samples for CV
        if is_classification:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
            try:
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
                cv_score = scores.mean()
            except: pass
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
            try:
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
                cv_score = scores.mean()
            except: pass

    if is_classification:
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['train_accuracy'] = accuracy_score(y_train, y_pred_train)
        metrics['cv_score'] = cv_score
        metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')
        metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    else:
        metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_pred_train))
        metrics['cv_score'] = cv_score
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['r2'] = r2_score(y_test, y_pred)
        metrics['train_r2'] = r2_score(y_train, y_pred_train)

    metrics = {k: _sanitize_metric(v) for k, v in metrics.items()}

    # Save model artifact
    model_id = str(uuid.uuid4())
    artifact = {
        'model': model,
        'feature_names': X.columns.tolist(),
        'target_column': config.target_column,
        'is_classification': is_classification,
        'model_type': actual_model_type,
        'drop_columns': config.drop_columns,
        'transformers': transformers
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
    import copy
    
    baseline_model = train_model(config)
    if not config.smart_regularization:
        return baseline_model
        
    is_classification = baseline_model.task_type == "classification"
    metrics = baseline_model.metrics
    
    overfit = False
    if is_classification:
        train_acc = metrics.train_accuracy or 0
        test_acc = metrics.accuracy or 0
        if (train_acc - test_acc) > 0.05:
            overfit = True
    else:
        train_r2 = metrics.train_r2 or 0
        test_r2 = metrics.r2 or 0
        if (train_r2 - test_r2) > 0.05:
            overfit = True
            
    if not overfit:
        return baseline_model
        
    reg_config = copy.deepcopy(config)
    if reg_config.hyperparams is None:
        reg_config.hyperparams = {}
        
    mt = reg_config.model_type
    if mt == "random_forest":
        reg_config.hyperparams["max_depth"] = 5
        reg_config.hyperparams["min_samples_split"] = 10
    elif mt == "xgboost":
        reg_config.hyperparams["max_depth"] = 3
        reg_config.hyperparams["reg_lambda"] = 5.0
        reg_config.hyperparams["reg_alpha"] = 1.0
    elif mt == "lightgbm":
        reg_config.hyperparams["max_depth"] = 3
        reg_config.hyperparams["reg_lambda"] = 5.0
        reg_config.hyperparams["reg_alpha"] = 1.0
    elif mt == "mlp":
        reg_config.hyperparams["alpha"] = 0.05
        reg_config.hyperparams["early_stopping"] = True
    elif mt == "logistic_regression":
        if is_classification:
            reg_config.hyperparams["C"] = 0.01
        else:
            reg_config.hyperparams["alpha"] = 10.0
            
    reg_model = train_model(reg_config)
    
    if is_classification:
        reg_test = reg_model.metrics.accuracy or 0
        base_test = baseline_model.metrics.accuracy or 0
    else:
        reg_test = reg_model.metrics.r2 or -9999
        base_test = baseline_model.metrics.r2 or -9999
        
    if reg_test >= base_test:
        return reg_model
        
    return baseline_model
