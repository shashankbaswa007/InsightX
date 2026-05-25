import os
import uuid
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

    # Drop requested columns
    df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors='ignore')

    # Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Determine task type based on target dtype and unique values
    is_classification = False
    if y.dtype == 'object' or y.nunique() < 20: # Heuristic for classification
        is_classification = True
        if y.dtype == 'object':
            # Encode target if it's categorical strings
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=target_column)

    # Impute missing values (simple strategy for baseline)
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

def train_model(config: TrainingConfig) -> TrainedModel:
    """
    Trains a model based on the configuration and saves it.
    """
    df = load_dataset(config.dataset_id)
    X, y, is_classification = preprocess_data(df, config.target_column, config.drop_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=DEFAULT_RANDOM_STATE
    )

    # Select model
    if is_classification:
        if config.model_type == "random_forest":
            model = RandomForestClassifier(random_state=DEFAULT_RANDOM_STATE)
        elif config.model_type == "gradient_boosting":
            model = GradientBoostingClassifier(random_state=DEFAULT_RANDOM_STATE)
        elif config.model_type == "logistic_regression":
            model = LogisticRegression(random_state=DEFAULT_RANDOM_STATE, max_iter=1000)
        else:
            # Default fallback
            model = RandomForestClassifier(random_state=DEFAULT_RANDOM_STATE)
            config.model_type = "random_forest"
    else:
        if config.model_type == "random_forest":
            model = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE)
        elif config.model_type == "gradient_boosting":
            model = GradientBoostingRegressor(random_state=DEFAULT_RANDOM_STATE)
        elif config.model_type == "logistic_regression":
            # Linear regression instead of logistic for regression tasks
            model = LinearRegression()
            config.model_type = "linear_regression"
        else:
            model = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE)
            config.model_type = "random_forest"

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {}
    
    if is_classification:
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')
        metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        # classification_report returns a string by default, output_dict=True returns dict
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    else:
        metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['r2'] = r2_score(y_test, y_pred)

    # Save model artifact and feature names (critical for SHAP/LIME later)
    model_id = str(uuid.uuid4())
    artifact = {
        'model': model,
        'feature_names': X.columns.tolist(),
        'target_column': config.target_column,
        'is_classification': is_classification,
        'model_type': config.model_type
    }
    
    joblib.dump(artifact, os.path.join(MODEL_DIR, f"{model_id}.joblib"))

    return TrainedModel(
        id=model_id,
        dataset_id=config.dataset_id,
        model_type=config.model_type,
        target_column=config.target_column,
        feature_names=X.columns.tolist(),
        task_type="classification" if is_classification else "regression",
        metrics=ModelMetrics(**metrics),
        created_at=datetime.utcnow().isoformat()
    )
