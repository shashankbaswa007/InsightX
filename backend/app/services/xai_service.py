import os
import joblib
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import re
from typing import List, Dict, Any

from app.config import MODEL_DIR
from app.models.schemas import FeatureImportance, ShapValues, LimeExplanation, PredictionExplanation
from app.services.ml_service import load_dataset, preprocess_data, apply_transformers


def _normalize_shap_values(shap_values_raw, is_classification: bool, prediction: int, n_features: int) -> np.ndarray:
    """
    Normalize SHAP output into a flat 1D array of shape (n_features,).
    Handles all known output formats from different SHAP explainers and sklearn versions:
      - list of arrays (one per class, common for TreeExplainer + classification)
      - 3D ndarray of shape (n_samples, n_features, n_classes)
      - 2D ndarray of shape (n_samples, n_features)
      - shap.Explanation object (newer SHAP versions)
    """
    # Handle shap.Explanation objects (newer SHAP API)
    if hasattr(shap_values_raw, 'values'):
        shap_values_raw = shap_values_raw.values

    if isinstance(shap_values_raw, list):
        # List of arrays: one per class for classification
        if is_classification and prediction < len(shap_values_raw):
            vals = np.array(shap_values_raw[prediction])
        else:
            vals = np.array(shap_values_raw[0])
        # Flatten to 1D: could be (1, n_features) or (n_features,)
        return vals.flatten()[:n_features]

    arr = np.array(shap_values_raw)

    if arr.ndim == 3:
        # Shape: (n_samples, n_features, n_classes)
        if is_classification and prediction < arr.shape[2]:
            return arr[0, :, prediction].flatten()[:n_features]
        return arr[0, :, 0].flatten()[:n_features]

    if arr.ndim == 2:
        # Shape: (n_samples, n_features)
        return arr[0].flatten()[:n_features]

    if arr.ndim == 1:
        return arr[:n_features]

    # Absolute fallback
    return np.zeros(n_features)


def _extract_lime_feature_name(condition: str, feature_names: List[str]) -> str:
    """
    Parse the actual feature name from a LIME condition string.
    LIME conditions look like: 'feature_name > 0.45' or '0.10 < feature_name <= 0.50'
    We match against known feature names (longest match first to avoid partial matches).
    """
    # Sort by length descending to match longer names first (e.g., "feature_10" before "feature_1")
    for name in sorted(feature_names, key=len, reverse=True):
        if name in condition:
            return name
    # Fallback: return the condition itself as the feature name
    return condition.split('<')[0].split('>')[0].split('=')[0].strip()


def load_model_artifact(model_id: str) -> Dict[str, Any]:
    file_path = os.path.join(MODEL_DIR, f"{model_id}.joblib")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model {model_id} not found.")
    return joblib.load(file_path)

def get_global_explanations(model_id: str) -> List[FeatureImportance]:
    """
    Computes global feature importance using SHAP.
    """
    artifact = load_model_artifact(model_id)
    model = artifact['model']
    feature_names = artifact['feature_names']
    
    # Normally, computing SHAP on the whole dataset is expensive,
    # so in a real system we'd compute this during training or on a background thread.
    # We will use the model's built-in feature importances if available (e.g. Random Forest, GBM)
    # as a fast proxy for global SHAP summary, or compute SHAP on a background sample.
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
    else:
        # Fallback to equal importance if unknown
        importances = np.ones(len(feature_names)) / len(feature_names)
        
    # Create FeatureImportance objects and sort descending
    fi_list = [
        FeatureImportance(feature=name, importance=float(imp))
        for name, imp in zip(feature_names, importances)
    ]
    fi_list.sort(key=lambda x: x.importance, reverse=True)
    
    return fi_list

def get_local_explanation(model_id: str, dataset_id: str, row_index: int) -> PredictionExplanation:
    """
    Computes local explanation (SHAP + LIME) for a specific prediction row.
    """
    artifact = load_model_artifact(model_id)
    model = artifact['model']
    feature_names = artifact['feature_names']
    target_column = artifact['target_column']
    is_classification = artifact['is_classification']
    
    # Load and preprocess data to get the exact feature vector
    # Use the same drop_columns from training to ensure feature alignment
    drop_columns = artifact.get('drop_columns', [])
    df = load_dataset(dataset_id)
    X, y, _ = preprocess_data(df, target_column, drop_columns)
    
    # APPLY TRANSFORMATIONS SO X MATCHES WHAT THE MODEL WAS TRAINED ON
    transformers = artifact.get('transformers', {})
    X = apply_transformers(X, transformers)
    
    if row_index < 0 or row_index >= len(X):
        raise ValueError("row_index out of bounds")
        
    # Extract the single row
    X_row = X.iloc[[row_index]]
    X_background = X.sample(n=min(100, len(X)), random_state=42) # Background dataset for SHAP/LIME
    
    # 1. Prediction
    if is_classification:
        prediction = int(model.predict(X_row)[0])
        probability = float(model.predict_proba(X_row)[0][prediction]) if hasattr(model, 'predict_proba') else None
    else:
        prediction = float(model.predict(X_row)[0])
        probability = None
        
    # 2. SHAP Values
    # Use TreeExplainer for trees, LinearExplainer for linear models
    model_type = artifact.get('model_type', '')
    try:
        if model_type in ['random_forest', 'gradient_boosting']:
            explainer = shap.TreeExplainer(model)
            shap_values_raw = explainer.shap_values(X_row)
        else:
            # Fallback to KernelExplainer (can be slow, using small background)
            explainer = shap.KernelExplainer(model.predict, X_background)
            shap_values_raw = explainer.shap_values(X_row)

        # Normalize SHAP output to a 1D array of shape (n_features,)
        shap_vals = _normalize_shap_values(shap_values_raw, is_classification, prediction, len(feature_names))
                
        shap_res = [
            ShapValues(
                feature=feature_names[i], 
                shap_value=float(shap_vals[i]), 
                feature_value=float(X_row.iloc[0, i])
            )
            for i in range(len(feature_names))
        ]
        shap_res.sort(key=lambda x: abs(x.shap_value), reverse=True)
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        shap_res = []

    # 3. LIME Explanation
    try:
        if is_classification:
            # Use dynamic class names from the model
            class_names = [str(c) for c in model.classes_] if hasattr(model, 'classes_') else [str(i) for i in range(max(2, int(prediction) + 1))]
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_background.values, 
                feature_names=feature_names, 
                class_names=class_names,
                mode='classification'
            )
            lime_exp = lime_explainer.explain_instance(X_row.values[0], model.predict_proba, num_features=10)
        else:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_background.values, 
                feature_names=feature_names, 
                mode='regression'
            )
            lime_exp = lime_explainer.explain_instance(X_row.values[0], model.predict, num_features=10)
            
        lime_res = [
            LimeExplanation(
                feature=_extract_lime_feature_name(condition, feature_names),
                weight=float(weight),
                condition=condition
            )
            for condition, weight in lime_exp.as_list()
        ]
    except Exception as e:
        print(f"LIME explanation failed: {e}")
        lime_res = []

    return PredictionExplanation(
        row_index=row_index,
        prediction=prediction,
        probability=probability,
        shap_values=shap_res,
        lime_explanation=lime_res
    )
