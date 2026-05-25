import os
import joblib
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
from typing import List, Dict, Any

from app.config import MODEL_DIR
from app.models.schemas import FeatureImportance, ShapValues, LimeExplanation, PredictionExplanation
from app.services.ml_service import load_dataset, preprocess_data

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
    df = load_dataset(dataset_id)
    X, y, _ = preprocess_data(df, target_column, []) # We need the same drop_columns ideally, assuming empty for now
    
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
            # shap_values_raw structure depends on scikit-learn version and task type.
            # For RF classifier, it's a list of arrays (one per class). We take the one for the predicted class.
            if is_classification:
                if isinstance(shap_values_raw, list):
                    shap_vals = shap_values_raw[prediction][0]
                elif len(shap_values_raw.shape) == 3:
                    # Shape: (num_samples, num_features, num_classes)
                    shap_vals = shap_values_raw[0, :, prediction]
                else:
                    shap_vals = shap_values_raw[0]
            else:
                shap_vals = shap_values_raw[0] if len(shap_values_raw.shape) > 1 else shap_values_raw
        else:
            # Fallback to KernelExplainer (can be slow, using small background)
            explainer = shap.KernelExplainer(model.predict, X_background)
            shap_vals = explainer.shap_values(X_row, l1_reg="num_features(10)")[0]
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[prediction]
                
        shap_res = [
            ShapValues(
                feature=feature_names[i], 
                shap_value=float(np.squeeze(shap_vals[i])), 
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
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_background.values, 
                feature_names=feature_names, 
                class_names=['0', '1'], # simplified
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
                feature=feature_names[0], # The feature name is embedded in the condition, we extract simplified
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
