from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import joblib
import pandas as pd

from app.config import MODEL_DIR
from app.services.ml_service import load_dataset, preprocess_data, apply_transformers

router = APIRouter(prefix="/api/bias", tags=["Bias Detection"])

class BiasRequest(BaseModel):
    dataset_id: str
    protected_attribute: str

@router.post("/analyze/{model_id}")
async def analyze_bias(model_id: str, request: BiasRequest):
    """
    Computes fairness metrics based on a protected attribute.
    Currently returns Selection Rate per group for classification models.
    """
    try:
        artifact_path = os.path.join(MODEL_DIR, f"{model_id}.joblib")
        if not os.path.exists(artifact_path):
            raise FileNotFoundError("Model artifact not found.")
            
        artifact = joblib.load(artifact_path)
        model = artifact["model"]
        is_classification = artifact.get("is_classification", False)
        
        if not is_classification:
            raise ValueError("Bias detection is currently only supported for classification models.")
            
        df = load_dataset(request.dataset_id)
        
        if request.protected_attribute not in df.columns:
            raise ValueError(f"Protected attribute '{request.protected_attribute}' not found in dataset.")
            
        # We need the original protected attribute values for grouping
        protected_values = df[request.protected_attribute].copy()
        
        X, y, _ = preprocess_data(df, artifact["target_column"], artifact["drop_columns"])
        transformers = artifact.get("transformers", {})
        X = apply_transformers(X, transformers)
        
        # Align indices
        protected_values = protected_values.loc[X.index]
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate Selection Rate for each group
        # Selection Rate = (predicted positive) / (total in group)
        # We assume the positive class is the max value (e.g. 1 in binary classification)
        positive_class = max(y_pred)
        
        results = []
        unique_groups = protected_values.dropna().unique()
        
        for group in unique_groups:
            group_mask = (protected_values == group)
            total_in_group = group_mask.sum()
            
            if total_in_group == 0:
                continue
                
            group_preds = y_pred[group_mask]
            positive_preds = sum(group_preds == positive_class)
            selection_rate = positive_preds / total_in_group
            
            results.append({
                "group": str(group),
                "total": int(total_in_group),
                "positive_predictions": int(positive_preds),
                "selection_rate": float(selection_rate)
            })
            
        return {
            "protected_attribute": request.protected_attribute,
            "positive_class": str(positive_class),
            "metrics": results
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
