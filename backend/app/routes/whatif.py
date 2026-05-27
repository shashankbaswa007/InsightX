from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import os
import joblib
import pandas as pd
import numpy as np

from app.config import MODEL_DIR
from app.services.ml_service import load_dataset, preprocess_data

router = APIRouter(prefix="/api/whatif", tags=["What-If Analysis"])

class WhatIfRequest(BaseModel):
    dataset_id: str
    row_index: int
    modified_features: Dict[str, Any]

@router.post("/predict/{model_id}")
async def whatif_predict(model_id: str, request: WhatIfRequest):
    """
    Predicts the outcome of a specific row after applying modifications.
    """
    try:
        artifact_path = os.path.join(MODEL_DIR, f"{model_id}.joblib")
        if not os.path.exists(artifact_path):
            raise FileNotFoundError("Model artifact not found.")
            
        artifact = joblib.load(artifact_path)
        model = artifact["model"]
        
        # Create a single-row DataFrame from the modified features
        row_X = pd.DataFrame([request.modified_features])
        
        # Drop target column and dropped columns
        cols_to_drop = [artifact["target_column"]] + artifact.get("drop_columns", [])
        row_X = row_X.drop(columns=[c for c in cols_to_drop if c in row_X.columns], errors="ignore")
        
        # Handle missing values if user cleared an input
        row_X = row_X.fillna(0)
        
        # One-hot encode categorical features (if any)
        cat_cols = row_X.select_dtypes(exclude=[np.number]).columns
        if len(cat_cols) > 0:
            row_X = pd.get_dummies(row_X, columns=cat_cols)
            
        # Ensure the features perfectly match the model's expected features
        # (pd.get_dummies might drop/create columns if a user inputs a novel categorical value)
        row_X = row_X.reindex(columns=artifact["feature_names"], fill_value=0)
        
        prediction = model.predict(row_X)[0].item()
        
        result = {"prediction": prediction}
        
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(row_X)[0]
            result["probabilities"] = probas.tolist()
            
        return result
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
