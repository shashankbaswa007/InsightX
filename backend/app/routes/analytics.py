from fastapi import APIRouter, HTTPException
import os
import joblib
import pandas as pd
import numpy as np

from app.config import MODEL_DIR
from app.services.ml_service import load_dataset, preprocess_data, apply_transformers
from app.models.schemas import TrainedModel

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])

@router.get("/correlation/{dataset_id}")
async def get_correlation_matrix(dataset_id: str):
    """
    Computes Pearson correlation matrix for numeric columns in the dataset.
    """
    try:
        df = load_dataset(dataset_id)
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {"features": [], "matrix": []}
            
        corr_matrix = numeric_df.corr().replace({np.nan: 0})
        features = corr_matrix.columns.tolist()
        
        # Flatten matrix for chart
        data = []
        for i, f1 in enumerate(features):
            for j, f2 in enumerate(features):
                data.append({
                    "x": f1,
                    "y": f2,
                    "value": round(corr_matrix.iloc[i, j], 3)
                })
                
        return {
            "features": features,
            "data": data
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/confidence/{model_id}/{dataset_id}")
async def get_confidence_distribution(model_id: str, dataset_id: str):
    """
    Computes prediction probability distribution on the dataset.
    Only applicable for classification models.
    """
    try:
        artifact_path = os.path.join(MODEL_DIR, f"{model_id}.joblib")
        if not os.path.exists(artifact_path):
            raise FileNotFoundError("Model artifact not found.")
            
        artifact = joblib.load(artifact_path)
        model = artifact["model"]
        is_classification = artifact.get("is_classification", False)
        
        if not is_classification:
            return {"distribution": [], "message": "Confidence distribution is only applicable to classification models."}
            
        df = load_dataset(dataset_id)
        X, _, _ = preprocess_data(df, artifact["target_column"], artifact["drop_columns"])
        transformers = artifact.get("transformers", {})
        X = apply_transformers(X, transformers)
        
        if not hasattr(model, "predict_proba"):
             return {"distribution": [], "message": "Model does not support predict_proba."}
             
        probas = model.predict_proba(X)
        max_probas = np.max(probas, axis=1)
        
        # Create bins for distribution
        bins = np.linspace(0.0, 1.0, 21) # 20 bins
        hist, _ = np.histogram(max_probas, bins=bins)
        
        distribution = []
        for i in range(len(hist)):
            distribution.append({
                "bin_start": round(bins[i], 2),
                "bin_end": round(bins[i+1], 2),
                "count": int(hist[i])
            })
            
        return {"distribution": distribution}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
