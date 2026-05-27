"""
AutoML endpoint — trains all 3 algorithms on the same split and returns a leaderboard.
"""

import os
import uuid
import time
import math
import numpy as np
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)
import joblib

from app.config import MODEL_DIR, DEFAULT_RANDOM_STATE
from app.services.ml_service import load_dataset, preprocess_data, _sanitize_metric
from app.models.schemas import TrainedModel, ModelMetrics

router = APIRouter(prefix="/api/train", tags=["AutoML"])


class AutoMLConfig(BaseModel):
    dataset_id: str
    target_column: str
    drop_columns: List[str] = Field(default_factory=list)
    test_size: float = Field(0.2, ge=0.05, le=0.5)


from app.services.ml_service import smart_train_model
from app.models.schemas import TrainingConfig

@router.post("/automl", response_model=List[TrainedModel])
async def train_automl(config: AutoMLConfig):
    """
    Train all 5 algorithms using the smart anti-overfitting pipeline.
    Returns a list of TrainedModel sorted by primary metric (accuracy or R²).
    """
    try:
        models_to_test = ["random_forest", "logistic_regression", "xgboost", "lightgbm", "mlp", "stacking"]
        results: List[TrainedModel] = []
        
        # We need to know if it's classification to sort correctly
        is_classification = None
        
        for mt in models_to_test:
            # Create a TrainingConfig for this specific model
            t_config = TrainingConfig(
                dataset_id=config.dataset_id,
                target_column=config.target_column,
                drop_columns=config.drop_columns,
                test_size=config.test_size,
                model_type=mt,
                smart_regularization=True
            )
            
            # This calls the smart loop which detects overfitting and retrains if needed
            trained_model = smart_train_model(t_config)
            results.append(trained_model)
            
            if is_classification is None:
                is_classification = trained_model.task_type == "classification"

        # Sort: classification by accuracy desc, regression by R² desc
        if is_classification:
            results.sort(key=lambda m: m.metrics.accuracy or 0, reverse=True)
        else:
            results.sort(key=lambda m: m.metrics.r2 or 0, reverse=True)

        return results

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AutoML error: {str(e)}")
