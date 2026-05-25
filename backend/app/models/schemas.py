from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

# ─── Dataset Models ───────────────────────────────────────────────

class DatasetColumn(BaseModel):
    name: str
    dtype: str
    unique_count: int
    null_count: int
    sample_values: List[Any]

class DatasetMeta(BaseModel):
    id: str
    filename: str
    rows: int
    columns: List[DatasetColumn]
    uploaded_at: str

# ─── Training Models ──────────────────────────────────────────────

class TrainingConfig(BaseModel):
    dataset_id: str
    target_column: str
    drop_columns: List[str] = Field(default_factory=list)
    model_type: str = Field(..., description="random_forest, gradient_boosting, or logistic_regression")
    test_size: float = Field(0.2, ge=0.05, le=0.5)

class ModelMetrics(BaseModel):
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict[str, Any]] = None

class TrainedModel(BaseModel):
    id: str
    dataset_id: str
    model_type: str
    target_column: str
    feature_names: List[str]
    task_type: str  # "classification" or "regression"
    metrics: ModelMetrics
    created_at: str

# ─── XAI Models ───────────────────────────────────────────────────

class FeatureImportance(BaseModel):
    feature: str
    importance: float

class ShapValues(BaseModel):
    feature: str
    shap_value: float
    feature_value: float

class LimeExplanation(BaseModel):
    feature: str
    weight: float
    condition: str

class PredictionExplanation(BaseModel):
    row_index: int
    prediction: float
    probability: Optional[float] = None
    shap_values: List[ShapValues]
    lime_explanation: List[LimeExplanation]
