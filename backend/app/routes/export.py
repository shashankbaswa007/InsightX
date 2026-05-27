"""
Export endpoints — download model artifacts and generate usage code snippets.
"""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import MODEL_DIR
import joblib

router = APIRouter(prefix="/api/export", tags=["Export"])


@router.get("/download/{model_id}")
async def download_model(model_id: str):
    """Download the trained model as a .joblib file."""
    file_path = os.path.join(MODEL_DIR, f"{model_id}.joblib")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model not found")

    return FileResponse(
        path=file_path,
        filename=f"insightx_model_{model_id[:8]}.joblib",
        media_type="application/octet-stream",
    )


@router.get("/snippet/{model_id}")
async def get_code_snippet(model_id: str):
    """Generate Python and cURL code snippets for using the exported model."""
    file_path = os.path.join(MODEL_DIR, f"{model_id}.joblib")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model not found")

    artifact = joblib.load(file_path)
    feature_names = artifact.get("feature_names", [])
    target_column = artifact.get("target_column", "target")
    model_type = artifact.get("model_type", "unknown")
    is_classification = artifact.get("is_classification", True)

    python_snippet = f'''import joblib
import pandas as pd

# Load the trained model
artifact = joblib.load("insightx_model_{model_id[:8]}.joblib")
model = artifact["model"]
feature_names = artifact["feature_names"]

# Prepare your input data (must match training features)
# Example: single row prediction
input_data = pd.DataFrame({{
{chr(10).join(f'    "{f}": [0],' for f in feature_names[:5])}
    # ... add all {len(feature_names)} features
}})

# One-hot encode categorical columns if needed
# input_data = pd.get_dummies(input_data)
# input_data = input_data.reindex(columns=feature_names, fill_value=0)

prediction = model.predict(input_data)
print(f"Prediction: {{prediction[0]}}")
{"" if not is_classification else """
# For classification, get probabilities
if hasattr(model, 'predict_proba'):
    proba = model.predict_proba(input_data)
    print(f"Probabilities: {proba[0]}")
"""}'''

    curl_snippet = f'''# Download the model
curl -O http://localhost:8000/api/export/download/{model_id}

# Use the What-If API for live predictions
curl -X POST http://localhost:8000/api/whatif/predict/{model_id} \\
  -H "Content-Type: application/json" \\
  -d '{{
    "dataset_id": "<your_dataset_id>",
    "row_index": 0,
    "modified_features": {{}}
  }}'
'''

    return {
        "model_id": model_id,
        "model_type": model_type,
        "target_column": target_column,
        "is_classification": is_classification,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "python": python_snippet,
        "curl": curl_snippet,
    }
