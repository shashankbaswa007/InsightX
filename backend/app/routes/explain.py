from fastapi import APIRouter, HTTPException
from typing import List
from app.models.schemas import FeatureImportance, PredictionExplanation
from app.services.xai_service import get_global_explanations, get_local_explanation

router = APIRouter(prefix="/api/explain", tags=["Explanations"])

@router.get("/global/{model_id}", response_model=List[FeatureImportance])
async def global_explanations(model_id: str):
    """
    Get global feature importance for a trained model.
    """
    try:
        return get_global_explanations(model_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting global explanations: {str(e)}")

@router.get("/local/{model_id}/{dataset_id}/{row_index}", response_model=PredictionExplanation)
async def local_explanation(model_id: str, dataset_id: str, row_index: int):
    """
    Get local explanation (SHAP + LIME) for a specific row in the dataset.
    """
    try:
        return get_local_explanation(model_id, dataset_id, row_index)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting local explanation: {str(e)}")
