from fastapi import APIRouter, HTTPException
from app.models.schemas import TrainingConfig, TrainedModel
from app.services.ml_service import train_model

router = APIRouter(prefix="/api/train", tags=["Training"])

@router.post("/", response_model=TrainedModel)
async def start_training(config: TrainingConfig):
    """
    Train a machine learning model based on the uploaded dataset and configuration.
    """
    try:
        # In a real enterprise app, this would be dispatched to a background worker (e.g. Celery).
        # For this prototype/MVP, we run it synchronously.
        trained_model_meta = train_model(config)
        return trained_model_meta
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")
