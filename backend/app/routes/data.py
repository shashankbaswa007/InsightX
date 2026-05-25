import os
import uuid
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import UPLOAD_DIR
from app.models.schemas import DatasetMeta, DatasetColumn

router = APIRouter(prefix="/api/data", tags=["Data"])

@router.post("/upload", response_model=DatasetMeta)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV or JSON file, save it to the uploads directory,
    and return metadata about the dataset.
    """
    if not file.filename.endswith(('.csv', '.json')):
        raise HTTPException(status_code=400, detail="Only .csv and .json files are supported")

    dataset_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{dataset_id}{ext}")

    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load data to extract metadata
        if ext == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_json(file_path)

        columns_meta = []
        for col in df.columns:
            # Get up to 5 non-null sample values
            samples = df[col].dropna().head(5).tolist()
            # Convert NaN to None for JSON serialization
            samples = [x if pd.notna(x) else None for x in samples]
            
            col_meta = DatasetColumn(
                name=str(col),
                dtype=str(df[col].dtype),
                unique_count=df[col].nunique(),
                null_count=int(df[col].isnull().sum()),
                sample_values=samples
            )
            columns_meta.append(col_meta)

        meta = DatasetMeta(
            id=dataset_id,
            filename=file.filename,
            rows=len(df),
            columns=columns_meta,
            uploaded_at=datetime.utcnow().isoformat()
        )
        return meta

    except Exception as e:
        # Clean up file if there was an error parsing it
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/{dataset_id}/preview")
async def get_dataset_preview(dataset_id: str, rows: int = 10):
    """
    Get a preview of the dataset rows.
    """
    # Find the file
    file_path = None
    for ext in ['.csv', '.json']:
        potential_path = os.path.join(UPLOAD_DIR, f"{dataset_id}{ext}")
        if os.path.exists(potential_path):
            file_path = potential_path
            break
            
    if not file_path:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, nrows=rows)
        else:
            df = pd.read_json(file_path).head(rows)
            
        # Replace NaNs with None for valid JSON
        df = df.where(pd.notna(df), None)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")
