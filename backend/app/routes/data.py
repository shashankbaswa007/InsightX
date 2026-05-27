import os
import uuid
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import UPLOAD_DIR, MAX_UPLOAD_SIZE_MB
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

            # Enforce upload size limit
            max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
            if len(content) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File size ({len(content) / 1024 / 1024:.1f} MB) exceeds the {MAX_UPLOAD_SIZE_MB} MB limit."
                )

            buffer.write(content)
        
        # Load data to extract metadata
        if ext == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_json(file_path)

        columns_meta = []
        for col in df.columns:
            # Get up to 5 non-null, finite sample values
            import numpy as np
            import math
            samples = df[col].dropna().head(5).tolist()
            # Convert NaN/Inf to None and numpy types to Python types for JSON serialization
            clean_samples = []
            for x in samples:
                if isinstance(x, (float, np.floating)) and (math.isnan(float(x)) or math.isinf(float(x))):
                    clean_samples.append(None)
                elif isinstance(x, np.floating):
                    clean_samples.append(float(x))
                elif isinstance(x, np.integer):
                    clean_samples.append(int(x))
                else:
                    clean_samples.append(x)
            samples = clean_samples
            
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
    # Cap rows to a reasonable limit
    rows = max(1, min(rows, 500))

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

        # Replace NaN and Infinity with None for valid JSON serialization
        # Note: df.where(pd.notna(df), None) doesn't reliably work because Pandas
        # converts None back to NaN for float columns. We must sanitize at dict level.
        import numpy as np
        import math
        records = df.to_dict(orient="records")
        sanitized = []
        for row in records:
            clean_row = {}
            for k, v in row.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    clean_row[k] = None
                elif isinstance(v, np.floating):
                    fv = float(v)
                    clean_row[k] = None if (math.isnan(fv) or math.isinf(fv)) else fv
                elif isinstance(v, np.integer):
                    clean_row[k] = int(v)
                else:
                    clean_row[k] = v
            sanitized.append(clean_row)
        return sanitized
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")
