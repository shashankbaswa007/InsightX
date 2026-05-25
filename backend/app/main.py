"""
InsightX AI - FastAPI ML Backend
Main entry point for the ML engine server.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import data, train, explain

app = FastAPI(
    title="InsightX AI - ML Engine",
    description="Explainable AI backend for model training, SHAP/LIME explanations, and bias detection.",
    version="1.0.0",
)

# CORS — allow the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router)
app.include_router(train.router)
app.include_router(explain.router)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "InsightX ML Engine"}
