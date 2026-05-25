import axios from "axios";

/**
 * Axios client configured for the FastAPI ML backend.
 * All ML-related API calls go through this client.
 */
const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  timeout: 120000, // 2 minutes — model training can be slow
  headers: {
    "Content-Type": "application/json",
  },
});

// Response interceptor for consistent error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.detail ||
      error.message ||
      "An unexpected error occurred";
    console.error("[InsightX API Error]:", message);
    return Promise.reject(new Error(message));
  }
);

export default api;

// ─── API Wrapper Functions ──────────────────────────────────────────

import type { 
  DatasetMeta, 
  TrainingConfig, 
  TrainedModel, 
  FeatureImportance, 
  PredictionExplanation 
} from "@/types";

export const uploadDataset = async (file: File): Promise<DatasetMeta> => {
  const formData = new FormData();
  formData.append("file", file);
  
  const response = await api.post<DatasetMeta>("/api/data/upload", formData);
  return response.data;
};

export const getDatasetPreview = async (datasetId: string, rows: number = 10): Promise<any[]> => {
  const response = await api.get<any[]>(`/api/data/${datasetId}/preview?rows=${rows}`);
  return response.data;
};

export const trainModel = async (config: TrainingConfig): Promise<TrainedModel> => {
  const response = await api.post<TrainedModel>("/api/train/", config);
  return response.data;
};

export const getGlobalExplanation = async (modelId: string): Promise<FeatureImportance[]> => {
  const response = await api.get<FeatureImportance[]>(`/api/explain/global/${modelId}`);
  return response.data;
};

export const getLocalExplanation = async (
  modelId: string, 
  datasetId: string, 
  rowIndex: number
): Promise<PredictionExplanation> => {
  const response = await api.get<PredictionExplanation>(`/api/explain/local/${modelId}/${datasetId}/${rowIndex}`);
  return response.data;
};
