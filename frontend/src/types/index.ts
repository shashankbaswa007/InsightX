/**
 * InsightX AI — Core Type Definitions
 */

// ─── Dataset Types ───────────────────────────────────────────────

export interface DatasetColumn {
  name: string;
  dtype: string;
  unique_count: number;
  null_count: number;
  sample_values: (string | number | null)[];
}

export interface DatasetMeta {
  id: string;
  filename: string;
  rows: number;
  columns: DatasetColumn[];
  uploaded_at: string;
}

// ─── Training Types ──────────────────────────────────────────────

export interface TrainingConfig {
  dataset_id: string;
  target_column: string;
  drop_columns: string[];
  model_type: "random_forest" | "gradient_boosting" | "logistic_regression" | "xgboost" | "lightgbm" | "mlp";
  test_size: number;
}

export interface ModelMetrics {
  accuracy?: number;
  train_accuracy?: number;
  f1_score?: number;
  precision?: number;
  recall?: number;
  rmse?: number;
  train_rmse?: number;
  mae?: number;
  r2?: number;
  confusion_matrix?: number[][];
  classification_report?: Record<string, Record<string, number>>;
}

export interface TrainedModel {
  id: string;
  dataset_id: string;
  model_type: string;
  target_column: string;
  feature_names: string[];
  task_type: "classification" | "regression";
  metrics: ModelMetrics;
  created_at: string;
}


// ─── XAI / Explanation Types ─────────────────────────────────────

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ShapValues {
  feature: string;
  shap_value: number;
  feature_value: number | string;
}

export interface LimeExplanation {
  feature: string;
  weight: number;
  condition: string;
}

export interface PredictionExplanation {
  row_index: number;
  prediction: number | string;
  probability?: number;
  shap_values: ShapValues[];
  lime_explanation: LimeExplanation[];
}

// ─── What-If Types ───────────────────────────────────────────────

export interface WhatIfFeature {
  name: string;
  value: number;
  min: number;
  max: number;
  step: number;
  dtype: "numeric" | "categorical";
  categories?: string[];
}

export interface WhatIfResult {
  prediction: number | string;
  probability?: number;
  feature_contributions: FeatureImportance[];
}

// ─── Bias / Fairness Types ───────────────────────────────────────

export interface BiasMetric {
  metric_name: string;
  overall: number;
  by_group: Record<string, number>;
  disparity_ratio: number;
}

export interface FairnessReport {
  protected_attribute: string;
  group_counts: Record<string, number>;
  metrics: BiasMetric[];
  flagged: boolean;
}

// ─── UI State Types ──────────────────────────────────────────────

export type AppPhase =
  | "upload"
  | "configure"
  | "training"
  | "results"
  | "explain"
  | "whatif"
  | "bias";

export interface AppState {
  phase: AppPhase;
  dataset: DatasetMeta | null;
  model: TrainedModel | null;
  isLoading: boolean;
  error: string | null;
}
