"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Lightbulb, Database, X, Activity, BarChart3, PieChart } from "lucide-react";
import AppShell from "@/components/layout/app-shell";
import { useAppStore } from "@/lib/store";
import { getGlobalExplanation, getDatasetPreview, getLocalExplanation, getCorrelationHeatmap, getConfidenceDistribution } from "@/lib/api";
import type { FeatureImportance, PredictionExplanation } from "@/types";
import { FeatureImportanceChart } from "@/components/charts/feature-importance-chart";
import { CorrelationHeatmap } from "@/components/charts/correlation-heatmap";
import { ConfusionMatrix } from "@/components/charts/confusion-matrix";
import { ConfidenceDistribution } from "@/components/charts/confidence-distribution";
import { DataTable } from "@/components/dashboard/data-table";
import { PredictionExplanationView } from "@/components/dashboard/prediction-explanation";
import { ModelInsightsCard } from "@/components/dashboard/model-insights";
import { Skeleton } from "@/components/ui/skeleton";

export default function ExplanationsPage() {
  const router = useRouter();
  const { dataset, model } = useAppStore();

  const [globalData, setGlobalData] = useState<FeatureImportance[] | null>(null);
  const [tableData, setTableData] = useState<any[]>([]);
  const [heatmapData, setHeatmapData] = useState<any | null>(null);
  const [confidenceData, setConfidenceData] = useState<any[] | null>(null);
  
  const [activeRowIndex, setActiveRowIndex] = useState<number | undefined>();
  const [localExplanation, setLocalExplanation] = useState<PredictionExplanation | null>(null);
  
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingLocal, setIsLoadingLocal] = useState(false);

  useEffect(() => {
    if (!dataset || !model) {
      router.push("/upload");
      return;
    }

    const loadInitialData = async () => {
      try {
        setIsLoading(true);
        const promises = [
          getGlobalExplanation(model.id),
          getDatasetPreview(dataset.id, 50),
          getCorrelationHeatmap(dataset.id)
        ];
        
        if (model.task_type === "classification") {
            promises.push(getConfidenceDistribution(model.id, dataset.id));
        }

        const results = await Promise.all(promises);
        
        setGlobalData(results[0]);
        setTableData(results[1]);
        setHeatmapData(results[2]);
        
        if (model.task_type === "classification" && results[3]?.distribution) {
            setConfidenceData(results[3].distribution);
        }
      } catch (err) {
        console.error("Failed to load explanation data", err);
      } finally {
        setIsLoading(false);
      }
    };

    loadInitialData();
  }, [dataset, model, router]);

  const handleRowClick = async (rowIndex: number) => {
    if (!dataset || !model) return;
    
    if (activeRowIndex === rowIndex) {
      setActiveRowIndex(undefined);
      setLocalExplanation(null);
      return;
    }

    try {
      setIsLoadingLocal(true);
      setActiveRowIndex(rowIndex);
      const localRes = await getLocalExplanation(model.id, dataset.id, rowIndex);
      setLocalExplanation(localRes);
    } catch (err) {
      console.error("Failed to fetch local explanation", err);
    } finally {
      setIsLoadingLocal(false);
    }
  };

  if (!dataset || !model) return null;

  return (
    <AppShell>
      <div className="mx-auto max-w-[1400px] space-y-8 pb-16">
        
        {/* Header */}
        <div className="flex justify-between items-end">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
              <Lightbulb className="h-8 w-8 text-primary" />
              Model Analytics & Explanations
            </h1>
            <p className="mt-2 text-zinc-400 max-w-2xl">
              Deep dive into how your {model.model_type.replace('_', ' ')} model operates. 
              Understand feature impacts, correlations, confidence distributions, and raw predictions.
            </p>
          </div>
        </div>

        {/* AI Training Insights Card */}
        {isLoading ? (
          <Skeleton className="w-full h-48 rounded-2xl mb-8" />
        ) : (
          <ModelInsightsCard model={model} globalData={globalData} />
        )}

        {/* Top Grid: Feature Importance & Heatmap */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <div className="h-full">
            {isLoading ? (
              <Skeleton className="w-full h-[450px] rounded-xl" />
            ) : globalData ? (
              <FeatureImportanceChart data={globalData} />
            ) : null}
          </div>

          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="rounded-xl border border-border/50 bg-card/40 backdrop-blur-md p-6 shadow-sm flex flex-col h-full min-h-[450px]"
          >
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <Activity className="h-5 w-5 text-indigo-400" />
                Dataset Feature Correlation
              </h3>
              <p className="text-sm text-zinc-400">
                Pearson correlation matrix indicating linear relationships between numeric features.
              </p>
            </div>
            <div className="flex-1 overflow-auto custom-scrollbar flex items-center justify-center">
              {isLoading ? (
                <Skeleton className="w-full h-full rounded-md" />
              ) : heatmapData ? (
                <CorrelationHeatmap data={heatmapData} />
              ) : null}
            </div>
          </motion.div>
        </div>

        {/* Middle Grid: Confusion Matrix & Confidence */}
        {model.task_type === "classification" && (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            {model.metrics?.confusion_matrix && (
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="rounded-xl border border-border/50 bg-card/40 backdrop-blur-md p-6 shadow-sm"
              >
                <div className="mb-4">
                  <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                    <PieChart className="h-5 w-5 text-emerald-400" />
                    Confusion Matrix
                  </h3>
                  <p className="text-sm text-zinc-400">
                    Detailed breakdown of correct vs incorrect classifications across the test set.
                  </p>
                </div>
                <div className="flex justify-center bg-black/20 rounded-lg py-4">
                   <ConfusionMatrix 
                     matrix={model.metrics.confusion_matrix} 
                   />
                </div>
              </motion.div>
            )}

            {confidenceData && confidenceData.length > 0 && (
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
                className="rounded-xl border border-border/50 bg-card/40 backdrop-blur-md p-6 shadow-sm"
              >
                <div className="mb-4">
                  <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                    <BarChart3 className="h-5 w-5 text-violet-400" />
                    Prediction Confidence
                  </h3>
                  <p className="text-sm text-zinc-400">
                    Distribution of the model's confidence scores across the dataset.
                  </p>
                </div>
                <div className="pt-4 bg-black/20 rounded-lg p-4">
                  <ConfidenceDistribution data={confidenceData} />
                </div>
              </motion.div>
            )}
          </div>
        )}

        {/* Data Table & Local Explanations */}
        <div className="space-y-4 pt-4">
          <div className="flex items-center gap-2 border-b border-zinc-800 pb-2">
            <Database className="h-5 w-5 text-indigo-400" />
            <h2 className="text-xl font-semibold text-white">Dataset & Local Explanations</h2>
          </div>
          <p className="text-sm text-zinc-400">
            Select a specific row from the dataset below to see a detailed SHAP and LIME breakdown of its prediction.
          </p>

          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 items-start">
            
            {/* Table */}
            <div className={`transition-all duration-500 ease-in-out ${localExplanation ? 'lg:col-span-2' : 'lg:col-span-5'}`}>
              {isLoading ? (
                <Skeleton className="w-full h-[500px] rounded-xl" />
              ) : (
                <DataTable 
                  columns={dataset.columns.map(c => c.name)}
                  data={tableData}
                  onRowClick={handleRowClick}
                  activeRowIndex={activeRowIndex}
                  className="max-h-[600px] border-zinc-800"
                />
              )}
            </div>

            {/* Local Explanation Panel */}
            <AnimatePresence mode="popLayout">
              {activeRowIndex !== undefined && (
                <motion.div 
                  initial={{ opacity: 0, x: 50, width: 0 }}
                  animate={{ opacity: 1, x: 0, width: "auto" }}
                  exit={{ opacity: 0, x: 50, width: 0 }}
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                  className="lg:col-span-3 sticky top-24"
                >
                  <div className="flex justify-between items-center mb-4 bg-card/80 p-4 rounded-xl border border-border/50">
                    <h3 className="text-lg font-medium text-white">Explanation for Row {activeRowIndex}</h3>
                    <button 
                      onClick={() => {
                        setActiveRowIndex(undefined);
                        setLocalExplanation(null);
                      }}
                      className="p-1 rounded-md text-zinc-500 hover:text-white hover:bg-zinc-800 transition-colors"
                    >
                      <X className="h-5 w-5" />
                    </button>
                  </div>

                  {isLoadingLocal ? (
                    <div className="space-y-4">
                      <Skeleton className="h-32 w-full rounded-2xl" />
                      <div className="grid grid-cols-2 gap-4">
                        <Skeleton className="h-64 w-full rounded-xl" />
                        <Skeleton className="h-64 w-full rounded-xl" />
                      </div>
                    </div>
                  ) : localExplanation ? (
                    <PredictionExplanationView explanation={localExplanation} />
                  ) : null}
                </motion.div>
              )}
            </AnimatePresence>

          </div>
        </div>
      </div>
    </AppShell>
  );
}
