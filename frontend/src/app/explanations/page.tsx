"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Lightbulb, Database, X } from "lucide-react";
import AppShell from "@/components/layout/app-shell";
import { useAppStore } from "@/lib/store";
import { getGlobalExplanation, getDatasetPreview, getLocalExplanation } from "@/lib/api";
import type { FeatureImportance, PredictionExplanation } from "@/types";
import { FeatureImportanceChart } from "@/components/charts/feature-importance-chart";
import { DataTable } from "@/components/dashboard/data-table";
import { PredictionExplanationView } from "@/components/dashboard/prediction-explanation";
import { Skeleton } from "@/components/ui/skeleton";

export default function ExplanationsPage() {
  const router = useRouter();
  const { dataset, model } = useAppStore();

  const [globalData, setGlobalData] = useState<FeatureImportance[] | null>(null);
  const [tableData, setTableData] = useState<any[]>([]);
  
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
        const [globalRes, previewRes] = await Promise.all([
          getGlobalExplanation(model.id),
          getDatasetPreview(dataset.id, 50) // Fetch up to 50 rows for preview
        ]);
        
        setGlobalData(globalRes);
        setTableData(previewRes);
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
    
    // If clicking the same row, close it
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
      <div className="mx-auto max-w-7xl space-y-8 pb-16">
        
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <Lightbulb className="h-8 w-8 text-primary" />
            Model Explanations
          </h1>
          <p className="mt-2 text-zinc-400">
            Understand how the {model.model_type.replace('_', ' ')} model makes its predictions.
          </p>
        </div>

        {/* Global Feature Importance */}
        {isLoading ? (
          <Skeleton className="w-full h-[400px] rounded-xl" />
        ) : globalData ? (
          <FeatureImportanceChart data={globalData} />
        ) : null}

        {/* Data Table & Local Explanations */}
        <div className="space-y-4">
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
                  className="max-h-[600px]"
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
                  <div className="flex justify-between items-center mb-4">
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
