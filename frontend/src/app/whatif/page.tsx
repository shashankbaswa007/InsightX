"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { SlidersHorizontal, Calculator, AlertTriangle, ArrowRight } from "lucide-react";
import AppShell from "@/components/layout/app-shell";
import { useAppStore } from "@/lib/store";
import { predictWhatIf, getDatasetPreview } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";

export default function WhatIfPage() {
  const router = useRouter();
  const { dataset, model } = useAppStore();

  const [baselineRow, setBaselineRow] = useState<any | null>(null);
  const [modifiedFeatures, setModifiedFeatures] = useState<Record<string, any>>({});
  const [prediction, setPrediction] = useState<number | string | null>(null);
  const [probabilities, setProbabilities] = useState<number[] | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!dataset || !model) {
      router.push("/upload");
      return;
    }

    const loadBaseline = async () => {
      try {
        setIsLoading(true);
        // Load first row as baseline
        const data = await getDatasetPreview(dataset.id, 1);
        if (data && data.length > 0) {
          setBaselineRow(data[0]);
          setModifiedFeatures({...data[0]});
        }
      } catch (err) {
        setError("Failed to load baseline data");
      } finally {
        setIsLoading(false);
      }
    };

    loadBaseline();
  }, [dataset, model, router]);

  // Debounced prediction fetch
  useEffect(() => {
    if (!model || !dataset || Object.keys(modifiedFeatures).length === 0) return;

    const timer = setTimeout(async () => {
      try {
        setIsPredicting(true);
        setError(null);
        const res = await predictWhatIf(model.id, dataset.id, 0, modifiedFeatures);
        setPrediction(res.prediction);
        setProbabilities(res.probabilities || null);
      } catch (err: any) {
        setError(err.message || "Failed to generate prediction");
      } finally {
        setIsPredicting(false);
      }
    }, 500); // 500ms debounce

    return () => clearTimeout(timer);
  }, [modifiedFeatures, model, dataset]);

  const handleFeatureChange = (colName: string, value: string) => {
    // Attempt to convert to number if possible, else keep string
    const numVal = Number(value);
    const finalVal = isNaN(numVal) || value === '' ? value : numVal;
    
    setModifiedFeatures(prev => ({
      ...prev,
      [colName]: finalVal
    }));
  };

  const renderInput = (col: any) => {
    const isTarget = col.name === model?.target_column;
    if (isTarget) return null; // Don't edit target column

    const val = modifiedFeatures[col.name] ?? "";

    if (col.type === "numeric") {
      return (
        <div key={col.name} className="flex flex-col gap-2 p-4 bg-zinc-900/50 rounded-lg border border-zinc-800">
          <label className="text-sm font-medium text-zinc-300 flex justify-between">
            {col.name}
            <span className="text-xs text-zinc-500 font-mono">{val}</span>
          </label>
          <input
            type="number"
            step="any"
            value={val}
            onChange={(e) => handleFeatureChange(col.name, e.target.value)}
            className="w-full bg-black border border-zinc-700 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:border-primary transition-colors"
          />
        </div>
      );
    }

    return (
      <div key={col.name} className="flex flex-col gap-2 p-4 bg-zinc-900/50 rounded-lg border border-zinc-800">
        <label className="text-sm font-medium text-zinc-300">
          {col.name} <span className="text-xs text-zinc-500">(Categorical)</span>
        </label>
        <input
          type="text"
          value={val}
          onChange={(e) => handleFeatureChange(col.name, e.target.value)}
          className="w-full bg-black border border-zinc-700 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:border-primary transition-colors"
        />
      </div>
    );
  };

  if (!dataset || !model) return null;

  return (
    <AppShell>
      <div className="mx-auto max-w-6xl space-y-8 pb-16">
        
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <SlidersHorizontal className="h-8 w-8 text-primary" />
            What-If Analysis
          </h1>
          <p className="mt-2 text-zinc-400 max-w-2xl">
            Tweak feature values in real-time to see how they impact the model's prediction.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
          
          {/* Controls Panel */}
          <div className="lg:col-span-2 space-y-6 bg-card/40 backdrop-blur-md p-6 rounded-xl border border-border/50 shadow-sm">
            <h3 className="text-xl font-semibold text-white mb-4">Adjust Features</h3>
            
            {isLoading ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Skeleton className="h-24 w-full" />
                <Skeleton className="h-24 w-full" />
                <Skeleton className="h-24 w-full" />
                <Skeleton className="h-24 w-full" />
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {dataset.columns.map(renderInput)}
              </div>
            )}
            
            <div className="mt-6 flex justify-end">
              <button 
                onClick={() => setModifiedFeatures({...baselineRow})}
                className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors"
              >
                Reset to Baseline
              </button>
            </div>
          </div>

          {/* Result Panel */}
          <div className="lg:col-span-1 sticky top-24 space-y-6">
            <motion.div 
              className="bg-primary/10 border border-primary/30 p-6 rounded-xl shadow-lg relative overflow-hidden"
            >
              {isPredicting && (
                <div className="absolute top-0 left-0 w-full h-1 bg-primary/20">
                  <motion.div 
                    className="h-full bg-primary"
                    animate={{ x: ["-100%", "100%"] }}
                    transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                  />
                </div>
              )}
              
              <div className="flex items-center gap-2 mb-6">
                <Calculator className="h-6 w-6 text-primary" />
                <h3 className="text-xl font-bold text-white">Live Prediction</h3>
              </div>

              {error ? (
                <div className="bg-red-500/10 border border-red-500/30 p-4 rounded-lg flex items-start gap-3">
                  <AlertTriangle className="h-5 w-5 text-red-400 shrink-0 mt-0.5" />
                  <p className="text-sm text-red-300">{error}</p>
                </div>
              ) : (
                <div className="space-y-6">
                  <div>
                    <p className="text-sm text-zinc-400 mb-1">Predicted {model.target_column}</p>
                    <div className="text-4xl font-bold text-white flex items-center gap-3">
                      <ArrowRight className="h-6 w-6 text-zinc-500" />
                      {prediction !== null ? (
                        <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary to-violet-400">
                          {typeof prediction === 'number' && !Number.isInteger(prediction) ? prediction.toFixed(4) : prediction}
                        </span>
                      ) : (
                        "---"
                      )}
                    </div>
                  </div>

                  {probabilities && (
                    <div className="pt-4 border-t border-primary/20 space-y-3">
                      <p className="text-sm text-zinc-400">Class Probabilities</p>
                      {probabilities.map((prob, idx) => (
                        <div key={idx}>
                          <div className="flex justify-between text-xs mb-1">
                            <span className="text-zinc-300">Class {idx}</span>
                            <span className="text-primary font-medium">{(prob * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-black/40 rounded-full h-2">
                            <motion.div 
                              className="bg-primary h-2 rounded-full"
                              initial={{ width: 0 }}
                              animate={{ width: `${prob * 100}%` }}
                              transition={{ duration: 0.5 }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          </div>

        </div>
      </div>
    </AppShell>
  );
}
