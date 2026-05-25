"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Settings2, Play, Database, AlertCircle, ArrowRight } from "lucide-react";
import AppShell from "@/components/layout/app-shell";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { trainModel } from "@/lib/api";
import { useAppStore } from "@/lib/store";

export default function TrainingPage() {
  const router = useRouter();
  const { dataset, setModel } = useAppStore();
  
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [dropColumns, setDropColumns] = useState<string[]>([]);
  const [modelType, setModelType] = useState<"random_forest" | "gradient_boosting" | "logistic_regression">("random_forest");
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!dataset) {
      router.push("/upload");
    } else if (dataset.columns.length > 0) {
      // Auto-select the last column as target by default
      setTargetColumn(dataset.columns[dataset.columns.length - 1].name);
    }
  }, [dataset, router]);

  if (!dataset) return null;

  const toggleDropColumn = (colName: string) => {
    setDropColumns(prev => 
      prev.includes(colName) 
        ? prev.filter(c => c !== colName)
        : [...prev, colName]
    );
  };

  const handleTrain = async () => {
    try {
      setIsTraining(true);
      setError(null);
      
      const config = {
        dataset_id: dataset.id,
        target_column: targetColumn,
        drop_columns: dropColumns,
        model_type: modelType,
        test_size: 0.2
      };
      
      const trainedModel = await trainModel(config);
      setModel(trainedModel);
      
      router.push("/explanations");
    } catch (err: any) {
      setError(err.message || "Failed to train model");
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <AppShell>
      <div className="mx-auto max-w-5xl space-y-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">Model Configuration</h1>
          <p className="mt-2 text-zinc-400">
            Configure your training parameters for dataset: <span className="text-white font-medium">{dataset.filename}</span>
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* ── Left: Configuration Form ── */}
          <div className="md:col-span-2 space-y-6">
            <Card className="glass-card animated-border overflow-hidden">
              <div className="relative z-10 p-8 space-y-8">
                <div className="flex items-center gap-3 border-b border-white/10 pb-4">
                  <div className="p-2 rounded-lg bg-primary/20">
                    <Settings2 className="h-6 w-6 text-primary" />
                  </div>
                  <h2 className="text-xl font-bold text-white text-glow">Training Parameters</h2>
                </div>

              {/* Target Column */}
              <div className="space-y-3">
                <label className="text-sm font-medium text-zinc-300">Target Variable to Predict</label>
                <select
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  className="w-full rounded-md border border-zinc-800 bg-zinc-900/50 px-4 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  {dataset.columns.map(col => (
                    <option key={col.name} value={col.name}>{col.name} ({col.dtype})</option>
                  ))}
                </select>
              </div>

              {/* Model Algorithm */}
              <div className="space-y-3">
                <label className="text-sm font-medium text-zinc-300">Algorithm</label>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { id: "random_forest", label: "Random Forest" },
                    { id: "gradient_boosting", label: "Gradient Boosting" },
                    { id: "logistic_regression", label: "Linear/Logistic" },
                  ].map(algo => (
                    <button
                      key={algo.id}
                      onClick={() => setModelType(algo.id as any)}
                      className={`rounded-lg border px-4 py-3 text-sm font-medium transition-all ${
                        modelType === algo.id
                          ? "border-indigo-500 bg-indigo-500/10 text-indigo-400"
                          : "border-zinc-800 bg-zinc-900/50 text-zinc-400 hover:border-zinc-700 hover:bg-zinc-800"
                      }`}
                    >
                      {algo.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Feature Selection */}
              <div className="space-y-3">
                <label className="text-sm font-medium text-zinc-300">Drop Features (Optional)</label>
                <p className="text-xs text-zinc-500 mb-2">Select columns to exclude from training (e.g. IDs, names).</p>
                <div className="flex flex-wrap gap-2">
                  {dataset.columns.filter(c => c.name !== targetColumn).map(col => {
                    const isDropped = dropColumns.includes(col.name);
                    return (
                      <Badge
                        key={col.name}
                        variant={isDropped ? "destructive" : "outline"}
                        className="cursor-pointer select-none"
                        onClick={() => toggleDropColumn(col.name)}
                      >
                        {col.name}
                      </Badge>
                    );
                  })}
                </div>
              </div>
              </div>
            </Card>

            {error && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center gap-3 rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-red-400"
              >
                <AlertCircle className="h-5 w-5" />
                <p className="text-sm font-medium">{error}</p>
              </motion.div>
            )}

            <div className="flex justify-end pt-4">
              <Button
                variant="accent"
                size="lg"
                onClick={handleTrain}
                disabled={isTraining || !targetColumn}
                className="group relative overflow-hidden"
              >
                {isTraining ? (
                  <span className="flex items-center gap-2">
                    <span className="h-4 w-4 rounded-full border-2 border-white/20 border-t-white animate-spin" />
                    Training Model...
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    <Play className="h-4 w-4" />
                    Start Training
                  </span>
                )}
                {/* Glow effect on hover */}
                <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/10 to-transparent transition-transform duration-500 group-hover:translate-x-full" />
              </Button>
            </div>
          </div>

          {/* ── Right: Dataset Summary ── */}
          <div className="space-y-6">
            <Card className="glass-card overflow-hidden">
              <div className="relative z-10 p-6">
                <div className="flex items-center gap-3 border-b border-white/10 pb-4 mb-4">
                  <div className="p-2 rounded-lg bg-accent/20">
                    <Database className="h-5 w-5 text-accent" />
                  </div>
                  <h2 className="text-lg font-bold text-white">Dataset Info</h2>
                </div>
              
              <div className="space-y-4 text-sm">
                <div className="flex justify-between border-b border-zinc-800/50 pb-2">
                  <span className="text-zinc-500">Rows</span>
                  <span className="text-white font-medium">{dataset.rows.toLocaleString()}</span>
                </div>
                <div className="flex justify-between border-b border-zinc-800/50 pb-2">
                  <span className="text-zinc-500">Columns</span>
                  <span className="text-white font-medium">{dataset.columns.length}</span>
                </div>
                <div className="flex justify-between pb-2">
                  <span className="text-zinc-500">Target</span>
                  <Badge variant="secondary">{targetColumn || "None"}</Badge>
                </div>
              </div>
              </div>
            </Card>
          </div>

        </div>
      </div>
    </AppShell>
  );
}
