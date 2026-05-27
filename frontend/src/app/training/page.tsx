"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  Settings2, Play, Database, AlertCircle, ArrowRight, Trophy,
  ChevronDown, ChevronUp, Zap, Cpu, SlidersHorizontal,
} from "lucide-react";
import AppShell from "@/components/layout/app-shell";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { trainModel, trainAutoML } from "@/lib/api";
import { useAppStore } from "@/lib/store";
import type { TrainedModel } from "@/types";

// ── Hyperparameter config per algorithm ──────────────────────────────
const HYPERPARAM_DEFS: Record<string, { key: string; label: string; type: "int" | "float"; min: number; max: number; step: number; default: number }[]> = {
  random_forest: [
    { key: "n_estimators", label: "Number of Trees", type: "int", min: 10, max: 500, step: 10, default: 100 },
    { key: "max_depth", label: "Max Depth", type: "int", min: 2, max: 50, step: 1, default: 10 },
    { key: "min_samples_split", label: "Min Samples Split", type: "int", min: 2, max: 20, step: 1, default: 2 },
  ],
  xgboost: [
    { key: "n_estimators", label: "Number of Estimators", type: "int", min: 50, max: 500, step: 10, default: 100 },
    { key: "max_depth", label: "Max Depth", type: "int", min: 2, max: 20, step: 1, default: 6 },
    { key: "learning_rate", label: "Learning Rate", type: "float", min: 0.01, max: 1, step: 0.01, default: 0.3 },
  ],
  lightgbm: [
    { key: "n_estimators", label: "Number of Estimators", type: "int", min: 50, max: 500, step: 10, default: 100 },
    { key: "max_depth", label: "Max Depth", type: "int", min: -1, max: 20, step: 1, default: -1 },
    { key: "learning_rate", label: "Learning Rate", type: "float", min: 0.01, max: 1, step: 0.01, default: 0.1 },
  ],
  logistic_regression: [
    { key: "max_iter", label: "Max Iterations", type: "int", min: 100, max: 5000, step: 100, default: 1000 },
    { key: "C", label: "Regularization (C)", type: "float", min: 0.01, max: 10, step: 0.01, default: 1.0 },
  ],
  mlp: [
    { key: "max_iter", label: "Max Iterations", type: "int", min: 100, max: 2000, step: 100, default: 500 },
    { key: "alpha", label: "L2 Penalty (Alpha)", type: "float", min: 0.0001, max: 0.1, step: 0.0001, default: 0.0001 },
  ]
};

export default function TrainingPage() {
  const router = useRouter();
  const { dataset, setModel, setLeaderboard } = useAppStore();

  const [targetColumn, setTargetColumn] = useState<string>("");
  const [dropColumns, setDropColumns] = useState<string[]>([]);
  const [modelType, setModelType] = useState<"random_forest" | "xgboost" | "lightgbm" | "logistic_regression" | "mlp">("random_forest");
  const [mode, setMode] = useState<"single" | "automl">("single");
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Hyperparameters state
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [hyperparams, setHyperparams] = useState<Record<string, number>>({});

  // Leaderboard results
  const [leaderboardResults, setLeaderboardResults] = useState<TrainedModel[] | null>(null);

  useEffect(() => {
    if (!dataset) {
      router.push("/upload");
    } else if (dataset.columns.length > 0) {
      setTargetColumn(dataset.columns[dataset.columns.length - 1].name);
    }
  }, [dataset, router]);

  // Reset hyperparams when algorithm changes
  useEffect(() => {
    const defs = HYPERPARAM_DEFS[modelType] || [];
    const defaults: Record<string, number> = {};
    defs.forEach(d => { defaults[d.key] = d.default; });
    setHyperparams(defaults);
  }, [modelType]);

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
      setLeaderboardResults(null);

      if (mode === "automl") {
        const config = {
          dataset_id: dataset.id,
          target_column: targetColumn,
          drop_columns: dropColumns,
          test_size: 0.2,
        };

        const results = await trainAutoML(config);
        setLeaderboardResults(results);
        setLeaderboard(results);

        // Auto-select the best model
        if (results.length > 0) {
          setModel(results[0]);
        }
      } else {
        // Single model with hyperparams
        const config = {
          dataset_id: dataset.id,
          target_column: targetColumn,
          drop_columns: dropColumns,
          model_type: modelType,
          test_size: 0.2,
          hyperparams: showAdvanced ? hyperparams : undefined,
        };

        const trainedModel = await trainModel(config);
        setModel(trainedModel);
        router.push("/explanations");
      }
    } catch (err: any) {
      setError(err.message || "Failed to train model");
    } finally {
      setIsTraining(false);
    }
  };

  const handleSelectFromLeaderboard = (m: TrainedModel) => {
    setModel(m);
    router.push("/explanations");
  };

  const formatMetric = (val?: number | null) => {
    if (val === undefined || val === null) return "—";
    if (Math.abs(val) <= 1) return (val * 100).toFixed(1) + "%";
    return val.toFixed(3);
  };

  const algoLabel = (type: string) => {
    switch (type) {
      case "random_forest": return "Random Forest";
      case "xgboost": return "XGBoost 🚀";
      case "lightgbm": return "LightGBM ⚡";
      case "mlp": return "Neural Network 🧠";
      case "logistic_regression": return "Logistic Regression";
      case "linear_regression": return "Linear Regression";
      default: return type;
    }
  };

  return (
    <AppShell>
      <div className="mx-auto max-w-5xl space-y-8 pb-16">
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

                {/* Training Mode Toggle */}
                <div className="space-y-3">
                  <label className="text-sm font-medium text-zinc-300">Training Mode</label>
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      onClick={() => setMode("single")}
                      className={`flex items-center justify-center gap-2 rounded-lg border px-4 py-3 text-sm font-medium transition-all ${
                        mode === "single"
                          ? "border-indigo-500 bg-indigo-500/10 text-indigo-400"
                          : "border-zinc-800 bg-zinc-900/50 text-zinc-400 hover:border-zinc-700 hover:bg-zinc-800"
                      }`}
                    >
                      <Cpu className="h-4 w-4" />
                      Single Model
                    </button>
                    <button
                      onClick={() => setMode("automl")}
                      className={`flex items-center justify-center gap-2 rounded-lg border px-4 py-3 text-sm font-medium transition-all ${
                        mode === "automl"
                          ? "border-amber-500 bg-amber-500/10 text-amber-400"
                          : "border-zinc-800 bg-zinc-900/50 text-zinc-400 hover:border-zinc-700 hover:bg-zinc-800"
                      }`}
                    >
                      <Zap className="h-4 w-4" />
                      AutoML (Compare All)
                    </button>
                  </div>
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

                {/* Model Algorithm — only in single mode */}
                {mode === "single" && (
                  <div className="space-y-3">
                    <label className="text-sm font-medium text-zinc-300">Algorithm</label>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {[
                        { id: "xgboost", label: "XGBoost 🚀" },
                        { id: "lightgbm", label: "LightGBM ⚡" },
                        { id: "random_forest", label: "Random Forest" },
                        { id: "mlp", label: "Neural Network 🧠" },
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
                )}

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

                {/* ── Advanced Settings (Hyperparameters) ── */}
                {mode === "single" && (
                  <div className="border-t border-zinc-800 pt-4">
                    <button
                      onClick={() => setShowAdvanced(!showAdvanced)}
                      className="flex items-center gap-2 text-sm font-medium text-zinc-400 hover:text-white transition-colors"
                    >
                      <SlidersHorizontal className="h-4 w-4" />
                      Advanced Settings (Hyperparameters)
                      {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                    </button>

                    <AnimatePresence>
                      {showAdvanced && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={{ opacity: 0, height: 0 }}
                          transition={{ duration: 0.3 }}
                          className="overflow-hidden"
                        >
                          <div className="mt-4 space-y-5 bg-black/20 rounded-lg p-4 border border-zinc-800">
                            {(HYPERPARAM_DEFS[modelType] || []).map(param => (
                              <div key={param.key} className="space-y-2">
                                <div className="flex items-center justify-between">
                                  <label className="text-sm text-zinc-300">{param.label}</label>
                                  <span className="text-sm font-mono text-primary font-bold">
                                    {hyperparams[param.key] ?? param.default}
                                  </span>
                                </div>
                                <input
                                  type="range"
                                  min={param.min}
                                  max={param.max}
                                  step={param.step}
                                  value={hyperparams[param.key] ?? param.default}
                                  onChange={(e) => {
                                    const val = param.type === "int" ? parseInt(e.target.value) : parseFloat(e.target.value);
                                    setHyperparams(prev => ({ ...prev, [param.key]: val }));
                                  }}
                                  className="w-full accent-primary h-1.5 bg-zinc-700 rounded-full cursor-pointer"
                                />
                                <div className="flex justify-between text-[10px] text-zinc-600">
                                  <span>{param.min}</span>
                                  <span>{param.max}</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                )}
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
                disabled={isTraining || !targetColumn || (dataset.columns.length - 1 - dropColumns.length <= 0)}
                className="group relative overflow-hidden"
              >
                {isTraining ? (
                  <span className="flex items-center gap-2">
                    <span className="h-4 w-4 rounded-full border-2 border-white/20 border-t-white animate-spin" />
                    {mode === "automl" ? "Training 3 Models..." : "Training Model..."}
                  </span>
                ) : (
                  <span className="flex items-center gap-2">
                    {mode === "automl" ? <Zap className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    {mode === "automl" ? "Run AutoML" : "Start Training"}
                  </span>
                )}
                <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/10 to-transparent transition-transform duration-500 group-hover:translate-x-full" />
              </Button>
            </div>

            {/* ── AutoML Leaderboard ── */}
            <AnimatePresence>
              {leaderboardResults && leaderboardResults.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="space-y-4 pt-2"
                >
                  <div className="flex items-center gap-3">
                    <Trophy className="h-6 w-6 text-amber-400" />
                    <h3 className="text-xl font-bold text-white">AutoML Leaderboard</h3>
                  </div>

                  <div className="bg-card/40 backdrop-blur-md border border-border/50 rounded-xl overflow-hidden shadow-sm">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-zinc-800 text-zinc-400 text-xs uppercase tracking-wider">
                          <th className="py-3 px-4 text-left">Rank</th>
                          <th className="py-3 px-4 text-left">Algorithm</th>
                          {leaderboardResults[0].task_type === "classification" ? (
                            <>
                              <th className="py-3 px-4 text-right">Accuracy</th>
                              <th className="py-3 px-4 text-right">F1 Score</th>
                              <th className="py-3 px-4 text-right">Train Acc</th>
                              <th className="py-3 px-4 text-right">Overfit Gap</th>
                            </>
                          ) : (
                            <>
                              <th className="py-3 px-4 text-right">R²</th>
                              <th className="py-3 px-4 text-right">RMSE</th>
                              <th className="py-3 px-4 text-right">Train RMSE</th>
                              <th className="py-3 px-4 text-right">Overfit Gap</th>
                            </>
                          )}
                          <th className="py-3 px-4 text-center">Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {leaderboardResults.map((m, i) => {
                          const isClassification = m.task_type === "classification";
                          const trainMetric = isClassification ? m.metrics.train_accuracy : m.metrics.train_rmse;
                          const testMetric = isClassification ? m.metrics.accuracy : m.metrics.rmse;
                          let overfitGap = "—";
                          let gapColor = "text-zinc-400";

                          if (trainMetric !== undefined && trainMetric !== null && testMetric !== undefined && testMetric !== null) {
                            if (isClassification) {
                              const gap = ((trainMetric - testMetric) * 100);
                              overfitGap = `${gap.toFixed(1)}%`;
                              gapColor = gap > 10 ? "text-red-400" : gap > 5 ? "text-amber-400" : "text-emerald-400";
                            } else {
                              const gap = testMetric - trainMetric;
                              overfitGap = gap.toFixed(2);
                              gapColor = gap > trainMetric * 0.5 ? "text-red-400" : "text-emerald-400";
                            }
                          }

                          return (
                            <motion.tr
                              key={m.id}
                              initial={{ opacity: 0, x: -10 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: i * 0.1 }}
                              className={`border-b border-zinc-800/50 transition-colors hover:bg-zinc-800/30 ${
                                i === 0 ? "bg-amber-500/5" : ""
                              }`}
                            >
                              <td className="py-3.5 px-4">
                                <span className="flex items-center gap-2">
                                  {i === 0 ? (
                                    <Trophy className="h-4 w-4 text-amber-400" />
                                  ) : (
                                    <span className="text-zinc-500 font-mono">{i + 1}</span>
                                  )}
                                </span>
                              </td>
                              <td className="py-3.5 px-4 text-white font-medium capitalize">
                                {algoLabel(m.model_type)}
                              </td>

                              {isClassification ? (
                                <>
                                  <td className="py-3.5 px-4 text-right font-mono text-white font-semibold">
                                    {formatMetric(m.metrics.accuracy)}
                                  </td>
                                  <td className="py-3.5 px-4 text-right font-mono text-zinc-300">
                                    {formatMetric(m.metrics.f1_score)}
                                  </td>
                                  <td className="py-3.5 px-4 text-right font-mono text-zinc-400">
                                    {formatMetric(m.metrics.train_accuracy)}
                                  </td>
                                </>
                              ) : (
                                <>
                                  <td className="py-3.5 px-4 text-right font-mono text-white font-semibold">
                                    {m.metrics.r2?.toFixed(4) ?? "—"}
                                  </td>
                                  <td className="py-3.5 px-4 text-right font-mono text-zinc-300">
                                    {m.metrics.rmse?.toFixed(3) ?? "—"}
                                  </td>
                                  <td className="py-3.5 px-4 text-right font-mono text-zinc-400">
                                    {m.metrics.train_rmse?.toFixed(3) ?? "—"}
                                  </td>
                                </>
                              )}

                              <td className={`py-3.5 px-4 text-right font-mono font-medium ${gapColor}`}>
                                {overfitGap}
                              </td>
                              <td className="py-3.5 px-4 text-center">
                                <button
                                  onClick={() => handleSelectFromLeaderboard(m)}
                                  className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:text-primary/80 transition-colors px-3 py-1.5 rounded-md bg-primary/10 hover:bg-primary/20"
                                >
                                  Select <ArrowRight className="h-3 w-3" />
                                </button>
                              </td>
                            </motion.tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
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
                  <div className="flex justify-between border-b border-zinc-800/50 pb-2">
                    <span className="text-zinc-500">Target</span>
                    <Badge variant="secondary">{targetColumn || "None"}</Badge>
                  </div>
                  <div className="flex justify-between pb-2">
                    <span className="text-zinc-500">Mode</span>
                    <Badge variant={mode === "automl" ? "default" : "secondary"}>
                      {mode === "automl" ? "⚡ AutoML" : "Single"}
                    </Badge>
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
