"use client";

import { motion } from "framer-motion";
import { BrainCircuit, Target, TrendingUp, AlertCircle } from "lucide-react";
import type { FeatureImportance } from "@/types";

export function ModelInsightsCard({ model, globalData }: { model: any, globalData: FeatureImportance[] | null }) {
  if (!model) return null;

  const topFeature = globalData && globalData.length > 0 ? globalData[0].feature : "Unknown";
  const metrics = model.metrics || {};
  
  const isClassification = model.task_type === "classification";
  
  const formatMetric = (val: any) => {
    if (typeof val === 'number') {
      // For accuracy/f1/r2, percentages are nicer
      if (val <= 1 && val >= -1) return `${(val * 100).toFixed(1)}%`;
      return val.toFixed(3);
    }
    return "N/A";
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gradient-to-br from-primary/20 via-primary/5 to-transparent border border-primary/20 rounded-2xl p-6 shadow-lg mb-8 relative overflow-hidden"
    >
      {/* Decorative background element */}
      <div className="absolute -top-24 -right-24 w-64 h-64 bg-primary/10 rounded-full blur-3xl pointer-events-none" />

      <div className="flex flex-col md:flex-row gap-8 items-start relative z-10">
        
        {/* Left Side: Summary Text */}
        <div className="flex-1 space-y-4">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-primary/20 rounded-lg">
              <BrainCircuit className="h-6 w-6 text-primary" />
            </div>
            <h2 className="text-2xl font-bold text-white tracking-tight">AI Training Insights</h2>
          </div>
          
          <p className="text-zinc-300 leading-relaxed text-sm md:text-base">
            This <strong className="text-white">{model.model_type.replace('_', ' ')}</strong> model was trained 
            to predict <strong className="text-primary">{model.target_column}</strong>. 
            Based on the global explanation analysis, the most influential factor driving the model's decisions is <strong className="text-indigo-400">{topFeature}</strong>.
          </p>
          
          
          <div className="bg-black/30 border border-zinc-800 rounded-lg p-4 flex gap-3 items-start mt-4">
            <AlertCircle className="h-5 w-5 text-indigo-400 shrink-0 mt-0.5" />
            <div className="space-y-2">
              <p className="text-sm text-zinc-400">
                <strong>Interpretation:</strong> Adjusting the value of '{topFeature}' will likely yield the most significant changes in future predictions. Review the correlation heatmap below to see if '{topFeature}' suffers from multicollinearity with other features.
              </p>
              
              {/* Overfitting Check */}
              {isClassification && metrics.train_accuracy !== undefined && metrics.accuracy !== undefined && (
                <p className="text-sm">
                  <strong>Overfitting Check:</strong>{' '}
                  {metrics.train_accuracy - metrics.accuracy > 0.10 ? (
                    <span className="text-red-400">High Risk (Train Acc: {(metrics.train_accuracy*100).toFixed(1)}% vs Test Acc: {(metrics.accuracy*100).toFixed(1)}%). The model may be memorizing the training data.</span>
                  ) : (
                    <span className="text-emerald-400">Healthy Fit (Train Acc: {(metrics.train_accuracy*100).toFixed(1)}% vs Test Acc: {(metrics.accuracy*100).toFixed(1)}%). The model generalizes well to unseen data.</span>
                  )}
                </p>
              )}
              {!isClassification && metrics.train_rmse !== undefined && metrics.rmse !== undefined && (
                <p className="text-sm">
                  <strong>Overfitting Check:</strong>{' '}
                  {metrics.rmse > metrics.train_rmse * 1.5 ? (
                    <span className="text-red-400">High Risk (Train RMSE: {metrics.train_rmse.toFixed(2)} vs Test RMSE: {metrics.rmse.toFixed(2)}). The model performs significantly worse on unseen data.</span>
                  ) : (
                    <span className="text-emerald-400">Healthy Fit (Train RMSE: {metrics.train_rmse.toFixed(2)} vs Test RMSE: {metrics.rmse.toFixed(2)}). The model generalizes well to unseen data.</span>
                  )}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Right Side: Key Metrics Grid */}
        <div className="w-full md:w-auto shrink-0 min-w-[300px]">
          <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Target className="h-4 w-4" />
            Performance Metrics
          </h3>
          
          <div className="grid grid-cols-2 gap-3">
            {isClassification ? (
              <>
                <div className="bg-black/40 border border-zinc-800 rounded-xl p-4">
                  <p className="text-xs text-zinc-500 mb-1">Accuracy</p>
                  <p className="text-xl font-bold text-white">{formatMetric(metrics.accuracy)}</p>
                </div>
                <div className="bg-black/40 border border-zinc-800 rounded-xl p-4">
                  <p className="text-xs text-zinc-500 mb-1">F1 Score</p>
                  <p className="text-xl font-bold text-white">{formatMetric(metrics.f1_score)}</p>
                </div>
                <div className="bg-black/40 border border-zinc-800 rounded-xl p-4">
                  <p className="text-xs text-zinc-500 mb-1">Precision</p>
                  <p className="text-xl font-bold text-white">{formatMetric(metrics.precision)}</p>
                </div>
                <div className="bg-black/40 border border-zinc-800 rounded-xl p-4">
                  <p className="text-xs text-zinc-500 mb-1">Recall</p>
                  <p className="text-xl font-bold text-white">{formatMetric(metrics.recall)}</p>
                </div>
              </>
            ) : (
              <>
                <div className="bg-black/40 border border-zinc-800 rounded-xl p-4">
                  <p className="text-xs text-zinc-500 mb-1">RMSE</p>
                  <p className="text-xl font-bold text-white">{formatMetric(metrics.rmse)}</p>
                </div>
                <div className="bg-black/40 border border-zinc-800 rounded-xl p-4">
                  <p className="text-xs text-zinc-500 mb-1">MAE</p>
                  <p className="text-xl font-bold text-white">{formatMetric(metrics.mae)}</p>
                </div>
                <div className="col-span-2 bg-black/40 border border-zinc-800 rounded-xl p-4 flex items-center justify-between">
                  <div>
                    <p className="text-xs text-zinc-500 mb-1">R² Score</p>
                    <p className="text-xl font-bold text-white">{formatMetric(metrics.r2)}</p>
                  </div>
                  <TrendingUp className="h-6 w-6 text-emerald-400 opacity-50" />
                </div>
              </>
            )}
          </div>
        </div>

      </div>
    </motion.div>
  );
}
