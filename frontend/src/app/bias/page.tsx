"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Shield, Search, AlertCircle, BarChart2 } from "lucide-react";
import AppShell from "@/components/layout/app-shell";
import { useAppStore } from "@/lib/store";
import { analyzeBias } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

export default function BiasPage() {
  const router = useRouter();
  const { dataset, model } = useAppStore();

  const [protectedAttribute, setProtectedAttribute] = useState<string>("");
  const [results, setResults] = useState<any | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!dataset || !model) {
      router.push("/upload");
      return;
    }
    
    if (model.task_type !== "classification") {
      setError("Bias detection is currently only supported for classification models.");
    }
  }, [dataset, model, router]);

  const handleAnalyze = async () => {
    if (!protectedAttribute) return;
    
    try {
      setIsLoading(true);
      setError(null);
      const res = await analyzeBias(model!.id, dataset!.id, protectedAttribute);
      setResults(res);
    } catch (err: any) {
      setError(err.message || "Failed to analyze bias");
      setResults(null);
    } finally {
      setIsLoading(false);
    }
  };

  const getMetricColor = (val: number, isMin: boolean, isMax: boolean) => {
    if (isMax) return "hsl(var(--primary))"; // highest selection rate
    if (isMin) return "hsl(var(--destructive))"; // lowest selection rate
    return "hsl(var(--muted-foreground))";
  };

  if (!dataset || !model) return null;

  return (
    <AppShell>
      <div className="mx-auto max-w-5xl space-y-8 pb-16">
        
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <Shield className="h-8 w-8 text-rose-500" />
            Bias & Fairness Detection
          </h1>
          <p className="mt-2 text-zinc-400 max-w-2xl">
            Evaluate your model for potential disparities across different demographic groups. 
            Select a protected attribute to analyze Selection Rate differences.
          </p>
        </div>

        {error && !isLoading && (
          <div className="bg-red-500/10 border border-red-500/30 p-4 rounded-lg flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-red-400 shrink-0 mt-0.5" />
            <p className="text-sm text-red-300">{error}</p>
          </div>
        )}

        <div className="bg-card/40 backdrop-blur-md p-6 rounded-xl border border-border/50 shadow-sm flex flex-col md:flex-row gap-4 items-end">
          <div className="flex-1 w-full">
            <label className="block text-sm font-medium text-zinc-300 mb-2">
              Select Protected Attribute
            </label>
            <select
              value={protectedAttribute}
              onChange={(e) => setProtectedAttribute(e.target.value)}
              className="w-full bg-black border border-zinc-700 rounded-md px-4 py-3 text-sm text-white focus:outline-none focus:border-primary"
              disabled={model.task_type !== "classification"}
            >
              <option value="" disabled>-- Select a column --</option>
              {dataset.columns.filter(c => c.name !== model.target_column).map((col) => (
                <option key={col.name} value={col.name}>{col.name} ({col.dtype})</option>
              ))}
            </select>
          </div>
          
          <button
            onClick={handleAnalyze}
            disabled={!protectedAttribute || isLoading || model.task_type !== "classification"}
            className="h-[46px] px-6 bg-primary text-primary-foreground font-semibold rounded-md hover:bg-primary/90 transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            {isLoading ? "Analyzing..." : "Analyze Bias"}
            {!isLoading && <Search className="h-4 w-4" />}
          </button>
        </div>

        {isLoading && (
          <div className="space-y-4 pt-4">
            <Skeleton className="h-12 w-1/3" />
            <Skeleton className="h-[400px] w-full rounded-xl" />
          </div>
        )}

        {results && !isLoading && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6 pt-4"
          >
            <div className="flex items-center gap-2 border-b border-zinc-800 pb-2">
              <BarChart2 className="h-5 w-5 text-indigo-400" />
              <h2 className="text-xl font-semibold text-white">Analysis Results: {results.protected_attribute}</h2>
            </div>
            
            <p className="text-sm text-zinc-400">
              The chart below shows the <span className="text-white font-medium">Selection Rate</span> (percentage of positive predictions, i.e., class '{results.positive_class}') for each group within <span className="text-white font-medium">{results.protected_attribute}</span>. 
              Significant differences between groups may indicate demographic disparity.
            </p>

            <div className="bg-black/30 rounded-xl p-6 border border-zinc-800 h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={results.metrics}
                  margin={{ top: 20, right: 30, left: 20, bottom: 50 }}
                >
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#3f3f46" />
                  <XAxis 
                    dataKey="group" 
                    stroke="#a1a1aa" 
                    fontSize={12}
                    tickMargin={10}
                    angle={-45}
                    textAnchor="end"
                  />
                  <YAxis 
                    stroke="#a1a1aa" 
                    fontSize={12}
                    tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-zinc-900 border border-zinc-700 p-3 rounded-lg shadow-xl">
                            <p className="font-semibold text-white mb-1">Group: {data.group}</p>
                            <p className="text-sm text-zinc-400">Total Samples: {data.total}</p>
                            <p className="text-sm text-zinc-400">Positive Preds: {data.positive_predictions}</p>
                            <div className="mt-2 pt-2 border-t border-zinc-700">
                              <p className="text-sm text-primary font-bold">
                                Selection Rate: {(data.selection_rate * 100).toFixed(2)}%
                              </p>
                            </div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar dataKey="selection_rate" radius={[4, 4, 0, 0]}>
                    {results.metrics.map((entry: any, index: number) => {
                      const maxRate = Math.max(...results.metrics.map((m: any) => m.selection_rate));
                      const minRate = Math.min(...results.metrics.map((m: any) => m.selection_rate));
                      return (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={getMetricColor(entry.selection_rate, entry.selection_rate === minRate, entry.selection_rate === maxRate)} 
                        />
                      );
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Disparate Impact summary */}
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-5">
              <h4 className="font-semibold text-blue-400 mb-2">Fairness Insight</h4>
              {(() => {
                const maxRate = Math.max(...results.metrics.map((m: any) => m.selection_rate));
                const minRate = Math.min(...results.metrics.map((m: any) => m.selection_rate));
                if (maxRate === 0) return <p className="text-sm text-zinc-300">No positive predictions made for any group.</p>;
                
                const diRatio = minRate / maxRate;
                const minGroup = results.metrics.find((m: any) => m.selection_rate === minRate)?.group;
                const maxGroup = results.metrics.find((m: any) => m.selection_rate === maxRate)?.group;

                return (
                  <div>
                    <p className="text-sm text-zinc-300 mb-2">
                      The highest selection rate is for group <strong>{maxGroup}</strong> ({(maxRate * 100).toFixed(1)}%). 
                      The lowest is for group <strong>{minGroup}</strong> ({(minRate * 100).toFixed(1)}%).
                    </p>
                    <p className="text-sm text-zinc-300">
                      <strong>Disparate Impact Ratio: {(diRatio).toFixed(3)}</strong>
                    </p>
                    {diRatio < 0.8 && (
                      <p className="text-sm text-red-400 mt-2 font-medium">
                        Warning: The ratio is below the standard 0.8 threshold, indicating potential bias against group '{minGroup}'.
                      </p>
                    )}
                  </div>
                );
              })()}
            </div>

          </motion.div>
        )}

      </div>
    </AppShell>
  );
}
