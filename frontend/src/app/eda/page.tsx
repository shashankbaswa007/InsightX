"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  BarChart3, AlertCircle, Database, Hash, Type, TrendingUp,
  Percent, ArrowUpDown,
} from "lucide-react";
import AppShell from "@/components/layout/app-shell";
import { useAppStore } from "@/lib/store";
import { getEDA } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";

const COLORS = [
  "hsl(250, 80%, 60%)", "hsl(210, 80%, 55%)", "hsl(170, 70%, 50%)",
  "hsl(40, 90%, 55%)", "hsl(340, 75%, 55%)", "hsl(280, 70%, 60%)",
  "hsl(120, 60%, 45%)", "hsl(15, 85%, 55%)",
];

export default function EDAPage() {
  const router = useRouter();
  const { dataset } = useAppStore();

  const [data, setData] = useState<any | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<string>("name");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  useEffect(() => {
    if (!dataset) {
      router.push("/upload");
      return;
    }

    const load = async () => {
      try {
        setIsLoading(true);
        const res = await getEDA(dataset.id);
        setData(res);
      } catch (err: any) {
        setError(err.message || "Failed to load EDA");
      } finally {
        setIsLoading(false);
      }
    };

    load();
  }, [dataset, router]);

  const toggleSort = (key: string) => {
    if (sortKey === key) {
      setSortDir(prev => prev === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  const sortedStats = data?.column_stats
    ? [...data.column_stats].sort((a: any, b: any) => {
        let aVal = a[sortKey], bVal = b[sortKey];
        if (typeof aVal === "string") aVal = aVal.toLowerCase();
        if (typeof bVal === "string") bVal = bVal.toLowerCase();
        if (aVal < bVal) return sortDir === "asc" ? -1 : 1;
        if (aVal > bVal) return sortDir === "asc" ? 1 : -1;
        return 0;
      })
    : [];

  if (!dataset) return null;

  return (
    <AppShell>
      <div className="mx-auto max-w-[1400px] space-y-8 pb-16">

        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <BarChart3 className="h-8 w-8 text-primary" />
            Data Explorer &amp; EDA
          </h1>
          <p className="mt-2 text-zinc-400 max-w-2xl">
            Understand your dataset before training. Explore distributions, missing values, and column statistics for{" "}
            <span className="text-white font-medium">{dataset.filename}</span>.
          </p>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/30 p-4 rounded-lg flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-red-400 shrink-0 mt-0.5" />
            <p className="text-sm text-red-300">{error}</p>
          </div>
        )}

        {isLoading ? (
          <div className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-24 rounded-xl" />
              ))}
            </div>
            <Skeleton className="h-[400px] rounded-xl" />
          </div>
        ) : data ? (
          <>
            {/* Summary Cards */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {[
                { label: "Rows", value: data.summary.total_rows.toLocaleString(), icon: Database, color: "text-blue-400" },
                { label: "Columns", value: data.summary.total_columns, icon: Hash, color: "text-indigo-400" },
                { label: "Missing Cells", value: data.summary.total_missing_cells.toLocaleString(), icon: AlertCircle, color: "text-amber-400" },
                { label: "Missing %", value: `${data.summary.missing_pct}%`, icon: Percent, color: "text-rose-400" },
                { label: "Duplicate Rows", value: data.summary.duplicate_rows.toLocaleString(), icon: TrendingUp, color: "text-emerald-400" },
              ].map((card, i) => (
                <motion.div
                  key={card.label}
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.06 }}
                  className="bg-card/40 backdrop-blur-md border border-border/50 rounded-xl p-4 shadow-sm"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <card.icon className={`h-4 w-4 ${card.color}`} />
                    <span className="text-xs text-zinc-500 uppercase tracking-wider">{card.label}</span>
                  </div>
                  <p className="text-2xl font-bold text-white">{card.value}</p>
                </motion.div>
              ))}
            </div>

            {/* Missing Values Bar */}
            {data.missing_summary.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-card/40 backdrop-blur-md border border-border/50 rounded-xl p-6 shadow-sm"
              >
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <AlertCircle className="h-5 w-5 text-amber-400" />
                  Missing Value Distribution
                </h3>
                <div className="space-y-3">
                  {data.missing_summary.map((m: any) => (
                    <div key={m.column} className="flex items-center gap-4">
                      <span className="text-sm text-zinc-300 w-40 truncate shrink-0" title={m.column}>{m.column}</span>
                      <div className="flex-1 bg-zinc-800 rounded-full h-3 relative overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${m.pct}%` }}
                          transition={{ duration: 0.7 }}
                          className="h-full rounded-full"
                          style={{ background: `linear-gradient(90deg, hsl(40, 90%, 55%), hsl(0, 80%, 55%))` }}
                        />
                      </div>
                      <span className="text-xs text-zinc-400 w-20 text-right shrink-0">
                        {m.count} ({m.pct}%)
                      </span>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Column Statistics Table */}
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-card/40 backdrop-blur-md border border-border/50 rounded-xl p-6 shadow-sm overflow-x-auto"
            >
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Database className="h-5 w-5 text-indigo-400" />
                Column Statistics
              </h3>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-800 text-zinc-400">
                    {[
                      { key: "name", label: "Column" },
                      { key: "type", label: "Type" },
                      { key: "unique_count", label: "Unique" },
                      { key: "null_count", label: "Missing" },
                      { key: "null_pct", label: "Miss %" },
                      { key: "mean", label: "Mean" },
                      { key: "std", label: "Std" },
                      { key: "min", label: "Min" },
                      { key: "max", label: "Max" },
                      { key: "skewness", label: "Skew" },
                    ].map(h => (
                      <th
                        key={h.key}
                        onClick={() => toggleSort(h.key)}
                        className="py-3 px-3 text-left cursor-pointer hover:text-white transition-colors select-none"
                      >
                        <span className="flex items-center gap-1">
                          {h.label}
                          <ArrowUpDown className="h-3 w-3 opacity-40" />
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sortedStats.map((col: any, i: number) => (
                    <tr key={col.name} className="border-b border-zinc-800/50 hover:bg-zinc-800/30 transition-colors">
                      <td className="py-2.5 px-3 text-white font-medium">{col.name}</td>
                      <td className="py-2.5 px-3">
                        <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full ${
                          col.type === "numeric"
                            ? "bg-blue-500/10 text-blue-400"
                            : "bg-violet-500/10 text-violet-400"
                        }`}>
                          {col.type === "numeric" ? <Hash className="h-3 w-3" /> : <Type className="h-3 w-3" />}
                          {col.type}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-zinc-300">{col.unique_count}</td>
                      <td className="py-2.5 px-3 text-zinc-300">{col.null_count}</td>
                      <td className="py-2.5 px-3">
                        <span className={col.null_pct > 20 ? "text-red-400 font-medium" : "text-zinc-300"}>
                          {col.null_pct}%
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-zinc-300 font-mono text-xs">{col.mean ?? "—"}</td>
                      <td className="py-2.5 px-3 text-zinc-300 font-mono text-xs">{col.std ?? "—"}</td>
                      <td className="py-2.5 px-3 text-zinc-300 font-mono text-xs">{col.min ?? "—"}</td>
                      <td className="py-2.5 px-3 text-zinc-300 font-mono text-xs">{col.max ?? "—"}</td>
                      <td className="py-2.5 px-3 text-zinc-300 font-mono text-xs">{col.skewness ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </motion.div>

            {/* Distribution Histograms */}
            {data.histograms.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="space-y-4"
              >
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-emerald-400" />
                  Numeric Distributions
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                  {data.histograms.map((h: any, idx: number) => {
                    const chartData = h.counts.map((count: number, i: number) => ({
                      bin: `${h.bins[i].toFixed(1)}`,
                      count,
                    }));
                    return (
                      <div
                        key={h.column}
                        className="bg-card/40 backdrop-blur-md border border-border/50 rounded-xl p-5 shadow-sm"
                      >
                        <p className="text-sm font-semibold text-white mb-3 truncate">{h.column}</p>
                        <div className="h-[180px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: -15 }}>
                              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#3f3f46" />
                              <XAxis
                                dataKey="bin"
                                tick={{ fill: "#a1a1aa", fontSize: 9 }}
                                interval={Math.max(0, Math.floor(chartData.length / 5) - 1)}
                              />
                              <YAxis tick={{ fill: "#a1a1aa", fontSize: 10 }} />
                              <Tooltip
                                contentStyle={{ backgroundColor: "#18181b", border: "1px solid #3f3f46", borderRadius: 8 }}
                                labelStyle={{ color: "#fff" }}
                                itemStyle={{ color: "#a78bfa" }}
                              />
                              <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                                {chartData.map((_: any, i: number) => (
                                  <Cell key={i} fill={COLORS[idx % COLORS.length]} fillOpacity={0.8} />
                                ))}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </motion.div>
            )}

            {/* Categorical Value Counts */}
            {data.categorical_counts.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="space-y-4"
              >
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Type className="h-5 w-5 text-violet-400" />
                  Categorical Value Counts
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                  {data.categorical_counts.map((cat: any, idx: number) => (
                    <div
                      key={cat.column}
                      className="bg-card/40 backdrop-blur-md border border-border/50 rounded-xl p-5 shadow-sm"
                    >
                      <p className="text-sm font-semibold text-white mb-3 truncate">{cat.column}</p>
                      <div className="h-[180px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={cat.values}
                            layout="vertical"
                            margin={{ top: 5, right: 15, bottom: 5, left: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#3f3f46" />
                            <XAxis type="number" tick={{ fill: "#a1a1aa", fontSize: 10 }} />
                            <YAxis
                              type="category"
                              dataKey="value"
                              tick={{ fill: "#e4e4e7", fontSize: 11 }}
                              width={80}
                            />
                            <Tooltip
                              contentStyle={{ backgroundColor: "#18181b", border: "1px solid #3f3f46", borderRadius: 8 }}
                              labelStyle={{ color: "#fff" }}
                            />
                            <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                              {cat.values.map((_: any, i: number) => (
                                <Cell key={i} fill={COLORS[(idx + i) % COLORS.length]} fillOpacity={0.8} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </>
        ) : null}
      </div>
    </AppShell>
  );
}
