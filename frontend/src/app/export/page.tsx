"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  Download, Code2, Copy, Check, Package, Cpu, Target, Layers, FileDown,
} from "lucide-react";
import AppShell from "@/components/layout/app-shell";
import { useAppStore } from "@/lib/store";
import { getExportSnippet, getDownloadUrl } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";

export default function ExportPage() {
  const router = useRouter();
  const { model, dataset } = useAppStore();

  const [snippetData, setSnippetData] = useState<any | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"python" | "curl">("python");
  const [copied, setCopied] = useState(false);
  const [showAllFeatures, setShowAllFeatures] = useState(false);

  useEffect(() => {
    if (!model) {
      router.push("/training");
      return;
    }

    const load = async () => {
      try {
        setIsLoading(true);
        const res = await getExportSnippet(model.id);
        setSnippetData(res);
      } catch (err: any) {
        setError(err.message || "Failed to load export data");
      } finally {
        setIsLoading(false);
      }
    };

    load();
  }, [model, router]);

  const handleCopy = () => {
    if (!snippetData) return;
    const text = activeTab === "python" ? snippetData.python : snippetData.curl;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!model) return null;

  const metrics = model.metrics;
  const primaryMetric =
    model.task_type === "classification"
      ? `${((metrics.accuracy || 0) * 100).toFixed(1)}% Accuracy`
      : `${(metrics.r2 || 0).toFixed(3)} R²`;

  return (
    <AppShell>
      <div className="mx-auto max-w-5xl space-y-8 pb-16">

        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <Download className="h-8 w-8 text-primary" />
            Export &amp; Deploy
          </h1>
          <p className="mt-2 text-zinc-400 max-w-2xl">
            Download your trained model artifact or integrate it into your pipeline using the generated code snippets.
          </p>
        </div>

        {isLoading ? (
          <div className="space-y-6">
            <Skeleton className="h-40 w-full rounded-xl" />
            <Skeleton className="h-96 w-full rounded-xl" />
          </div>
        ) : error ? (
          <div className="bg-red-500/10 border border-red-500/30 p-4 rounded-lg text-sm text-red-300">{error}</div>
        ) : snippetData ? (
          <>
            {/* Model Summary */}
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gradient-to-br from-primary/15 via-primary/5 to-transparent border border-primary/20 rounded-2xl p-6 shadow-lg"
            >
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                {[
                  { icon: Cpu, label: "Algorithm", value: snippetData.model_type.replace("_", " ") },
                  { icon: Target, label: "Target", value: snippetData.target_column },
                  { icon: Layers, label: "Features", value: snippetData.feature_count },
                  { icon: Package, label: "Performance", value: primaryMetric },
                ].map((item, i) => (
                  <div key={item.label} className="flex items-start gap-3">
                    <div className="p-2 bg-primary/15 rounded-lg">
                      <item.icon className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <p className="text-xs text-zinc-500 uppercase tracking-wider">{item.label}</p>
                      <p className="text-sm font-semibold text-white mt-0.5 capitalize">{item.value}</p>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Download Section */}
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-card/40 backdrop-blur-md border border-border/50 rounded-xl p-6 shadow-sm flex flex-col md:flex-row items-center justify-between gap-6"
            >
              <div>
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <FileDown className="h-5 w-5 text-emerald-400" />
                  Download Model Artifact
                </h3>
                <p className="text-sm text-zinc-400 mt-1">
                  Download the serialized <code className="text-xs bg-zinc-800 px-1.5 py-0.5 rounded">.joblib</code> file. 
                  Load it in any Python environment with <code className="text-xs bg-zinc-800 px-1.5 py-0.5 rounded">joblib.load()</code>.
                </p>
              </div>
              <a
                href={getDownloadUrl(model.id)}
                download
                className="shrink-0 inline-flex items-center gap-2 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold px-6 py-3 rounded-lg transition-colors shadow-md"
              >
                <Download className="h-5 w-5" />
                Download .joblib
              </a>
            </motion.div>

            {/* Code Snippets */}
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-card/40 backdrop-blur-md border border-border/50 rounded-xl shadow-sm overflow-hidden"
            >
              {/* Tabs */}
              <div className="flex border-b border-zinc-800">
                {(["python", "curl"] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`px-6 py-3 text-sm font-medium transition-colors ${
                      activeTab === tab
                        ? "text-primary border-b-2 border-primary bg-primary/5"
                        : "text-zinc-500 hover:text-zinc-300"
                    }`}
                  >
                    {tab === "python" ? "Python" : "cURL"}
                  </button>
                ))}
                <div className="ml-auto flex items-center pr-4">
                  <button
                    onClick={handleCopy}
                    className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-white transition-colors px-3 py-1.5 rounded-md hover:bg-zinc-800"
                  >
                    {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                    {copied ? "Copied!" : "Copy"}
                  </button>
                </div>
              </div>

              {/* Code Block */}
              <div className="p-6 overflow-x-auto">
                <pre className="text-sm text-zinc-300 font-mono leading-relaxed whitespace-pre">
                  <code>{activeTab === "python" ? snippetData.python : snippetData.curl}</code>
                </pre>
              </div>
            </motion.div>

            {/* Feature List */}
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-card/40 backdrop-blur-md border border-border/50 rounded-xl p-6 shadow-sm"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Code2 className="h-5 w-5 text-violet-400" />
                  Expected Features ({snippetData.feature_count})
                </h3>
                {snippetData.feature_count > 10 && (
                  <button
                    onClick={() => setShowAllFeatures(!showAllFeatures)}
                    className="text-xs text-primary hover:underline"
                  >
                    {showAllFeatures ? "Show Less" : "Show All"}
                  </button>
                )}
              </div>
              <div className="flex flex-wrap gap-2">
                {(showAllFeatures
                  ? snippetData.feature_names
                  : snippetData.feature_names.slice(0, 10)
                ).map((f: string) => (
                  <span
                    key={f}
                    className="text-xs bg-zinc-800 border border-zinc-700 text-zinc-300 px-2.5 py-1 rounded-md font-mono"
                  >
                    {f}
                  </span>
                ))}
                {!showAllFeatures && snippetData.feature_count > 10 && (
                  <span className="text-xs text-zinc-500 px-2 py-1">
                    +{snippetData.feature_count - 10} more
                  </span>
                )}
              </div>
            </motion.div>
          </>
        ) : null}
      </div>
    </AppShell>
  );
}
