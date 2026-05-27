"use client";

import { motion } from "framer-motion";
import {
  Upload,
  Brain,
  BarChart3,
  Lightbulb,
  SlidersHorizontal,
  Shield,
  ArrowRight,
  Sparkles,
  Database,
  TrendingUp,
  Zap,
  FileUp,
} from "lucide-react";
import AppShell from "@/components/layout/app-shell";
import { useRouter } from "next/navigation";

// ─── Animation Variants ──────────────────────────────────────────

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.08, delayChildren: 0.1 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: "easeOut" as const },
  },
};

const glowVariants = {
  initial: { scale: 1, opacity: 0.5 },
  animate: {
    scale: [1, 1.05, 1],
    opacity: [0.5, 0.8, 0.5],
    transition: { duration: 3, repeat: Infinity, ease: "easeInOut" as const },
  },
};

// ─── Stats Data ──────────────────────────────────────────────────

const stats = [
  {
    label: "Datasets Uploaded",
    value: "0",
    change: "Ready to start",
    icon: Database,
    color: "text-indigo-400",
    bgColor: "bg-indigo-500/10",
    borderColor: "border-indigo-500/20",
  },
  {
    label: "Models Trained",
    value: "0",
    change: "Upload data first",
    icon: Brain,
    color: "text-violet-400",
    bgColor: "bg-violet-500/10",
    borderColor: "border-violet-500/20",
  },
  {
    label: "Explanations Generated",
    value: "0",
    change: "Train a model",
    icon: Lightbulb,
    color: "text-teal-400",
    bgColor: "bg-teal-500/10",
    borderColor: "border-teal-500/20",
  },
  {
    label: "Bias Checks",
    value: "0",
    change: "Ensure fairness",
    icon: Shield,
    color: "text-amber-400",
    bgColor: "bg-amber-500/10",
    borderColor: "border-amber-500/20",
  },
];

// ─── Pipeline Steps ──────────────────────────────────────────────

const pipelineSteps = [
  {
    step: "01",
    title: "Upload Dataset",
    description: "Drag & drop your CSV or JSON file to get started",
    icon: FileUp,
    status: "ready" as const,
    gradient: "from-indigo-500 to-violet-500",
  },
  {
    step: "02",
    title: "Configure & Train",
    description: "Select target variable and train a baseline model",
    icon: Brain,
    status: "locked" as const,
    gradient: "from-violet-500 to-purple-500",
  },
  {
    step: "03",
    title: "Explore Explanations",
    description: "Understand predictions with SHAP & LIME visualizations",
    icon: Lightbulb,
    status: "locked" as const,
    gradient: "from-purple-500 to-pink-500",
  },
  {
    step: "04",
    title: "What-If Analysis",
    description: "Simulate changes and see real-time prediction shifts",
    icon: SlidersHorizontal,
    status: "locked" as const,
    gradient: "from-pink-500 to-rose-500",
  },
  {
    step: "05",
    title: "Bias Detection",
    description: "Flag protected attributes and evaluate model fairness",
    icon: Shield,
    status: "locked" as const,
    gradient: "from-rose-500 to-amber-500",
  },
];

// ─── Quick Actions ───────────────────────────────────────────────

const quickActions = [
  {
    title: "Upload CSV / JSON",
    description: "Start your ML pipeline with a dataset",
    icon: Upload,
    gradient: "from-indigo-500 to-violet-600",
    href: "/upload",
  },
  {
    title: "View Metrics",
    description: "Accuracy, F1, RMSE and more",
    icon: BarChart3,
    gradient: "from-teal-500 to-emerald-600",
    href: "/training",
  },
  {
    title: "Feature Importance",
    description: "SHAP summary plots & global explanations",
    icon: TrendingUp,
    gradient: "from-violet-500 to-purple-600",
    href: "/explanations",
  },
];

// ─── Page Component ──────────────────────────────────────────────

export default function DashboardPage() {
  const router = useRouter();

  return (
    <AppShell>
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="space-y-8"
      >
        {/* ── Hero Banner ──────────────────────────────────────── */}
        <motion.div variants={itemVariants} className="relative overflow-hidden rounded-3xl glass animated-border p-1">
          {/* Animated glow */}
          <motion.div
            variants={glowVariants}
            initial="initial"
            animate="animate"
            className="absolute -top-24 -right-24 h-64 w-64 rounded-full bg-indigo-500/20 blur-3xl"
          />
          <motion.div
            variants={glowVariants}
            initial="initial"
            animate="animate"
            className="absolute -bottom-16 -left-16 h-48 w-48 rounded-full bg-violet-500/15 blur-3xl"
            style={{ animationDelay: "1.5s" }}
          />

          <div className="relative z-10 flex items-center justify-between px-8 py-10">
            <div className="max-w-2xl space-y-4">
              <div className="inline-flex items-center gap-2 rounded-full border border-indigo-500/30 bg-indigo-500/10 px-4 py-1.5 text-sm text-indigo-300">
                <Sparkles className="h-3.5 w-3.5" />
                Explainable AI Platform
              </div>
              <h1 className="text-4xl font-bold tracking-tight text-white">
                Welcome to{" "}
                <span className="bg-gradient-to-r from-indigo-400 via-violet-400 to-purple-400 bg-clip-text text-transparent">
                  InsightX AI
                </span>
              </h1>
              <p className="text-lg leading-relaxed text-zinc-400">
                Upload datasets, train models, and discover{" "}
                <span className="text-zinc-200">why</span> your AI makes
                decisions. Powered by SHAP, LIME, and Fairlearn.
              </p>
              <motion.button
                onClick={() => router.push("/upload")}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="mt-2 inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-500/25 transition-shadow hover:shadow-indigo-500/40"
              >
                <Upload className="h-4 w-4" />
                Upload Your First Dataset
                <ArrowRight className="h-4 w-4" />
              </motion.button>
            </div>

            {/* Decorative illustration */}
            <div className="hidden lg:block">
              <div className="relative">
                <div className="grid grid-cols-3 gap-3 opacity-60">
                  {[...Array(9)].map((_, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, scale: 0.5 }}
                      animate={{ opacity: [0.3, 0.7, 0.3], scale: 1 }}
                      transition={{
                        duration: 2 + i * 0.3,
                        repeat: Infinity,
                        delay: i * 0.15,
                      }}
                      className="h-12 w-12 rounded-lg border border-indigo-500/20 bg-indigo-500/5"
                    />
                  ))}
                </div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <Zap className="h-10 w-10 text-indigo-400/80" />
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* ── Stats Grid ───────────────────────────────────────── */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {stats.map((stat, i) => {
            const Icon = stat.icon;
            return (
              <motion.div
                key={stat.label}
                variants={itemVariants}
                whileHover={{ y: -2, transition: { duration: 0.2 } }}
                className={`group relative overflow-hidden rounded-2xl glass-card p-6 transition-all hover:shadow-[0_0_30px_rgba(139,92,246,0.15)]`}
              >
                <div className="flex items-start justify-between">
                  <div className="space-y-2">
                    <p className="text-sm text-zinc-500">{stat.label}</p>
                    <p className="text-3xl font-bold text-white">
                      {stat.value}
                    </p>
                    <p className={`text-xs ${stat.color}`}>{stat.change}</p>
                  </div>
                  <div
                    className={`rounded-lg ${stat.bgColor} p-2.5 transition-transform group-hover:scale-110`}
                  >
                    <Icon className={`h-5 w-5 ${stat.color}`} />
                  </div>
                </div>
                {/* Bottom accent line */}
                <div
                  className={`absolute bottom-0 left-0 h-0.5 w-full ${stat.bgColor} opacity-50`}
                />
              </motion.div>
            );
          })}
        </div>

        {/* ── Quick Actions ────────────────────────────────────── */}
        <motion.div variants={itemVariants}>
          <h2 className="mb-4 text-lg font-semibold text-white">
            Quick Actions
          </h2>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            {quickActions.map((action) => {
              const Icon = action.icon;
              return (
                <motion.button
                  key={action.title}
                  onClick={() => router.push(action.href)}
                  whileHover={{ y: -3, transition: { duration: 0.2 } }}
                  whileTap={{ scale: 0.98 }}
                  className="group relative overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--card)] p-6 text-left transition-all hover:border-[var(--border-hover)]"
                >
                  {/* Gradient overlay on hover */}
                  <div
                    className={`absolute inset-0 bg-gradient-to-br ${action.gradient} opacity-0 transition-opacity group-hover:opacity-5`}
                  />
                  <div className="relative z-10">
                    <div
                      className={`mb-4 inline-flex rounded-lg bg-gradient-to-br ${action.gradient} p-2.5 shadow-lg`}
                    >
                      <Icon className="h-5 w-5 text-white" />
                    </div>
                    <h3 className="text-sm font-semibold text-white">
                      {action.title}
                    </h3>
                    <p className="mt-1 text-xs text-zinc-500">
                      {action.description}
                    </p>
                  </div>
                  <ArrowRight className="absolute right-4 bottom-4 h-4 w-4 text-zinc-600 transition-all group-hover:translate-x-1 group-hover:text-zinc-400" />
                </motion.button>
              );
            })}
          </div>
        </motion.div>

        {/* ── ML Pipeline Steps ────────────────────────────────── */}
        <motion.div variants={itemVariants}>
          <h2 className="mb-4 text-lg font-semibold text-white">
            ML Pipeline
          </h2>
          <div className="relative">
            {/* Connecting line */}
            <div className="absolute left-[27px] top-12 bottom-12 w-px bg-gradient-to-b from-indigo-500/40 via-purple-500/40 to-amber-500/40 hidden md:block" />

            <div className="space-y-3">
              {pipelineSteps.map((step, i) => {
                const Icon = step.icon;
                const isReady = step.status === "ready";
                return (
                  <motion.div
                    key={step.step}
                    variants={itemVariants}
                    whileHover={
                      isReady
                        ? { x: 4, transition: { duration: 0.2 } }
                        : undefined
                    }
                    className={`group relative flex items-center gap-5 rounded-xl border p-4 transition-all ${
                      isReady
                        ? "cursor-pointer border-indigo-500/30 bg-indigo-500/5 hover:border-indigo-500/50 hover:bg-indigo-500/10"
                        : "cursor-default border-[var(--border)] bg-[var(--card)] opacity-50"
                    }`}
                  >
                    {/* Step number circle */}
                    <div
                      className={`flex h-14 w-14 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br ${step.gradient} ${
                        isReady ? "shadow-lg shadow-indigo-500/20" : "opacity-40"
                      }`}
                    >
                      <Icon className="h-6 w-6 text-white" />
                    </div>

                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono text-zinc-600">
                          {step.step}
                        </span>
                        <h3
                          className={`text-sm font-semibold ${
                            isReady ? "text-white" : "text-zinc-500"
                          }`}
                        >
                          {step.title}
                        </h3>
                        {isReady && (
                          <span className="inline-flex items-center rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-400 ring-1 ring-emerald-500/20">
                            Ready
                          </span>
                        )}
                      </div>
                      <p
                        className={`mt-0.5 text-xs ${
                          isReady ? "text-zinc-400" : "text-zinc-600"
                        }`}
                      >
                        {step.description}
                      </p>
                    </div>

                    {isReady && (
                      <ArrowRight className="h-4 w-4 shrink-0 text-indigo-400 transition-transform group-hover:translate-x-1" />
                    )}
                  </motion.div>
                );
              })}
            </div>
          </div>
        </motion.div>

        {/* ── Footer ───────────────────────────────────────────── */}
        <motion.div
          variants={itemVariants}
          className="flex items-center justify-between rounded-xl border border-[var(--border)] bg-[var(--card)] px-6 py-4 text-xs text-zinc-600"
        >
          <span>InsightX AI v1.0.0 — Open Source · $0 Budget</span>
          <span className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
            All systems operational
          </span>
        </motion.div>
      </motion.div>
    </AppShell>
  );
}
