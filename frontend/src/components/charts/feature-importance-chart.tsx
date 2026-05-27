"use client";

import React, { useMemo } from "react";
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
import { motion } from "framer-motion";

interface FeatureImportanceChartProps {
  data: { feature: string; importance: number }[];
}

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-popover text-popover-foreground rounded-lg border border-border shadow-xl p-3 z-50">
        <p className="font-semibold text-sm mb-1">{data.feature}</p>
        <p className="text-xs text-muted-foreground">
          Importance: <span className="font-medium text-foreground">{data.importance.toFixed(4)}</span>
        </p>
      </div>
    );
  }
  return null;
};

export const FeatureImportanceChart: React.FC<FeatureImportanceChartProps> = ({ data }) => {
  const sortedData = useMemo(() => {
    return [...data].sort((a, b) => a.importance - b.importance);
  }, [data]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="w-full h-full min-h-[450px] relative rounded-xl border border-border/50 bg-card/40 backdrop-blur-md p-4 shadow-sm"
    >
      <div className="mb-4">
        <h3 className="text-lg font-semibold tracking-tight text-foreground">
          Global Feature Importance
        </h3>
        <p className="text-sm text-muted-foreground">
          Features that have the most impact on the model's predictions overall.
        </p>
      </div>

      <div className="w-full h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={sortedData}
            margin={{ top: 10, right: 30, left: 20, bottom: 5 }}
          >
            <defs>
              <linearGradient id="colorImportance" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity={0.8} />
                <stop offset="100%" stopColor="hsl(var(--accent))" stopOpacity={0.8} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="hsl(var(--border))" opacity={0.5} />
            <XAxis 
              type="number" 
              tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
              axisLine={{ stroke: "hsl(var(--border))" }}
              tickLine={{ stroke: "hsl(var(--border))" }}
            />
            <YAxis 
              dataKey="feature" 
              type="category" 
              width={100}
              tick={{ fill: "hsl(var(--foreground))", fontSize: 12 }}
              axisLine={{ stroke: "hsl(var(--border))" }}
              tickLine={false}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: "hsl(var(--muted)/0.3)" }} />
            <Bar
              dataKey="importance"
              fill="url(#colorImportance)"
              radius={[0, 4, 4, 0]}
              animationDuration={1500}
              animationEasing="ease-out"
            >
              {sortedData.map((entry, index) => (
                <Cell key={`cell-${index}`} className="transition-all duration-300 hover:opacity-80" />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
};
