"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface ConfidenceBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

interface ConfidenceDistributionProps {
  data: ConfidenceBin[];
}

export function ConfidenceDistribution({ data }: ConfidenceDistributionProps) {
  if (!data || data.length === 0) {
    return (
      <div className="flex h-full w-full items-center justify-center text-sm text-zinc-500">
        Confidence distribution not available for this model.
      </div>
    );
  }

  // Format data for chart
  const chartData = data.map(d => ({
    name: `${(d.bin_start * 100).toFixed(0)}%-${(d.bin_end * 100).toFixed(0)}%`,
    count: d.count,
    midpoint: (d.bin_start + d.bin_end) / 2
  }));

  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={chartData}
          margin={{ top: 10, right: 30, left: 0, bottom: 20 }}
        >
          <defs>
            <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#3f3f46" />
          <XAxis 
            dataKey="name" 
            stroke="#a1a1aa" 
            fontSize={12}
            tickLine={false}
            axisLine={false}
            minTickGap={20}
          />
          <YAxis 
            stroke="#a1a1aa" 
            fontSize={12}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => `${value}`}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                return (
                  <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-3 shadow-xl">
                    <p className="mb-1 text-sm font-medium text-zinc-300">
                      Confidence Range: {label}
                    </p>
                    <p className="text-lg font-bold text-violet-400">
                      {payload[0].value} predictions
                    </p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Area 
            type="monotone" 
            dataKey="count" 
            stroke="#8b5cf6" 
            strokeWidth={3}
            fillOpacity={1} 
            fill="url(#colorCount)" 
            animationDuration={1500}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
