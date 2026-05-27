"use client";

import { useMemo } from "react";
import { motion } from "framer-motion";

interface HeatmapData {
  features: string[];
  data: { x: string; y: string; value: number }[];
}

export function CorrelationHeatmap({ data }: { data: HeatmapData }) {
  const { features, data: gridData } = data;

  // Helper to map correlation (-1 to 1) to a color
  const getColor = (val: number) => {
    // Red for negative, Blue for positive
    if (val < 0) {
      const intensity = Math.abs(val);
      return `rgba(239, 68, 68, ${intensity})`; // Tailwind red-500
    } else {
      return `rgba(59, 130, 246, ${val})`; // Tailwind blue-500
    }
  };

  if (!features || features.length === 0) {
    return <div className="p-8 text-center text-zinc-500">No numeric features found for correlation.</div>;
  }

  // To keep it readable, limit to top 15 features if there are too many
  const displayFeatures = features.slice(0, 15);

  return (
    <div className="w-full overflow-x-auto pb-4">
      <div className="min-w-[600px]">
        {/* Top Header Row */}
        <div className="flex">
          <div className="w-32 shrink-0"></div>
          {displayFeatures.map((f) => (
            <div key={`header-${f}`} className="w-12 shrink-0 -rotate-45 transform origin-bottom-left text-xs text-zinc-400 truncate pr-2 flex items-end">
              {f}
            </div>
          ))}
        </div>

        {/* Grid Body */}
        <div className="mt-2 flex flex-col gap-1">
          {displayFeatures.map((yFeature, i) => (
            <div key={`row-${yFeature}`} className="flex gap-1 items-center">
              {/* Row Header */}
              <div className="w-32 shrink-0 text-right text-xs text-zinc-400 truncate pr-4">
                {yFeature}
              </div>
              
              {/* Cells */}
              {displayFeatures.map((xFeature, j) => {
                const cellData = gridData.find(d => d.x === xFeature && d.y === yFeature);
                const val = cellData ? cellData.value : 0;
                
                return (
                  <motion.div
                    key={`cell-${xFeature}-${yFeature}`}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: (i * 0.05) + (j * 0.05), duration: 0.3 }}
                    className="w-12 h-12 shrink-0 rounded-md relative group flex items-center justify-center cursor-pointer border border-zinc-800/50"
                    style={{ backgroundColor: getColor(val) }}
                  >
                    <span className="text-[10px] text-white/90 opacity-0 group-hover:opacity-100 transition-opacity">
                      {val.toFixed(2)}
                    </span>
                    
                    {/* Tooltip */}
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:flex flex-col items-center z-10 w-max">
                      <div className="bg-zinc-800 text-xs text-white px-3 py-1.5 rounded-md shadow-xl border border-zinc-700">
                        <p><span className="text-zinc-400">X:</span> {xFeature}</p>
                        <p><span className="text-zinc-400">Y:</span> {yFeature}</p>
                        <p className="font-semibold mt-1">Correlation: {val.toFixed(2)}</p>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
