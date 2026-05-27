"use client";

import { motion } from "framer-motion";

interface ConfusionMatrixProps {
  matrix: number[][];
  labels?: string[];
}

export function ConfusionMatrix({ matrix, labels }: ConfusionMatrixProps) {
  if (!matrix || matrix.length === 0) return null;

  const size = matrix.length;
  const defaultLabels = Array.from({ length: size }, (_, i) => `Class ${i}`);
  const displayLabels = labels && labels.length === size ? labels : defaultLabels;

  // Calculate max value for coloring intensity
  const flatMatrix = matrix.flat();
  const maxValue = Math.max(...flatMatrix, 1); // Avoid division by zero

  const getColor = (val: number, isDiagonal: boolean) => {
    const intensity = val / maxValue;
    if (isDiagonal) {
      // True positives - shades of green/emerald
      return `rgba(16, 185, 129, ${Math.max(0.2, intensity)})`;
    } else {
      // False positives/negatives - shades of red/rose
      return `rgba(244, 63, 94, ${Math.max(0.1, intensity)})`;
    }
  };

  return (
    <div className="flex flex-col items-center p-4">
      <h3 className="text-sm font-medium text-zinc-400 mb-6">Confusion Matrix</h3>
      
      <div className="flex">
        {/* Y-axis label */}
        <div className="flex items-center justify-center -rotate-90 text-xs text-zinc-500 w-8">
          Actual
        </div>
        
        <div>
          {/* Top Labels (Predicted) */}
          <div className="flex mb-2 ml-16">
            {displayLabels.map((label, i) => (
              <div key={`pred-label-${i}`} className="w-16 text-center text-xs text-zinc-400 truncate px-1">
                {label}
              </div>
            ))}
          </div>
          
          {/* Matrix Rows */}
          <div className="flex flex-col gap-1">
            {matrix.map((row, i) => (
              <div key={`row-${i}`} className="flex gap-1 items-center">
                {/* Left Label (Actual) */}
                <div className="w-16 text-right text-xs text-zinc-400 pr-2 truncate">
                  {displayLabels[i]}
                </div>
                
                {/* Cells */}
                {row.map((cellValue, j) => {
                  const isDiagonal = i === j;
                  
                  return (
                    <motion.div
                      key={`cell-${i}-${j}`}
                      initial={{ scale: 0.8, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ delay: (i * size + j) * 0.05 }}
                      className="w-16 h-16 rounded-lg flex flex-col items-center justify-center relative group cursor-default border border-zinc-800/50 transition-colors hover:border-zinc-500"
                      style={{ backgroundColor: getColor(cellValue, isDiagonal) }}
                    >
                      <span className="text-white font-medium text-lg drop-shadow-md">
                        {cellValue}
                      </span>
                      
                      {/* Tooltip */}
                      <div className="absolute bottom-full mb-2 hidden group-hover:flex flex-col items-center z-10 w-max">
                        <div className="bg-zinc-800 text-xs text-white px-3 py-2 rounded-md shadow-xl border border-zinc-700 text-center">
                          <p>Actual: <span className="font-semibold text-primary">{displayLabels[i]}</span></p>
                          <p>Predicted: <span className="font-semibold text-primary">{displayLabels[j]}</span></p>
                          <div className="mt-1 pt-1 border-t border-zinc-700">
                            {isDiagonal ? "True Prediction ✓" : "False Prediction ✗"}
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            ))}
          </div>
          
          {/* X-axis label */}
          <div className="text-center text-xs text-zinc-500 mt-4 ml-16">
            Predicted
          </div>
        </div>
      </div>
    </div>
  );
}
