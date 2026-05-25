"use client";

import React, { useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { PredictionExplanation } from "@/types";
import { cn } from "@/lib/utils";
import { TrendingUp, TrendingDown, HelpCircle, Activity } from "lucide-react";

interface PredictionExplanationProps {
  explanation: PredictionExplanation;
  className?: string;
}

export const PredictionExplanationView: React.FC<PredictionExplanationProps> = ({ 
  explanation,
  className
}) => {
  const { prediction, probability, shap_values, lime_explanation } = explanation;

  // Sort SHAP values by absolute magnitude to show most impactful first
  const sortedShapValues = useMemo(() => {
    if (!shap_values) return [];
    return [...shap_values].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
  }, [shap_values]);

  const positiveShap = sortedShapValues.filter(s => s.shap_value > 0);
  const negativeShap = sortedShapValues.filter(s => s.shap_value < 0);

  const formatValue = (val: number | string) => {
    if (typeof val === 'number') {
      return Number.isInteger(val) ? val.toString() : val.toFixed(4);
    }
    return val;
  };

  return (
    <div className={cn("w-full space-y-6", className)}>
      <motion.div 
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col md:flex-row items-center justify-between gap-4 p-5 rounded-2xl bg-card border border-border/60 shadow-lg relative overflow-hidden"
      >
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-transparent pointer-events-none" />
        
        <div className="relative z-10 flex flex-col">
          <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-1">Prediction</span>
          <div className="flex items-baseline gap-3">
            <span className="text-4xl font-bold tracking-tight text-foreground">{formatValue(prediction)}</span>
            {probability !== undefined && (
              <span className="text-sm font-medium px-2.5 py-1 rounded-full bg-primary/10 text-primary border border-primary/20">
                {(probability * 100).toFixed(1)}% Confidence
              </span>
            )}
          </div>
        </div>
        
        <div className="relative z-10 w-full md:w-auto flex items-center justify-center p-3 rounded-xl bg-muted/30 border border-border/40 backdrop-blur-sm">
          <Activity className="w-8 h-8 text-primary/70" />
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* SHAP Explanation */}
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="flex flex-col rounded-xl border border-border/50 bg-card p-5 shadow-sm"
        >
          <div className="flex items-center gap-2 mb-4 pb-3 border-b border-border/40">
            <div className="p-1.5 rounded-md bg-teal-500/10 text-teal-500">
              <TrendingUp className="w-4 h-4" />
            </div>
            <h3 className="font-semibold text-foreground">SHAP Contributions</h3>
          </div>
          
          <div className="space-y-4">
            {sortedShapValues.length > 0 ? (
              <div className="flex flex-col gap-3">
                {sortedShapValues.slice(0, 8).map((item, idx) => {
                  const isPositive = item.shap_value > 0;
                  const maxAbs = Math.max(...sortedShapValues.map(s => Math.abs(s.shap_value)));
                  const percentWidth = Math.max(5, (Math.abs(item.shap_value) / maxAbs) * 100);
                  
                  return (
                    <motion.div 
                      key={item.feature}
                      initial={{ opacity: 0, x: isPositive ? -10 : 10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.2 + idx * 0.05 }}
                      className="group flex items-center gap-3 text-sm"
                    >
                      <div className="w-1/3 truncate font-medium text-foreground/80 group-hover:text-foreground transition-colors" title={item.feature}>
                        {item.feature}
                      </div>
                      
                      <div className="flex-1 flex items-center relative h-6 rounded-full bg-muted/30 overflow-hidden border border-border/50">
                        {/* Center dividing line */}
                        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-border z-10" />
                        
                        <div className="flex-1 h-full relative flex items-center justify-end pr-1">
                          {!isPositive && (
                            <motion.div 
                              initial={{ width: 0 }}
                              animate={{ width: `${percentWidth}%` }}
                              className="h-4 bg-gradient-to-l from-rose-500/80 to-rose-400/80 rounded-l-md shadow-[0_0_10px_rgba(244,63,94,0.3)]"
                            />
                          )}
                        </div>
                        <div className="flex-1 h-full relative flex items-center justify-start pl-1">
                          {isPositive && (
                            <motion.div 
                              initial={{ width: 0 }}
                              animate={{ width: `${percentWidth}%` }}
                              className="h-4 bg-gradient-to-r from-teal-500/80 to-teal-400/80 rounded-r-md shadow-[0_0_10px_rgba(20,184,166,0.3)]"
                            />
                          )}
                        </div>
                      </div>
                      
                      <div className={cn(
                        "w-16 text-right font-mono text-xs font-semibold",
                        isPositive ? "text-teal-500" : "text-rose-500"
                      )}>
                        {isPositive ? "+" : ""}{item.shap_value.toFixed(3)}
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            ) : (
              <div className="py-8 text-center text-muted-foreground flex flex-col items-center gap-2">
                <HelpCircle className="w-8 h-8 opacity-20" />
                <p>No SHAP explanations available for this prediction.</p>
              </div>
            )}
          </div>
        </motion.div>

        {/* LIME Explanation */}
        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="flex flex-col rounded-xl border border-border/50 bg-card p-5 shadow-sm"
        >
          <div className="flex items-center gap-2 mb-4 pb-3 border-b border-border/40">
            <div className="p-1.5 rounded-md bg-indigo-500/10 text-indigo-500">
              <Activity className="w-4 h-4" />
            </div>
            <h3 className="font-semibold text-foreground">LIME Local Weights</h3>
          </div>

          <div className="space-y-4">
            {lime_explanation && lime_explanation.length > 0 ? (
              <div className="flex flex-col gap-3">
                {lime_explanation.map((item, idx) => {
                  const isPositive = item.weight > 0;
                  
                  return (
                    <motion.div 
                      key={`${item.feature}-${idx}`}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 + idx * 0.05 }}
                      className="p-3 rounded-lg border border-border/50 bg-muted/20 hover:bg-muted/40 transition-colors flex items-center justify-between"
                    >
                      <div className="flex flex-col max-w-[70%]">
                        <span className="text-xs text-muted-foreground mb-0.5">{item.feature}</span>
                        <span className="text-sm font-medium text-foreground truncate" title={item.condition}>
                          {item.condition}
                        </span>
                      </div>
                      
                      <div className={cn(
                        "flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold font-mono",
                        isPositive ? "bg-teal-500/10 text-teal-600 dark:text-teal-400" : "bg-rose-500/10 text-rose-600 dark:text-rose-400"
                      )}>
                        {isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                        {isPositive ? "+" : ""}{item.weight.toFixed(4)}
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            ) : (
              <div className="py-8 text-center text-muted-foreground flex flex-col items-center gap-2">
                <HelpCircle className="w-8 h-8 opacity-20" />
                <p>No LIME explanations available for this prediction.</p>
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  );
};
