"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { Search, ChevronLeft, ChevronRight } from "lucide-react";

interface DataTableProps {
  columns: string[];
  data: any[];
  onRowClick: (rowIndex: number) => void;
  activeRowIndex?: number;
  className?: string;
}

export const DataTable: React.FC<DataTableProps> = ({
  columns,
  data,
  onRowClick,
  activeRowIndex,
  className
}) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 10;

  // Simple client-side search filtering
  const filteredData = React.useMemo(() => {
    if (!searchTerm) return data;
    const lowerSearch = searchTerm.toLowerCase();
    return data.filter(row => {
      return Object.values(row).some(val => 
        String(val).toLowerCase().includes(lowerSearch)
      );
    });
  }, [data, searchTerm]);

  const totalPages = Math.ceil(filteredData.length / rowsPerPage);
  const currentData = filteredData.slice(
    (currentPage - 1) * rowsPerPage,
    currentPage * rowsPerPage
  );

  return (
    <div className={cn("w-full flex flex-col rounded-xl border border-border bg-card shadow-sm overflow-hidden", className)}>
      {/* Table Header Controls */}
      <div className="p-4 flex flex-col sm:flex-row items-center justify-between gap-4 border-b border-border/50 bg-muted/20">
        <div className="relative w-full sm:w-72">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-4 w-4 text-muted-foreground" />
          </div>
          <input
            type="text"
            placeholder="Search rows..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setCurrentPage(1); // Reset page on search
            }}
            className="w-full pl-9 pr-4 py-2 text-sm rounded-lg border border-border bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
          />
        </div>
        
        <div className="text-sm text-muted-foreground font-medium">
          Showing {filteredData.length > 0 ? (currentPage - 1) * rowsPerPage + 1 : 0} - {Math.min(currentPage * rowsPerPage, filteredData.length)} of {filteredData.length}
        </div>
      </div>

      {/* Table Content */}
      <div className="overflow-x-auto w-full custom-scrollbar">
        <table className="w-full text-sm text-left whitespace-nowrap">
          <thead className="text-xs text-muted-foreground uppercase bg-muted/50 border-b border-border/50 sticky top-0 z-10">
            <tr>
              <th scope="col" className="px-6 py-4 font-semibold tracking-wider">
                #
              </th>
              {columns.map((col, idx) => (
                <th key={idx} scope="col" className="px-6 py-4 font-semibold tracking-wider">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {currentData.length > 0 ? (
              currentData.map((row, idx) => {
                const actualIndex = data.indexOf(row); // Use actual index in original data for the click handler
                const isActive = activeRowIndex === actualIndex;
                
                return (
                  <motion.tr
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.02 }}
                    key={actualIndex}
                    onClick={() => onRowClick(actualIndex)}
                    className={cn(
                      "cursor-pointer border-b border-border/30 transition-all duration-200 hover:bg-muted/40",
                      isActive ? "bg-primary/5 hover:bg-primary/10 border-primary/20 shadow-[inset_4px_0_0_0_hsl(var(--primary))]" : ""
                    )}
                  >
                    <td className="px-6 py-4 font-medium text-muted-foreground">
                      {actualIndex}
                    </td>
                    {columns.map((col, colIdx) => (
                      <td key={colIdx} className={cn(
                        "px-6 py-4",
                        isActive ? "text-foreground font-medium" : "text-foreground/80"
                      )}>
                        {row[col] !== null && row[col] !== undefined ? String(row[col]) : "-"}
                      </td>
                    ))}
                  </motion.tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={columns.length + 1} className="px-6 py-12 text-center text-muted-foreground">
                  No data found matching your search.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Controls */}
      {totalPages > 1 && (
        <div className="p-3 flex items-center justify-between border-t border-border/50 bg-muted/10">
          <button
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="p-2 rounded-md hover:bg-muted text-muted-foreground disabled:opacity-50 disabled:hover:bg-transparent transition-colors"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          
          <div className="flex items-center gap-1">
            {Array.from({ length: Math.min(5, totalPages) }).map((_, i) => {
              // Simple logic to show a window of pages
              let pageNum = currentPage;
              if (currentPage <= 3) pageNum = i + 1;
              else if (currentPage >= totalPages - 2) pageNum = totalPages - 4 + i;
              else pageNum = currentPage - 2 + i;
              
              if (pageNum > 0 && pageNum <= totalPages) {
                return (
                  <button
                    key={pageNum}
                    onClick={() => setCurrentPage(pageNum)}
                    className={cn(
                      "w-8 h-8 flex items-center justify-center rounded-md text-sm font-medium transition-colors",
                      currentPage === pageNum 
                        ? "bg-primary text-primary-foreground shadow-sm" 
                        : "hover:bg-muted text-muted-foreground"
                    )}
                  >
                    {pageNum}
                  </button>
                );
              }
              return null;
            })}
          </div>

          <button
            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
            className="p-2 rounded-md hover:bg-muted text-muted-foreground disabled:opacity-50 disabled:hover:bg-transparent transition-colors"
          >
            <ChevronRight className="w-5 h-5" />
          </button>
        </div>
      )}
    </div>
  );
};
