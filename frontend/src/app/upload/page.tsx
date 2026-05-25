"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Upload, FileType, CheckCircle, AlertCircle, ArrowRight } from "lucide-react";
import { useDropzone } from "react-dropzone";
import AppShell from "@/components/layout/app-shell";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { uploadDataset } from "@/lib/api";
import { useAppStore } from "@/lib/store";

export default function UploadPage() {
  const router = useRouter();
  const setDataset = useAppStore((state) => state.setDataset);
  
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setError(null);
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/csv": [".csv"],
      "application/json": [".json"],
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024, // 50MB
  });

  const handleUpload = async () => {
    if (!file) return;
    
    try {
      setIsUploading(true);
      setError(null);
      const metadata = await uploadDataset(file);
      setDataset(metadata);
      
      // Navigate to training configuration
      router.push("/training");
    } catch (err: any) {
      setError(err.message || "Failed to upload file");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <AppShell>
      <div className="mx-auto max-w-4xl space-y-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">Upload Dataset</h1>
          <p className="mt-2 text-zinc-400">
            Upload your CSV or JSON data to begin training a machine learning model.
          </p>
        </div>

        <Card className="glass-card animated-border overflow-hidden">
          <div className="relative z-10 p-12">
            <div
              {...getRootProps()}
              className={`flex flex-col items-center justify-center space-y-6 rounded-2xl p-12 transition-all duration-300 ${
                isDragActive ? "bg-primary/20 border-primary scale-[1.02]" : "hover:bg-white/5"
              } cursor-pointer border-2 border-dashed border-white/10`}
            >
              <input {...getInputProps()} />
              
              <motion.div 
                whileHover={{ scale: 1.1, rotate: 5 }}
                className="flex h-24 w-24 items-center justify-center rounded-full bg-gradient-to-br from-primary/20 to-accent/20 border border-white/10 shadow-[0_0_30px_rgba(139,92,246,0.3)]"
              >
                <Upload className="h-12 w-12 text-white" />
              </motion.div>
              
              <div className="text-center space-y-3">
                <h3 className="text-2xl font-bold text-white text-glow">
                  {isDragActive ? "Drop to Ignite Pipeline" : "Drag & Drop Dataset Here"}
                </h3>
                <p className="text-base text-zinc-400">
                  Supports CSV or JSON (max 50MB)
                </p>
              </div>
            </div>
          </div>
        </Card>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-3 rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-red-400"
          >
            <AlertCircle className="h-5 w-5" />
            <p className="text-sm font-medium">{error}</p>
          </motion.div>
        )}

        {file && !error && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Card className="glass flex items-center justify-between p-6">
              <div className="flex items-center gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-zinc-800">
                  <FileType className="h-6 w-6 text-emerald-400" />
                </div>
                <div>
                  <p className="font-medium text-white">{file.name}</p>
                  <p className="text-sm text-zinc-400">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
              </div>
              <CheckCircle className="h-5 w-5 text-emerald-400" />
            </Card>

            <div className="mt-8 flex justify-end">
              <Button
                variant="accent"
                size="lg"
                onClick={handleUpload}
                disabled={isUploading}
                className="group"
              >
                {isUploading ? (
                  "Uploading and Analyzing..."
                ) : (
                  <>
                    Continue to Configuration
                    <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
                  </>
                )}
              </Button>
            </div>
          </motion.div>
        )}
      </div>
    </AppShell>
  );
}
