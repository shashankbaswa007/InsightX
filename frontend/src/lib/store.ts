import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { DatasetMeta, TrainedModel } from "@/types";

interface AppState {
  dataset: DatasetMeta | null;
  model: TrainedModel | null;
  setDataset: (dataset: DatasetMeta | null) => void;
  setModel: (model: TrainedModel | null) => void;
  reset: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      dataset: null,
      model: null,
      setDataset: (dataset) => set({ dataset }),
      setModel: (model) => set({ model }),
      reset: () => set({ dataset: null, model: null }),
    }),
    {
      name: "insightx-store", // Key used in localStorage
    }
  )
);
