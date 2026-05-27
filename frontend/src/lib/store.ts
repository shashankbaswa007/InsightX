import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { DatasetMeta, TrainedModel } from "@/types";

interface AppState {
  dataset: DatasetMeta | null;
  model: TrainedModel | null;
  leaderboard: TrainedModel[];
  setDataset: (dataset: DatasetMeta | null) => void;
  setModel: (model: TrainedModel | null) => void;
  setLeaderboard: (leaderboard: TrainedModel[]) => void;
  reset: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      dataset: null,
      model: null,
      leaderboard: [],
      setDataset: (dataset) => set({ dataset }),
      setModel: (model) => set({ model }),
      setLeaderboard: (leaderboard) => set({ leaderboard }),
      reset: () => set({ dataset: null, model: null, leaderboard: [] }),
    }),
    {
      name: "insightx-store", // Key used in localStorage
    }
  )
);
