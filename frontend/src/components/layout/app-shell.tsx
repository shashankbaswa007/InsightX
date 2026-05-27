"use client";

import { motion } from "framer-motion";
import Sidebar, { SidebarProvider, useSidebar } from "@/components/layout/sidebar";
import TopBar from "@/components/layout/top-bar";

// ─── Inner Shell (needs sidebar context) ────────────────────────────

function ShellInner({ children }: { children: React.ReactNode }) {
  const { collapsed } = useSidebar();

  return (
    <div className="flex min-h-screen">
      <Sidebar />

      {/* ── Main Content Area ─────────────────────────────────── */}
      <motion.div
        animate={{ marginLeft: collapsed ? 72 : 280 }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
        className="flex flex-1 flex-col min-w-0 max-md:!ml-0"
      >
        <TopBar pageTitle="Dashboard" />

        {/* ── Scrollable Content ──────────────────────────────── */}
        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-[1400px] p-6">
            {children}
          </div>
        </main>
      </motion.div>
    </div>
  );
}

// ─── App Shell (wraps with providers) ───────────────────────────────

export default function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <SidebarProvider>
      <ShellInner>{children}</ShellInner>
    </SidebarProvider>
  );
}
