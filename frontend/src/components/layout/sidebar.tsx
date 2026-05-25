"use client";

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  type ReactNode,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  Brain,
  LayoutDashboard,
  Upload,
  Cpu,
  Lightbulb,
  SlidersHorizontal,
  Shield,
  ChevronLeft,
  ChevronRight,
  Settings,
} from "lucide-react";

// ─── Sidebar Context ────────────────────────────────────────────────

interface SidebarContextValue {
  collapsed: boolean;
  toggle: () => void;
  activeItem: string;
  setActiveItem: (id: string) => void;
}

const SidebarContext = createContext<SidebarContextValue | null>(null);

export function useSidebar() {
  const ctx = useContext(SidebarContext);
  if (!ctx) {
    throw new Error("useSidebar must be used within a SidebarProvider");
  }
  return ctx;
}

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [activeItem, setActiveItem] = useState("dashboard");
  const toggle = useCallback(() => setCollapsed((prev) => !prev), []);

  return (
    <SidebarContext.Provider value={{ collapsed, toggle, activeItem, setActiveItem }}>
      {children}
    </SidebarContext.Provider>
  );
}

// ─── Navigation Config ──────────────────────────────────────────────

interface NavItem {
  id: string;
  label: string;
  icon: React.ElementType;
}

const NAV_ITEMS: NavItem[] = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "upload", label: "Upload Data", icon: Upload },
  { id: "training", label: "Model Training", icon: Cpu },
  { id: "explanations", label: "Explanations", icon: Lightbulb },
  { id: "whatif", label: "What-If Analysis", icon: SlidersHorizontal },
  { id: "bias", label: "Bias Detection", icon: Shield },
];

// ─── Animation Variants ─────────────────────────────────────────────

const sidebarVariants = {
  expanded: { width: 280 },
  collapsed: { width: 72 },
};

const labelVariants = {
  show: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.2, delay: 0.05 },
  },
  hide: {
    opacity: 0,
    x: -8,
    transition: { duration: 0.15 },
  },
};

// ─── Sidebar Component ──────────────────────────────────────────────

export default function Sidebar() {
  const { collapsed, toggle, activeItem, setActiveItem } = useSidebar();

  return (
    <motion.aside
      variants={sidebarVariants}
      animate={collapsed ? "collapsed" : "expanded"}
      initial={false}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
      className={cn(
        "fixed left-0 top-0 z-40 flex h-screen flex-col",
        "bg-sidebar border-r border-border",
        "overflow-hidden"
      )}
    >
      {/* ── Logo Section ──────────────────────────────────────── */}
      <div className="flex h-16 shrink-0 items-center gap-3 px-5">
        <div className="relative flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/15">
          <Brain className="h-5 w-5 text-primary" />
          {/* Sparkle glow */}
          <div className="absolute inset-0 rounded-lg bg-primary/10 blur-sm" />
        </div>
        <AnimatePresence mode="wait">
          {!collapsed && (
            <motion.span
              key="logo-text"
              variants={labelVariants}
              initial="hide"
              animate="show"
              exit="hide"
              className="whitespace-nowrap text-lg font-bold tracking-tight text-foreground"
            >
              Insight
              <span className="text-primary">X</span>
            </motion.span>
          )}
        </AnimatePresence>
      </div>

      {/* ── Separator ─────────────────────────────────────────── */}
      <div className="mx-4 h-px bg-border" />

      {/* ── Navigation Items ──────────────────────────────────── */}
      <nav className="mt-4 flex flex-1 flex-col gap-1 px-3">
        {NAV_ITEMS.map((item) => {
          const isActive = activeItem === item.id;
          const Icon = item.icon;

          return (
            <button
              key={item.id}
              onClick={() => setActiveItem(item.id)}
              title={collapsed ? item.label : undefined}
              className={cn(
                "group relative flex h-11 items-center gap-3 rounded-lg px-3",
                "transition-colors duration-200 cursor-pointer",
                "text-foreground-subtle hover:bg-sidebar-hover hover:text-foreground",
                isActive && [
                  "bg-sidebar-active text-foreground",
                  "hover:bg-sidebar-active",
                ]
              )}
            >
              {/* Active indicator — left accent bar */}
              {isActive && (
                <motion.div
                  layoutId="sidebar-active-indicator"
                  className="absolute left-0 top-1.5 bottom-1.5 w-[3px] rounded-full bg-primary"
                  transition={{ type: "spring", stiffness: 350, damping: 30 }}
                />
              )}

              {/* Active glow */}
              {isActive && (
                <div className="absolute inset-0 rounded-lg bg-primary/5 blur-sm pointer-events-none" />
              )}

              <Icon
                className={cn(
                  "h-5 w-5 shrink-0 transition-colors duration-200",
                  isActive
                    ? "text-primary"
                    : "text-foreground-subtle group-hover:text-foreground"
                )}
              />

              <AnimatePresence mode="wait">
                {!collapsed && (
                  <motion.span
                    key={`label-${item.id}`}
                    variants={labelVariants}
                    initial="hide"
                    animate="show"
                    exit="hide"
                    className={cn(
                      "whitespace-nowrap text-sm font-medium",
                      isActive && "font-semibold"
                    )}
                  >
                    {item.label}
                  </motion.span>
                )}
              </AnimatePresence>
            </button>
          );
        })}
      </nav>

      {/* ── Bottom Section ────────────────────────────────────── */}
      <div className="mt-auto flex flex-col gap-1 px-3 pb-3">
        <div className="mx-1 mb-2 h-px bg-border" />

        {/* Settings */}
        <button
          title={collapsed ? "Settings" : undefined}
          className={cn(
            "group flex h-11 items-center gap-3 rounded-lg px-3",
            "text-foreground-subtle transition-colors duration-200",
            "hover:bg-sidebar-hover hover:text-foreground cursor-pointer"
          )}
        >
          <Settings className="h-5 w-5 shrink-0 transition-colors duration-200 group-hover:text-foreground" />
          <AnimatePresence mode="wait">
            {!collapsed && (
              <motion.span
                key="settings-label"
                variants={labelVariants}
                initial="hide"
                animate="show"
                exit="hide"
                className="whitespace-nowrap text-sm font-medium"
              >
                Settings
              </motion.span>
            )}
          </AnimatePresence>
        </button>

        {/* Version Tag */}
        <AnimatePresence mode="wait">
          {!collapsed && (
            <motion.div
              key="version-tag"
              variants={labelVariants}
              initial="hide"
              animate="show"
              exit="hide"
              className="px-3 pb-1"
            >
              <span className="text-[11px] font-mono text-foreground-subtle/50">
                v1.0.0
              </span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Toggle Button */}
        <button
          onClick={toggle}
          className={cn(
            "flex h-10 items-center justify-center rounded-lg",
            "text-foreground-subtle transition-all duration-200",
            "hover:bg-sidebar-hover hover:text-foreground cursor-pointer",
            collapsed ? "mx-0" : "mx-0"
          )}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <motion.div
            animate={{ rotate: collapsed ? 180 : 0 }}
            transition={{ duration: 0.3 }}
          >
            <ChevronLeft className="h-4 w-4" />
          </motion.div>
        </button>
      </div>
    </motion.aside>
  );
}
