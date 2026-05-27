"use client";

import { cn } from "@/lib/utils";
import { Search, Bell, ChevronDown, Menu } from "lucide-react";
import { useSidebar } from "@/components/layout/sidebar";

// ─── Top Bar Component ──────────────────────────────────────────────

interface TopBarProps {
  pageTitle: string;
}

export default function TopBar({ pageTitle }: TopBarProps) {
  const { toggleMobile } = useSidebar();

  return (
    <header
      className={cn(
        "sticky top-0 z-30 flex h-16 shrink-0 items-center justify-between",
        "border-b border-border px-6",
        "glass"
      )}
    >
      {/* ── Left: Page Title & Breadcrumb ─────────────────────── */}
      <div className="flex flex-col justify-center gap-2 md:flex-row md:items-center">
        <button
          onClick={toggleMobile}
          className="md:hidden flex h-9 w-9 items-center justify-center rounded-lg text-foreground-subtle transition-colors hover:bg-background-tertiary hover:text-foreground cursor-pointer"
        >
          <Menu className="h-5 w-5" />
        </button>
        <div className="hidden md:flex flex-col justify-center">
          <div className="flex items-center gap-2 text-xs text-foreground-subtle">
            <span className="opacity-60">InsightX</span>
            <span className="opacity-40">/</span>
            <span>{pageTitle}</span>
          </div>
          <h1 className="text-lg font-semibold tracking-tight text-foreground">
            {pageTitle}
          </h1>
        </div>
      </div>

      {/* ── Right: Actions ────────────────────────────────────── */}
      <div className="flex items-center gap-1">
        {/* Search Button */}
        <button
          className={cn(
            "flex h-9 w-9 items-center justify-center rounded-lg",
            "text-foreground-subtle transition-colors duration-200",
            "hover:bg-background-tertiary hover:text-foreground cursor-pointer"
          )}
          title="Search"
        >
          <Search className="h-[18px] w-[18px]" />
        </button>

        {/* Notification Bell */}
        <button
          className={cn(
            "relative flex h-9 w-9 items-center justify-center rounded-lg",
            "text-foreground-subtle transition-colors duration-200",
            "hover:bg-background-tertiary hover:text-foreground cursor-pointer"
          )}
          title="Notifications"
        >
          <Bell className="h-[18px] w-[18px]" />
          {/* Notification Dot */}
          <span
            className={cn(
              "absolute right-2 top-2 h-2 w-2 rounded-full",
              "bg-primary ring-2 ring-background"
            )}
          />
        </button>

        {/* Separator */}
        <div className="mx-2 h-6 w-px bg-border" />

        {/* User Avatar */}
        <button
          className={cn(
            "flex items-center gap-2 rounded-lg px-2 py-1.5",
            "text-foreground transition-colors duration-200",
            "hover:bg-background-tertiary cursor-pointer"
          )}
          title="Account"
        >
          <div
            className={cn(
              "flex h-8 w-8 items-center justify-center rounded-full",
              "bg-primary/15 text-primary text-xs font-bold",
              "ring-1 ring-primary/20"
            )}
          >
            SB
          </div>
          <ChevronDown className="h-3.5 w-3.5 text-foreground-subtle" />
        </button>
      </div>
    </header>
  );
}
