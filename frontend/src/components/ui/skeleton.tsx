import * as React from "react";
import { cn } from "@/lib/utils";

const Skeleton = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "rounded-md bg-background-tertiary animate-shimmer",
      className
    )}
    style={{
      backgroundImage:
        "linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.04) 50%, transparent 100%)",
      backgroundSize: "200% 100%",
    }}
    {...props}
  />
));
Skeleton.displayName = "Skeleton";

export { Skeleton };
