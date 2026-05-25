"use client";

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all duration-200 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0 cursor-pointer",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground shadow-sm hover:bg-primary-hover hover:shadow-md active:scale-[0.98]",
        secondary:
          "bg-background-tertiary text-foreground border border-border hover:bg-border hover:border-border-hover active:scale-[0.98]",
        outline:
          "border border-border bg-transparent text-foreground hover:bg-background-tertiary hover:border-border-hover active:scale-[0.98]",
        ghost:
          "text-foreground-muted hover:bg-background-tertiary hover:text-foreground",
        destructive:
          "bg-destructive/15 text-destructive border border-destructive/20 hover:bg-destructive/25 hover:border-destructive/40 active:scale-[0.98]",
        accent:
          "bg-accent/15 text-accent border border-accent/20 hover:bg-accent/25 hover:border-accent/40 active:scale-[0.98]",
      },
      size: {
        sm: "h-8 px-3 text-xs rounded-md",
        default: "h-9 px-4 py-2",
        lg: "h-11 px-6 text-base rounded-lg",
        icon: "h-9 w-9 rounded-md",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
