import * as React from "react"
import { cn } from "@/lib/utils"

export interface ButtonProps
    extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: "default" | "outline" | "ghost"
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant = "default", ...props }, ref) => {
        return (
            <button
                ref={ref}
                className={cn(
                    "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
                    "h-10 px-4 py-2",
                    variant === "default" && "bg-primary text-primary-foreground hover:bg-primary/90 hover:shadow-md hover:scale-[1.02]",
                    variant === "outline" && "border border-border bg-background hover:bg-muted/30 hover:text-accent-foreground",
                    variant === "ghost" && "hover:bg-muted/50 hover:text-accent-foreground",
                    className
                )}
                {...props}
            />
        )
    }
)
Button.displayName = "Button"

export { Button }
