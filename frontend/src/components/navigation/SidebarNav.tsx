'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { LayoutDashboard, BrainCircuit, MessageSquare, Video } from 'lucide-react';
import { cn } from '@/lib/utils';

export function SidebarNav() {
    const pathname = usePathname();

    // Check if we are inside a case context: /cases/[id]
    const isCaseRoute = pathname.startsWith('/cases/');

    // Only show these links if we are inside a specific case
    if (!isCaseRoute) {
        return null; // Return empty space for generic dashboards (/officer/dashboard)
    }

    // Extract the caseId from the URL
    // e.g., /cases/123-abc/insights -> paths[2] is "123-abc"
    const paths = pathname.split('/');
    const caseId = paths[2];

    if (!caseId) return null;

    const navItems = [
        { href: `/cases/${caseId}`, label: 'Case Dashboard', icon: LayoutDashboard },
        { href: `/cases/${caseId}/insights`, label: 'Insights', icon: BrainCircuit },
        { href: `/cases/${caseId}/chat`, label: 'AI Investigation', icon: MessageSquare },
        { href: `/cases/${caseId}/cctv`, label: 'CCTV Analysis', icon: Video },
    ];

    return (
        <nav className="flex-1 px-4 space-y-1 mt-6">
            <p className="px-2 text-[10px] font-bold text-muted-foreground uppercase tracking-wider mb-3">Investigation Modules</p>
            {navItems.map((item) => {
                // If the pathname strictly equals the item href, it's the exact tab match
                const isActive = pathname === item.href;
                return (
                    <Link
                        key={item.href}
                        href={item.href}
                        className={cn(
                            "flex items-center gap-3 w-full px-3 py-2.5 rounded-md text-sm font-medium transition-colors",
                            isActive
                                ? "bg-primary/20 text-primary border border-primary/20 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.1)]"
                                : "text-foreground/70 hover:bg-muted hover:text-foreground border border-transparent"
                        )}
                    >
                        <item.icon className={cn("w-4 h-4", isActive ? "text-primary" : "text-muted-foreground")} />
                        {item.label}
                    </Link>
                );
            })}
        </nav>
    );
}
