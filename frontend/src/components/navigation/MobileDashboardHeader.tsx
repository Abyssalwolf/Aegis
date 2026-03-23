'use client';

import { useEffect, useState } from 'react';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { Shield, LogOut, Menu, X } from 'lucide-react';
import { logoutAction } from '@/lib/auth';
import { UserProfileActions } from '@/components/auth/UserProfileActions';
import { SidebarNav } from '@/components/navigation/SidebarNav';
import { cn } from '@/lib/utils';
import { getApiV1Url } from '@/lib/api';

export function MobileDashboardHeader({ token }: { token: string }) {
    const [open, setOpen] = useState(false);
    const [homeHref, setHomeHref] = useState('/officer/dashboard');
    const pathname = usePathname();

    useEffect(() => {
        setOpen(false);
    }, [pathname]);

    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const r = await fetch(`${getApiV1Url()}/officer/me`, {
                    headers: { Authorization: `Bearer ${token}` },
                    cache: 'no-store',
                });
                if (!cancelled && !r.ok) setHomeHref('/admin/dashboard');
            } catch {
                if (!cancelled) setHomeHref('/admin/dashboard');
            }
        })();
        return () => {
            cancelled = true;
        };
    }, [token]);

    return (
        <>
            <div className="md:hidden flex items-center justify-between p-4 border-b border-border bg-card">
                <Link href={homeHref} className="flex items-center gap-2 min-w-0">
                    <Shield className="w-6 h-6 text-primary shrink-0" />
                    <span className="font-semibold tracking-tight truncate">AEGIS</span>
                </Link>
                <button
                    type="button"
                    onClick={() => setOpen(true)}
                    className="p-2 bg-muted/50 rounded-md text-muted-foreground hover:text-foreground"
                    aria-label="Open menu"
                >
                    <Menu className="w-5 h-5" />
                </button>
            </div>

            {open ? (
                <>
                    <button
                        type="button"
                        className="fixed inset-0 z-40 bg-black/60 md:hidden"
                        aria-label="Close menu"
                        onClick={() => setOpen(false)}
                    />
                    <aside
                        className={cn(
                            'fixed top-0 left-0 z-50 h-full w-[min(100%,18rem)] flex flex-col',
                            'border-r border-border bg-card shadow-xl md:hidden',
                            'animate-in slide-in-from-left duration-200'
                        )}
                    >
                        <div className="p-4 flex items-center justify-between border-b border-border">
                            <div className="flex items-center gap-2">
                                <div className="p-2 bg-primary/20 rounded-lg border border-primary/30">
                                    <Shield className="w-5 h-5 text-primary" />
                                </div>
                                <span className="font-bold text-lg tracking-tight">AEGIS</span>
                            </div>
                            <button
                                type="button"
                                onClick={() => setOpen(false)}
                                className="p-2 rounded-md hover:bg-muted text-muted-foreground"
                                aria-label="Close menu"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        <SidebarNav />

                        <div className="p-4 mt-auto border-t border-border/50">
                            <UserProfileActions token={token} />
                            <form action={logoutAction}>
                                <button className="flex items-center gap-3 w-full px-3 py-2.5 rounded-md hover:bg-destructive/10 text-destructive text-sm font-medium transition-colors">
                                    <LogOut className="w-4 h-4" />
                                    Sign Out
                                </button>
                            </form>
                        </div>
                    </aside>
                </>
            ) : null}
        </>
    );
}
