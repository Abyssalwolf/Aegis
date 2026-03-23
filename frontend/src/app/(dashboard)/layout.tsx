import { getAccessToken } from '@/lib/auth';
import { redirect } from 'next/navigation';
import { Shield, LogOut } from 'lucide-react';
import { logoutAction } from '@/lib/auth';
import { UserProfileActions } from '@/components/auth/UserProfileActions';
import { SidebarNav } from '@/components/navigation/SidebarNav';
import { MobileDashboardHeader } from '@/components/navigation/MobileDashboardHeader';

export default async function DashboardLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    const token = await getAccessToken();
    if (!token) {
        redirect('/');
    }

    // To build an adaptive sleek sidebar, we'd normally know rank and role here
    // For aesthetics, we'll keep a generic top-bar/sidebar fusion
    return (
        <div className="min-h-screen bg-background text-foreground flex flex-col md:flex-row">
            <MobileDashboardHeader token={token} />

            {/* Sidebar Desktop */}
            <aside className="hidden w-64 md:flex flex-col border-r border-border bg-card/50 backdrop-blur-xl">
                <div className="p-6 flex items-center gap-3">
                    <div className="p-2 bg-primary/20 rounded-lg border border-primary/30">
                        <Shield className="w-6 h-6 text-primary shadow-glow" />
                    </div>
                    <span className="font-bold text-xl tracking-tight text-white drop-shadow-md">AEGIS</span>
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

            {/* Main Content Area */}
            <main className="flex-1 overflow-y-auto relative bg-background">
                <div className="absolute top-0 left-0 right-0 h-64 bg-gradient-to-b from-primary/5 to-transparent pointer-events-none z-0" />
                <div className="relative z-10 p-4 md:p-8 lg:p-12 max-w-7xl mx-auto h-full space-y-8 animate-fade-in-up">
                    {children}
                </div>
            </main>
        </div>
    );
}
