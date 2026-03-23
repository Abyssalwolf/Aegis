import Link from 'next/link';
import { MessageSquare, LayoutDashboard, ArrowRight } from 'lucide-react';

/**
 * Top-level /investigation — AI Investigation (RAG chat) lives under each case:
 * /cases/[caseId]/chat. This page replaces a bare 404 when users open /investigation.
 */
export default function InvestigationHubPage() {
    return (
        <div className="max-w-xl mx-auto text-center space-y-6 py-12">
            <div className="inline-flex p-4 rounded-2xl bg-primary/10 border border-primary/20">
                <MessageSquare className="w-10 h-10 text-primary" />
            </div>
            <div>
                <h1 className="text-2xl font-bold tracking-tight">AI Investigation</h1>
                <p className="text-muted-foreground mt-3 text-sm leading-relaxed">
                    Document Q&amp;A runs <strong>per case</strong>. Open a case from your dashboard,
                    then choose <strong>AI Investigation</strong> in the sidebar, or use the button
                    on the case page.
                </p>
            </div>
            <div className="flex flex-col sm:flex-row gap-3 justify-center pt-2">
                <Link
                    href="/officer/dashboard"
                    className="inline-flex items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
                >
                    <LayoutDashboard className="w-4 h-4" />
                    Officer dashboard
                    <ArrowRight className="w-4 h-4 opacity-80" />
                </Link>
                <Link
                    href="/admin/dashboard"
                    className="inline-flex items-center justify-center gap-2 rounded-lg border border-border px-4 py-2.5 text-sm font-medium text-foreground hover:bg-muted/60 transition-colors"
                >
                    Admin dashboard
                </Link>
            </div>
            <p className="text-xs text-muted-foreground font-mono pt-4">
                URL pattern: <span className="text-foreground">/cases/&lt;case-id&gt;/chat</span>
            </p>
        </div>
    );
}
