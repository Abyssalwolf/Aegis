import { getAccessToken } from '@/lib/auth';
import { getApiV1Url } from '@/lib/api';
import { MessageSquare, ArrowLeft } from 'lucide-react';
import Link from 'next/link';
import ChatInterface from '@/components/cases/ChatInterface';

export default async function ChatPage({ params }: { params: Promise<{ id: string }> }) {
    const { id } = await params;
    const token = await getAccessToken();

    // Fetch case details (for the title shown in the chat header)
    const caseRes = await fetch(`${getApiV1Url()}/cases/${id}`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store',
    });

    if (!caseRes.ok) {
        return (
            <div className="p-12 text-center text-destructive">
                Failed to load case. You may not have access.
            </div>
        );
    }

    const caseData = await caseRes.json();

    // Pre-fetch documents so the sidebar shows immediately without a client-side waterfall
    const docsRes = await fetch(`${getApiV1Url()}/cases/${id}/documents`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store',
    });
    const documents = docsRes.ok ? await docsRes.json() : [];

    return (
        <div className="flex flex-col h-full">
            <div className="mb-4 shrink-0">
                <Link
                    href={`/cases/${id}`}
                    className="inline-flex items-center text-sm font-medium text-muted-foreground hover:text-foreground transition-colors gap-2"
                >
                    <ArrowLeft className="w-4 h-4" />
                    Back to Case
                </Link>
                <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3 text-foreground mt-2">
                    <MessageSquare className="w-6 h-6 text-primary" />
                    AI Investigation — RAG Chat
                </h1>
                <p className="text-muted-foreground text-sm mt-0.5">
                    Query indexed evidence documents using AI-powered retrieval.
                </p>
            </div>

            <ChatInterface
                caseId={id}
                token={token || ''}
                initialDocuments={documents}
                caseName={caseData.title}
            />
        </div>
    );
}
