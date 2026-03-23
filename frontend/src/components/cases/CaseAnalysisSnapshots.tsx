'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { Camera, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { getApiV1Url } from '@/lib/api';

export type AnalysisSnapshotRow = {
    id: string;
    case_id: string;
    analysis_type: string;
    result_text: string;
    created_at: string;
};

export function CaseAnalysisSnapshots({
    caseId,
    token,
    initial,
}: {
    caseId: string;
    token: string;
    initial: AnalysisSnapshotRow[];
}) {
    const router = useRouter();
    const [saving, setSaving] = useState(false);
    const [err, setErr] = useState<string | null>(null);

    const capture = async () => {
        setSaving(true);
        setErr(null);
        try {
            const res = await fetch(`${getApiV1Url()}/cases/${caseId}/analysis`, {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}` },
            });
            if (!res.ok) {
                const j = await res.json().catch(() => ({}));
                setErr(typeof j.detail === 'string' ? j.detail : `Save failed (${res.status})`);
                return;
            }
            router.refresh();
        } catch {
            setErr('Network error');
        } finally {
            setSaving(false);
        }
    };

    return (
        <section className="bg-card/50 backdrop-blur-md border border-border rounded-xl p-6 shadow-sm">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-4">
                <div>
                    <h2 className="text-xl font-semibold flex items-center gap-2">
                        <Camera className="w-5 h-5 text-primary" />
                        Insights snapshots
                    </h2>
                    <p className="text-xs text-muted-foreground mt-1 max-w-xl">
                        Saves the current supervisor brief and blackboard subset from the RAG Insights service into this case (for
                        reports / audit). Run agents in Insights first for meaningful content.
                    </p>
                </div>
                <Button type="button" onClick={capture} disabled={saving} className="shrink-0 gap-2">
                    {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Camera className="w-4 h-4" />}
                    Capture snapshot
                </Button>
            </div>
            {err && (
                <div className="mb-4 rounded-lg border border-destructive/40 bg-destructive/10 text-destructive text-sm px-3 py-2">
                    {err}
                </div>
            )}
            <div className="space-y-4 max-h-[480px] overflow-y-auto">
                {initial.length === 0 ? (
                    <p className="text-sm text-muted-foreground border border-dashed border-border rounded-lg p-6 text-center">
                        No snapshots yet. Open Insights, run the pipeline, then capture here.
                    </p>
                ) : (
                    initial.map((row) => (
                        <details
                            key={row.id}
                            className="rounded-lg border border-border bg-background/40 open:pb-3"
                        >
                            <summary className="cursor-pointer px-4 py-3 text-sm font-medium list-none flex justify-between gap-2 [&::-webkit-details-marker]:hidden">
                                <span>
                                    {row.analysis_type.replace(/_/g, ' ')} ·{' '}
                                    {new Date(row.created_at).toLocaleString()}
                                </span>
                                <span className="text-xs text-muted-foreground font-normal">Expand</span>
                            </summary>
                            <pre className="mx-4 mt-0 text-xs whitespace-pre-wrap text-muted-foreground font-sans leading-relaxed border-t border-border pt-3 max-h-64 overflow-y-auto">
                                {row.result_text}
                            </pre>
                        </details>
                    ))
                )}
            </div>
        </section>
    );
}
