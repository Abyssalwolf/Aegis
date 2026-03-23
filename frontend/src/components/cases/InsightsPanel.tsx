'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import {
    ArrowLeft,
    BrainCircuit,
    ChevronDown,
    ChevronUp,
    Loader2,
    RefreshCw,
    Sparkles,
    AlertTriangle,
    MessageSquare,
    FileText,
    Lightbulb,
    Radio,
    UploadCloud,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { getApiV1Url } from '@/lib/api';

type BlackboardPayload = {
    messages?: unknown[];
    anomalies?: unknown[];
    findings?: unknown[];
    insights?: unknown[];
    case_id?: number;
    status?: string | null;
};

function CollapsibleSection({
    title,
    icon,
    count,
    defaultOpen = false,
    children,
}: {
    title: string;
    icon: React.ReactNode;
    count: number;
    defaultOpen?: boolean;
    children: React.ReactNode;
}) {
    const [open, setOpen] = useState(defaultOpen);
    return (
        <div className="border border-border rounded-lg overflow-hidden bg-card/40">
            <button
                type="button"
                onClick={() => setOpen((v) => !v)}
                className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium hover:bg-muted/40 transition-colors"
            >
                <span className="flex items-center gap-2">
                    {icon}
                    {title}
                    <span className="text-xs font-normal text-muted-foreground">({count})</span>
                </span>
                {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
            {open && <div className="px-4 pb-4 border-t border-border/60 max-h-[420px] overflow-y-auto">{children}</div>}
        </div>
    );
}

function JsonCard({ item }: { item: Record<string, unknown> }) {
    const agent = typeof item.agent === 'string' ? item.agent : '—';
    const content =
        typeof item.content === 'string'
            ? item.content
            : typeof item.summary === 'string'
              ? item.summary
              : JSON.stringify(item, null, 2);
    return (
        <div className="text-xs rounded-md border border-border/80 bg-background/50 p-3 mb-2 last:mb-0">
            <div className="text-[10px] uppercase tracking-wide text-muted-foreground mb-1">{agent}</div>
            <pre className="whitespace-pre-wrap text-foreground/90 font-sans leading-relaxed">{content}</pre>
        </div>
    );
}

interface InsightsPanelProps {
    caseId: string;
    token: string;
    caseTitle: string;
}

export default function InsightsPanel({ caseId, token, caseTitle }: InsightsPanelProps) {
    const base = `${getApiV1Url()}/cases/${caseId}/insights`;

    const [blackboard, setBlackboard] = useState<BlackboardPayload | null>(null);
    const [brief, setBrief] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [reporting, setReporting] = useState(false);
    const [taskLog, setTaskLog] = useState<string | null>(null);
    /** Full markdown report from the last successful supervisor Celery task */
    const [supervisorReport, setSupervisorReport] = useState<string | null>(null);
    const [watchingTask, setWatchingTask] = useState<string | null>(null);
    const [liveOn, setLiveOn] = useState(false);
    const [liveFeed, setLiveFeed] = useState<Record<string, unknown>[]>([]);
    const [streamError, setStreamError] = useState<string | null>(null);
    const streamStopped = useRef(false);

    const authHeaders = { Authorization: `Bearer ${token}` };

    const loadBlackboard = useCallback(async () => {
        setError(null);
        try {
            const res = await fetch(`${base}/blackboard`, { headers: authHeaders, cache: 'no-store' });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                setError(typeof err.detail === 'string' ? err.detail : `Blackboard failed (${res.status})`);
                setBlackboard(null);
                return;
            }
            setBlackboard(await res.json());
        } catch {
            setError('Could not reach the API. Is the backend running?');
            setBlackboard(null);
        } finally {
            setLoading(false);
        }
    }, [base, token]);

    const loadBrief = useCallback(async () => {
        try {
            const res = await fetch(`${base}/blackboard/brief`, { headers: authHeaders, cache: 'no-store' });
            if (res.ok) {
                const data = await res.json();
                setBrief(typeof data.brief === 'string' ? data.brief : null);
            }
        } catch {
            /* optional */
        }
    }, [base, token]);

    const reportStorageKey = `insights-supervisor-report:${caseId}`;

    useEffect(() => {
        loadBlackboard();
        loadBrief();
        try {
            const saved = sessionStorage.getItem(reportStorageKey);
            setSupervisorReport(saved && saved.trim() ? saved : null);
        } catch {
            /* private mode / quota */
        }
    }, [loadBlackboard, loadBrief, reportStorageKey]);

    useEffect(() => {
        if (!supervisorReport) return;
        try {
            sessionStorage.setItem(reportStorageKey, supervisorReport);
        } catch {
            /* ignore */
        }
    }, [supervisorReport, reportStorageKey]);

    useEffect(() => {
        if (!watchingTask) return;
        let cancelled = false;
        const tick = async () => {
            try {
                const res = await fetch(`${base}/tasks/${watchingTask}`, {
                    headers: authHeaders,
                    cache: 'no-store',
                });
                if (!res.ok || cancelled) return;
                const data = await res.json();
                const st = data.status as string;
                setTaskLog(`Task ${watchingTask}: ${st}`);
                if (st === 'SUCCESS') {
                    const r = data.result as Record<string, unknown> | undefined;
                    if (r && typeof r.supervisor_report === 'string' && r.supervisor_report.trim()) {
                        setSupervisorReport(r.supervisor_report);
                    }
                }
                if (st === 'FAILURE' || st === 'REVOKED') {
                    const r = data.result as Record<string, unknown> | undefined;
                    const err =
                        r && typeof r.error === 'string'
                            ? r.error
                            : st === 'FAILURE'
                              ? 'Supervisor task failed.'
                              : 'Supervisor task revoked.';
                    setTaskLog(`${err} (task ${watchingTask})`);
                }
                if (st === 'SUCCESS' || st === 'FAILURE' || st === 'REVOKED') {
                    setWatchingTask(null);
                    await loadBlackboard();
                    await loadBrief();
                }
            } catch {
                /* ignore */
            }
        };
        tick();
        const id = setInterval(tick, 2500);
        return () => {
            cancelled = true;
            clearInterval(id);
        };
    }, [watchingTask, base, token, loadBlackboard, loadBrief]);

    /** SSE via fetch + Authorization (EventSource cannot send Bearer). */
    useEffect(() => {
        if (!liveOn) {
            streamStopped.current = true;
            setStreamError(null);
            return;
        }
        streamStopped.current = false;
        setLiveFeed([]);
        const ac = new AbortController();
        const headers = { Authorization: `Bearer ${token}` };

        (async () => {
            try {
                const res = await fetch(`${base}/stream`, { headers, signal: ac.signal, cache: 'no-store' });
                if (!res.ok || !res.body) {
                    setStreamError(`Stream HTTP ${res.status}`);
                    setLiveOn(false);
                    return;
                }
                const reader = res.body.getReader();
                const dec = new TextDecoder();
                let buf = '';
                while (!streamStopped.current) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buf += dec.decode(value, { stream: true });
                    const parts = buf.split('\n\n');
                    buf = parts.pop() ?? '';
                    for (const block of parts) {
                        for (const line of block.split('\n')) {
                            const t = line.trim();
                            if (!t || t.startsWith(':')) continue;
                            if (t.startsWith('data: ')) {
                                try {
                                    const payload = JSON.parse(t.slice(6)) as Record<string, unknown>;
                                    if (payload.error) continue;
                                    setLiveFeed((prev) => [...prev.slice(-49), payload]);
                                } catch {
                                    /* non-JSON line */
                                }
                            }
                        }
                    }
                }
                if (!ac.signal.aborted && !streamStopped.current) {
                    setLiveOn(false);
                }
            } catch {
                if (ac.signal.aborted) return;
                setStreamError('Live stream disconnected.');
                setLiveOn(false);
            }
        })();

        return () => {
            streamStopped.current = true;
            ac.abort();
        };
    }, [liveOn, base, token]);

    const onReport = async () => {
        setReporting(true);
        setTaskLog(null);
        setError(null);
        try {
            const res = await fetch(`${base}/report`, { method: 'POST', headers: authHeaders });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                setError(typeof data.detail === 'string' ? data.detail : `Report failed (${res.status})`);
                return;
            }
            if (data.task_id) {
                setWatchingTask(data.task_id);
                setTaskLog(`Supervisor task ${data.task_id}`);
            }
        } catch {
            setError('Report request failed.');
        } finally {
            setReporting(false);
        }
    };

    const messages = blackboard?.messages ?? [];
    const anomalies = blackboard?.anomalies ?? [];
    const findings = blackboard?.findings ?? [];
    const insights = blackboard?.insights ?? [];

    return (
        <div className="flex flex-col gap-6 min-h-[70vh]">
            <div className="shrink-0 flex flex-wrap items-start justify-between gap-4">
                <div>
                    <Link
                        href={`/cases/${caseId}`}
                        className="inline-flex items-center text-sm font-medium text-muted-foreground hover:text-foreground gap-2"
                    >
                        <ArrowLeft className="w-4 h-4" />
                        Back to Case
                    </Link>
                    <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3 text-foreground mt-2">
                        <BrainCircuit className="w-7 h-7 text-primary" />
                        Insights — {caseTitle}
                    </h1>
                    <p className="text-muted-foreground text-sm mt-1 max-w-2xl">
                        Blackboard and supervisor output (Redis + Celery on the RAG service).{' '}
                        <strong>Upload evidence</strong> from the case page or <strong>AI Investigation</strong> — each upload
                        ingests for chat search and <strong>queues this agent pipeline in parallel</strong>. Use Refresh,{' '}
                        <strong>Live stream</strong>, or wait for Celery. Requires RAG + worker running.
                    </p>
                </div>
                <div className="flex flex-wrap gap-2">
                    <Button type="button" variant="outline" size="sm" onClick={() => { loadBlackboard(); loadBrief(); }} disabled={loading}>
                        <RefreshCw className={`w-4 h-4 mr-1.5 ${loading ? 'animate-spin' : ''}`} />
                        Refresh
                    </Button>
                    <Button
                        type="button"
                        variant={liveOn ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setLiveOn((v) => !v)}
                        title="Uses fetch + Bearer token (SSE); keep tab open while agents run"
                    >
                        <Radio className={`w-4 h-4 mr-1.5 ${liveOn ? 'animate-pulse' : ''}`} />
                        {liveOn ? 'Stop live' : 'Live stream'}
                    </Button>
                    <Button type="button" size="sm" onClick={onReport} disabled={reporting}>
                        {reporting ? <Loader2 className="w-4 h-4 animate-spin mr-1.5" /> : <Sparkles className="w-4 h-4 mr-1.5" />}
                        Supervisor report
                    </Button>
                </div>
            </div>

            {error && (
                <div className="rounded-lg border border-destructive/40 bg-destructive/10 text-destructive text-sm px-4 py-3">
                    {error}
                </div>
            )}
            {taskLog && (
                <div className="rounded-lg border border-border bg-muted/30 text-xs px-3 py-2 text-muted-foreground flex items-center gap-2">
                    {watchingTask ? <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" /> : null}
                    {taskLog}
                </div>
            )}
            {streamError && (
                <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 text-amber-200 text-xs px-3 py-2">{streamError}</div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-1 space-y-4">
                    <div className="rounded-xl border border-border bg-card/50 p-4">
                        <h2 className="text-sm font-semibold flex items-center gap-2 mb-3">
                            <UploadCloud className="w-4 h-4 text-primary" />
                            Document upload
                        </h2>
                        <p className="text-xs text-muted-foreground mb-3 leading-relaxed">
                            Use the same upload flow as everywhere else: open{' '}
                            <Link href={`/cases/${caseId}`} className="text-primary underline-offset-2 hover:underline">
                                Case
                            </Link>{' '}
                            or{' '}
                            <Link href={`/cases/${caseId}/chat`} className="text-primary underline-offset-2 hover:underline">
                                AI Investigation
                            </Link>
                            , pick the <strong>evidence type</strong>, and submit. The backend runs{' '}
                            <strong>RAG ingest + agent queue together</strong> (no separate Insights upload).
                        </p>
                    </div>

                    <div className="rounded-xl border border-border bg-card/50 p-4">
                        <h2 className="text-sm font-semibold mb-2">Case status (blackboard)</h2>
                        {loading ? (
                            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
                        ) : (
                            <p className="text-sm font-mono text-foreground/90">{blackboard?.status ?? '—'}</p>
                        )}
                    </div>
                </div>

                <div className="lg:col-span-2 space-y-3">
                    {supervisorReport && (
                        <CollapsibleSection
                            title="Consolidated supervisor report"
                            icon={<Sparkles className="w-4 h-4 text-primary" />}
                            count={1}
                            defaultOpen
                        >
                            <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                                <p className="text-[11px] text-muted-foreground">
                                    From your last supervisor run (kept for this browser tab until you clear it).
                                </p>
                                <Button
                                    type="button"
                                    variant="ghost"
                                    size="sm"
                                    className="h-7 text-xs"
                                    onClick={() => {
                                        setSupervisorReport(null);
                                        try {
                                            sessionStorage.removeItem(reportStorageKey);
                                        } catch {
                                            /* ignore */
                                        }
                                    }}
                                >
                                    Clear report
                                </Button>
                            </div>
                            <div className="rounded-md border border-border/80 bg-background/80 p-3 max-h-[min(70vh,720px)] overflow-y-auto">
                                <pre className="text-xs whitespace-pre-wrap font-sans text-foreground/95 leading-relaxed">
                                    {supervisorReport}
                                </pre>
                            </div>
                        </CollapsibleSection>
                    )}

                    {brief && (
                        <CollapsibleSection title="Supervisor brief (markdown)" icon={<FileText className="w-4 h-4" />} count={1} defaultOpen={false}>
                            <pre className="text-xs whitespace-pre-wrap font-sans text-muted-foreground leading-relaxed">{brief}</pre>
                        </CollapsibleSection>
                    )}

                    {(liveOn || liveFeed.length > 0) && (
                        <CollapsibleSection
                            title="Live stream (recent events)"
                            icon={<Radio className="w-4 h-4 text-emerald-400" />}
                            count={liveFeed.length}
                            defaultOpen={liveOn}
                        >
                            {!liveOn && liveFeed.length === 0 ? (
                                <p className="text-xs text-muted-foreground py-2">Turn on Live stream to capture blackboard events.</p>
                            ) : liveFeed.length === 0 ? (
                                <p className="text-xs text-muted-foreground py-2 flex items-center gap-2">
                                    <Loader2 className="w-3.5 h-3.5 animate-spin" /> Listening…
                                </p>
                            ) : (
                                liveFeed.map((item, i) => <JsonCard key={`live-${i}`} item={item} />)
                            )}
                        </CollapsibleSection>
                    )}

                    <CollapsibleSection
                        title="Findings"
                        icon={<FileText className="w-4 h-4 text-emerald-400" />}
                        count={findings.length}
                        defaultOpen
                    >
                        {findings.length === 0 ? (
                            <p className="text-xs text-muted-foreground py-2">No findings yet.</p>
                        ) : (
                            findings.map((f, i) => (
                                <JsonCard key={`f-${i}`} item={f as Record<string, unknown>} />
                            ))
                        )}
                    </CollapsibleSection>

                    <CollapsibleSection title="Insights" icon={<Lightbulb className="w-4 h-4 text-amber-400" />} count={insights.length}>
                        {insights.length === 0 ? (
                            <p className="text-xs text-muted-foreground py-2">No supervisor insights yet.</p>
                        ) : (
                            insights.map((f, i) => (
                                <JsonCard key={`i-${i}`} item={f as Record<string, unknown>} />
                            ))
                        )}
                    </CollapsibleSection>

                    <CollapsibleSection
                        title="Anomalies"
                        icon={<AlertTriangle className="w-4 h-4 text-orange-400" />}
                        count={anomalies.length}
                    >
                        {anomalies.length === 0 ? (
                            <p className="text-xs text-muted-foreground py-2">None.</p>
                        ) : (
                            anomalies.map((f, i) => (
                                <JsonCard key={`a-${i}`} item={f as Record<string, unknown>} />
                            ))
                        )}
                    </CollapsibleSection>

                    <CollapsibleSection
                        title="Messages"
                        icon={<MessageSquare className="w-4 h-4 text-sky-400" />}
                        count={messages.length}
                    >
                        {messages.length === 0 ? (
                            <p className="text-xs text-muted-foreground py-2">None.</p>
                        ) : (
                            messages.map((f, i) => (
                                <JsonCard key={`m-${i}`} item={f as Record<string, unknown>} />
                            ))
                        )}
                    </CollapsibleSection>
                </div>
            </div>
        </div>
    );
}
