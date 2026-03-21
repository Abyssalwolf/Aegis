'use client';

import { useState, useRef, useEffect } from 'react';
import {
    Send, FileText, Loader2, UploadCloud, CheckCircle2,
    AlertCircle, Clock, ChevronDown, ChevronUp, BookOpen, X,
    RefreshCw, Search, SlidersHorizontal, Sparkles, Brain,
} from 'lucide-react';
import { Button } from '@/components/ui/button';

interface Document {
    id: string;
    filename: string | null;
    document_type: string;
    ingest_status: string;
    rag_document_id: string | null;
    created_at: string;
}

interface SourceReference {
    index: number;
    document_id: string;
    source_path: string;
    page_number: number | null;
    relevance_score: number;
    chunk_type: string;
}

interface Message {
    role: 'user' | 'assistant';
    content: string;
    reasoning?: string;
    sources?: SourceReference[];
    chunks_retrieved?: number;
    isError?: boolean;
}

interface ChatInterfaceProps {
    caseId: string;
    token: string;
    initialDocuments: Document[];
    caseName: string;
}

function StatusBadge({ status }: { status: string }) {
    const map: Record<string, { label: string; className: string; icon: React.ReactNode }> = {
        completed: {
            label: 'Indexed',
            className: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
            icon: <CheckCircle2 className="w-3 h-3" />,
        },
        processing: {
            label: 'Processing',
            className: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
            icon: <Loader2 className="w-3 h-3 animate-spin" />,
        },
        pending: {
            label: 'Pending',
            className: 'bg-muted/60 text-muted-foreground border-border',
            icon: <Clock className="w-3 h-3" />,
        },
        failed: {
            label: 'Failed',
            className: 'bg-red-500/20 text-red-400 border-red-500/30',
            icon: <AlertCircle className="w-3 h-3" />,
        },
        rag_unavailable: {
            label: 'RAG Offline',
            className: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
            icon: <AlertCircle className="w-3 h-3" />,
        },
    };
    const config = map[status] ?? map.pending;
    return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium border ${config.className}`}>
            {config.icon}
            {config.label}
        </span>
    );
}

function SourceCard({ sources }: { sources: SourceReference[] }) {
    const [open, setOpen] = useState(false);
    if (!sources.length) return null;
    return (
        <div className="mt-3 border border-border rounded-lg overflow-hidden">
            <button
                onClick={() => setOpen(v => !v)}
                className="w-full flex items-center justify-between px-3 py-2 text-xs text-muted-foreground hover:text-foreground hover:bg-muted/40 transition-colors"
            >
                <span className="flex items-center gap-1.5">
                    <BookOpen className="w-3.5 h-3.5" />
                    {sources.length} source{sources.length !== 1 ? 's' : ''} cited
                </span>
                {open ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
            {open && (
                <div className="divide-y divide-border">
                    {sources.map((s) => (
                        <div key={s.index} className="px-3 py-2 text-xs bg-muted/20">
                            <div className="flex items-center gap-2 mb-0.5">
                                <span className="font-semibold text-primary">[{s.index}]</span>
                                <span className="text-muted-foreground truncate max-w-[200px]">
                                    {s.source_path.split('/').pop() || s.source_path}
                                </span>
                                {s.page_number != null && (
                                    <span className="text-muted-foreground">p.{s.page_number}</span>
                                )}
                            </div>
                            <div className="text-muted-foreground">
                                Score: {(s.relevance_score * 100).toFixed(0)}% · {s.chunk_type}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

function ReasoningCard({ reasoning }: { reasoning: string }) {
    const [open, setOpen] = useState(false);
    if (!reasoning) return null;
    return (
        <div className="mt-3 border border-border rounded-lg overflow-hidden">
            <button
                onClick={() => setOpen(v => !v)}
                className="w-full flex items-center justify-between px-3 py-2 text-xs text-muted-foreground hover:text-foreground hover:bg-muted/40 transition-colors"
            >
                <span className="flex items-center gap-1.5">
                    <Brain className="w-3.5 h-3.5" />
                    View reasoning
                </span>
                {open ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
            {open && (
                <div className="px-3 py-2.5 text-xs bg-muted/20 text-muted-foreground whitespace-pre-wrap max-h-64 overflow-y-auto leading-relaxed">
                    {reasoning}
                </div>
            )}
        </div>
    );
}

// Each stage becomes active once the elapsed ms exceeds its `fromMs` threshold.
const PIPELINE_STAGES = [
    {
        id: 'rewrite',
        label: 'Rewriting query variants',
        detail: 'expanding coverage with alternate phrasings',
        icon: RefreshCw,
        fromMs: 0,
    },
    {
        id: 'search',
        label: 'Searching evidence files',
        detail: 'dense vectors + sparse BM25 retrieval',
        icon: Search,
        fromMs: 10_000,
    },
    {
        id: 'rank',
        label: 'Ranking relevant chunks',
        detail: 'cross-encoder relevance scoring',
        icon: SlidersHorizontal,
        fromMs: 14_000,
    },
    {
        id: 'generate',
        label: 'Generating answer',
        detail: 'synthesising response from sources',
        icon: Sparkles,
        fromMs: 17_000,
    },
] as const;

function ThinkingIndicator({ loading }: { loading: boolean }) {
    const [stageIndex, setStageIndex] = useState(0);
    const [elapsed, setElapsed] = useState(0);

    useEffect(() => {
        if (!loading) {
            setStageIndex(0);
            setElapsed(0);
            return;
        }
        const startTime = Date.now();
        const tick = setInterval(() => {
            const ms = Date.now() - startTime;
            setElapsed(Math.floor(ms / 1000));
            let next = 0;
            for (let i = 0; i < PIPELINE_STAGES.length; i++) {
                if (ms >= PIPELINE_STAGES[i].fromMs) next = i;
            }
            setStageIndex(next);
        }, 400);
        return () => clearInterval(tick);
    }, [loading]);

    if (!loading) return null;

    return (
        <div className="flex justify-start">
            <div className="bg-muted/60 rounded-2xl rounded-bl-sm px-4 py-4 min-w-[260px] max-w-[320px]">
                {/* Stage list */}
                <div className="space-y-3 mb-4">
                    {PIPELINE_STAGES.map((stage, i) => {
                        const Icon = stage.icon;
                        const isActive = i === stageIndex;
                        const isDone = i < stageIndex;
                        return (
                            <div
                                key={stage.id}
                                className={`flex items-start gap-2.5 text-xs transition-all duration-500 ${
                                    isActive
                                        ? 'opacity-100'
                                        : isDone
                                        ? 'opacity-50'
                                        : 'opacity-20'
                                }`}
                            >
                                {/* Icon bubble */}
                                <div className={`mt-0.5 w-5 h-5 flex items-center justify-center rounded-full shrink-0 transition-colors duration-300 ${
                                    isDone
                                        ? 'bg-emerald-500/20'
                                        : isActive
                                        ? 'bg-primary/20'
                                        : 'bg-muted'
                                }`}>
                                    {isDone ? (
                                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                                    ) : (
                                        <Icon className={`w-3 h-3 ${isActive ? 'text-primary animate-pulse' : 'text-muted-foreground'}`} />
                                    )}
                                </div>

                                {/* Label + detail */}
                                <div className="flex-1 min-w-0">
                                    <div className={`font-medium leading-tight ${isActive ? 'text-foreground' : 'text-muted-foreground'}`}>
                                        {stage.label}
                                        {isActive && (
                                            <span className="inline-flex gap-0.5 ml-1.5 align-middle">
                                                <span className="w-1 h-1 rounded-full bg-primary animate-bounce [animation-delay:0ms]" />
                                                <span className="w-1 h-1 rounded-full bg-primary animate-bounce [animation-delay:150ms]" />
                                                <span className="w-1 h-1 rounded-full bg-primary animate-bounce [animation-delay:300ms]" />
                                            </span>
                                        )}
                                    </div>
                                    {isActive && (
                                        <div className="text-[10px] text-muted-foreground mt-0.5 leading-tight">
                                            {stage.detail}
                                        </div>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>

                {/* Progress bar */}
                <div className="h-1 rounded-full bg-muted overflow-hidden">
                    <div
                        className="h-full bg-primary rounded-full transition-all duration-700 ease-out"
                        style={{
                            width: `${Math.min(
                                10 + (stageIndex / (PIPELINE_STAGES.length - 1)) * 80 +
                                Math.min(elapsed * 0.5, 9),
                                95
                            )}%`,
                        }}
                    />
                </div>
                <div className="flex justify-between items-center mt-1.5 text-[10px] text-muted-foreground">
                    <span>Processing</span>
                    <span>{elapsed}s elapsed</span>
                </div>
            </div>
        </div>
    );
}

function renderContent(text: string): React.ReactNode {
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, i) =>
        part.startsWith('**') && part.endsWith('**')
            ? <strong key={i}>{part.slice(2, -2)}</strong>
            : part
    );
}

export default function ChatInterface({ caseId, token, initialDocuments, caseName }: ChatInterfaceProps) {
    const [documents, setDocuments] = useState<Document[]>(initialDocuments);
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [uploading, setUploading] = useState(false);
    const bottomRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const sendMessage = async () => {
        const query = input.trim();
        if (!query || loading) return;

        setInput('');
        const userMsg: Message = { role: 'user', content: query };
        const updatedMessages = [...messages, userMsg];
        setMessages(updatedMessages);
        setLoading(true);

        // Build conversation history from all previous non-error messages
        const history = updatedMessages
            .filter(m => !m.isError)
            .map(m => ({ role: m.role, content: m.content }));

        try {
            const res = await fetch(`http://localhost:8000/api/v1/cases/${caseId}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify({
                    query,
                    top_k: 5,
                    messages: history.slice(0, -1),
                }),
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: err.detail || `Error ${res.status}: Failed to get a response.`,
                    isError: true,
                }]);
                return;
            }

            const data = await res.json();
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: data.answer,
                reasoning: data.reasoning || '',
                sources: data.sources,
                chunks_retrieved: data.chunks_retrieved,
            }]);
        } catch {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'Could not reach the server. Please check your connection.',
                isError: true,
            }]);
        } finally {
            setLoading(false);
        }
    };

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setUploading(true);
        const form = new FormData();
        form.append('file', file);
        form.append('document_type', file.type || 'application/octet-stream');

        try {
            const res = await fetch(`http://localhost:8000/api/v1/cases/${caseId}/documents`, {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}` },
                body: form,
            });

            if (res.ok) {
                const newDoc: Document = await res.json();
                setDocuments(prev => [newDoc, ...prev]);
            }
        } catch {
            // silently fail; user can retry
        } finally {
            setUploading(false);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

    const clearChat = () => setMessages([]);

    return (
        <div className="flex h-[calc(100vh-12rem)] gap-4">
            {/* Left Panel — Documents */}
            <div className="w-72 shrink-0 flex flex-col gap-3">
                <div className="bg-card/50 backdrop-blur-md border border-border rounded-xl p-4 flex flex-col gap-3 h-full">
                    <div className="flex items-center justify-between">
                        <h2 className="text-sm font-semibold flex items-center gap-2 text-foreground">
                            <FileText className="w-4 h-4 text-primary" />
                            Evidence Files
                        </h2>
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp,.webp"
                            className="hidden"
                            onChange={handleUpload}
                        />
                        <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 w-7 p-0"
                            title="Upload document"
                            onClick={() => fileInputRef.current?.click()}
                            disabled={uploading}
                        >
                            {uploading ? (
                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                            ) : (
                                <UploadCloud className="w-3.5 h-3.5" />
                            )}
                        </Button>
                    </div>

                    <div className="flex-1 overflow-y-auto space-y-2 pr-0.5">
                        {documents.length === 0 && (
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                className="w-full flex flex-col items-center justify-center p-6 border-2 border-dashed border-border rounded-lg text-center hover:border-primary/50 hover:bg-primary/5 transition-colors group"
                            >
                                <UploadCloud className="w-7 h-7 text-muted-foreground group-hover:text-primary mb-2 transition-colors" />
                                <span className="text-xs text-muted-foreground">Upload a PDF or image to begin</span>
                            </button>
                        )}
                        {documents.map((doc) => (
                            <div
                                key={doc.id}
                                className="p-3 rounded-lg border border-border bg-background/60 space-y-1.5"
                            >
                                <div className="flex items-start gap-2">
                                    <FileText className="w-4 h-4 text-primary mt-0.5 shrink-0" />
                                    <span className="text-xs font-medium leading-snug break-all line-clamp-2">
                                        {doc.filename || doc.document_type}
                                    </span>
                                </div>
                                <div className="flex items-center justify-between">
                                    <StatusBadge status={doc.ingest_status} />
                                    <span className="text-[10px] text-muted-foreground">
                                        {new Date(doc.created_at).toLocaleDateString()}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>

                    <p className="text-[10px] text-muted-foreground text-center border-t border-border pt-2">
                        Only "Indexed" files are queryable
                    </p>
                </div>
            </div>

            {/* Right Panel — Chat */}
            <div className="flex-1 flex flex-col bg-card/50 backdrop-blur-md border border-border rounded-xl overflow-hidden">
                {/* Chat header */}
                <div className="flex items-center justify-between px-5 py-3 border-b border-border shrink-0">
                    <div>
                        <p className="text-xs text-muted-foreground">Querying documents for</p>
                        <h2 className="text-sm font-semibold text-foreground truncate max-w-[360px]">{caseName}</h2>
                    </div>
                    {messages.length > 0 && (
                        <Button variant="ghost" size="sm" className="h-7 gap-1.5 text-xs" onClick={clearChat}>
                            <X className="w-3.5 h-3.5" />
                            Clear
                        </Button>
                    )}
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-5 space-y-5">
                    {messages.length === 0 && (
                        <div className="h-full flex flex-col items-center justify-center text-center text-muted-foreground gap-3">
                            <div className="p-4 bg-primary/10 rounded-full">
                                <Send className="w-7 h-7 text-primary" />
                            </div>
                            <div>
                                <p className="font-medium text-foreground">Ask anything about the case</p>
                                <p className="text-sm mt-1">The AI will search across all indexed evidence files.</p>
                            </div>
                            <div className="mt-2 flex flex-wrap gap-2 justify-center max-w-md">
                                {[
                                    'Summarize the key evidence',
                                    'Who are the main suspects?',
                                    'What is the timeline of events?',
                                ].map((suggestion) => (
                                    <button
                                        key={suggestion}
                                        onClick={() => setInput(suggestion)}
                                        className="px-3 py-1.5 text-xs rounded-full border border-border hover:border-primary/50 hover:bg-primary/5 transition-colors"
                                    >
                                        {suggestion}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {messages.map((msg, i) => (
                        <div
                            key={i}
                            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div
                                className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                                    msg.role === 'user'
                                        ? 'bg-primary text-primary-foreground rounded-br-sm'
                                        : msg.isError
                                        ? 'bg-red-500/10 border border-red-500/30 text-red-400 rounded-bl-sm'
                                        : 'bg-muted/60 text-foreground rounded-bl-sm'
                                }`}
                            >
                                <p className="whitespace-pre-wrap">{renderContent(msg.content)}</p>
                                {msg.reasoning && <ReasoningCard reasoning={msg.reasoning} />}
                                {msg.sources && <SourceCard sources={msg.sources} />}
                            </div>
                        </div>
                    ))}

                    <ThinkingIndicator loading={loading} />
                    <div ref={bottomRef} />
                </div>

                {/* Input */}
                <div className="p-4 border-t border-border shrink-0">
                    <form
                        onSubmit={(e) => { e.preventDefault(); sendMessage(); }}
                        className="flex gap-2"
                    >
                        <input
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder={documents.some(d => d.ingest_status === 'completed')
                                ? 'Ask a question about the case documents…'
                                : 'Upload and index documents above to start querying…'}
                            className="flex-1 bg-background border border-border rounded-xl px-4 py-2.5 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50 transition-all"
                            disabled={loading}
                        />
                        <Button
                            type="submit"
                            size="sm"
                            className="h-10 w-10 p-0 rounded-xl shrink-0"
                            disabled={!input.trim() || loading}
                        >
                            {loading ? (
                                <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                                <Send className="w-4 h-4" />
                            )}
                        </Button>
                    </form>
                    <p className="text-[10px] text-muted-foreground text-center mt-2">
                        Answers are generated from case documents only — always verify against source evidence.
                    </p>
                </div>
            </div>
        </div>
    );
}
