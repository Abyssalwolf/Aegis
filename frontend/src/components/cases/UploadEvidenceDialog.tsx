'use client';

import { useState, useRef, useEffect } from 'react';
import { UploadCloud, Loader2, FileWarning } from 'lucide-react';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { getApiV1Url } from '@/lib/api';

/** Evidence types (required). Must map to an agent route — same list drives parallel Insights queue. */
export const EVIDENCE_CATEGORY_OPTIONS: { value: string; label: string }[] = [
    { value: '', label: 'Select evidence type *' },
    { value: 'fir', label: 'FIR' },
    { value: 'case_diary', label: 'Case diary' },
    { value: 'statement', label: 'Statement' },
    { value: 'scene_of_crime', label: 'Scene of crime' },
    { value: 'forensic', label: 'Forensic' },
    { value: 'seizure', label: 'Seizure / property' },
    { value: 'arrest_remand', label: 'Arrest / remand' },
];

export interface EvidenceDocument {
    id: string;
    filename: string | null;
    display_name?: string | null;
    evidence_category?: string | null;
    description?: string | null;
    document_type: string;
    ingest_status: string;
    rag_document_id: string | null;
    created_at: string;
    /** Celery task id when parallel Insights/agents queue succeeded */
    insights_task_id?: string | null;
    insights_queue_status?: string | null;
}

interface UploadEvidenceDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    caseId: string;
    token: string;
    onUploaded: (doc: EvidenceDocument) => void;
}

export function UploadEvidenceDialog({
    open,
    onOpenChange,
    caseId,
    token,
    onUploaded,
}: UploadEvidenceDialogProps) {
    const [displayName, setDisplayName] = useState('');
    const [evidenceCategory, setEvidenceCategory] = useState('');
    const [description, setDescription] = useState('');
    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (!open) {
            setDisplayName('');
            setEvidenceCategory('');
            setDescription('');
            setFile(null);
            setError(null);
            setUploading(false);
            if (inputRef.current) inputRef.current.value = '';
        }
    }, [open]);

    const submit = async () => {
        if (!file) {
            setError('Choose a PDF or image file.');
            return;
        }
        const dn = displayName.trim();
        if (!dn) {
            setError('Display name is required.');
            return;
        }
        if (!evidenceCategory) {
            setError('Evidence type is required.');
            return;
        }
        setError(null);
        setUploading(true);
        const form = new FormData();
        form.append('file', file);
        form.append('display_name', dn);
        form.append('evidence_category', evidenceCategory);
        const desc = description.trim();
        if (desc) form.append('description', desc);

        try {
            const res = await fetch(`${getApiV1Url()}/cases/${caseId}/documents`, {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}` },
                body: form,
            });
            const raw = await res.text();
            if (!res.ok) {
                let msg = `Upload failed (${res.status})`;
                try {
                    const err = JSON.parse(raw) as { detail?: string | unknown };
                    if (typeof err.detail === 'string') msg = err.detail;
                } catch {
                    if (raw.trim()) msg = raw.slice(0, 200);
                }
                setError(msg);
                return;
            }
            let newDoc: EvidenceDocument;
            try {
                newDoc = JSON.parse(raw) as EvidenceDocument;
            } catch {
                setError('Invalid response from server.');
                return;
            }
            onUploaded(newDoc);
            onOpenChange(false);
        } catch (e) {
            setError(
                e instanceof TypeError && e.message.includes('fetch')
                    ? 'Could not reach the server. Check your connection.'
                    : e instanceof Error
                      ? e.message
                      : 'Could not reach the server. Check your connection.',
            );
        } finally {
            setUploading(false);
        }
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-md">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <UploadCloud className="w-5 h-5 text-primary" />
                        Upload evidence
                    </DialogTitle>
                    <DialogDescription>
                        File, display name, and evidence type are required. Each upload also queues the multi-agent Insights
                        pipeline in parallel (same document type). Short note is optional.
                    </DialogDescription>
                </DialogHeader>

                <div className="space-y-3 text-sm">
                    <input
                        ref={inputRef}
                        type="file"
                        accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp,.webp"
                        className="hidden"
                        onChange={(e) => {
                            const f = e.target.files?.[0] ?? null;
                            setFile(f);
                            setError(null);
                        }}
                    />
                    <div>
                        <Button
                            type="button"
                            variant="outline"
                            className="w-full justify-start gap-2 h-auto py-2 px-3"
                            onClick={() => inputRef.current?.click()}
                            disabled={uploading}
                        >
                            <UploadCloud className="w-4 h-4 shrink-0" />
                            <span className="truncate text-left">
                                {file ? file.name : 'Choose file…'}
                            </span>
                        </Button>
                    </div>

                    <input
                        type="text"
                        placeholder="Display name *"
                        value={displayName}
                        onChange={(e) => setDisplayName(e.target.value)}
                        className="w-full rounded-md border border-border bg-background px-3 py-2 text-foreground placeholder:text-muted-foreground"
                        disabled={uploading}
                        required
                        aria-required
                    />
                    <select
                        value={evidenceCategory}
                        onChange={(e) => setEvidenceCategory(e.target.value)}
                        className="w-full rounded-md border border-border bg-background px-3 py-2 text-foreground"
                        disabled={uploading}
                        required
                        aria-required
                    >
                        {EVIDENCE_CATEGORY_OPTIONS.map((o) => (
                            <option key={o.value || 'placeholder'} value={o.value} disabled={o.value === ''}>
                                {o.label}
                            </option>
                        ))}
                    </select>
                    <input
                        type="text"
                        placeholder="Short note (optional)"
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        className="w-full rounded-md border border-border bg-background px-3 py-2 text-foreground placeholder:text-muted-foreground"
                        disabled={uploading}
                    />

                    {error && (
                        <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                            <FileWarning className="w-4 h-4 shrink-0 mt-0.5" />
                            {error}
                        </div>
                    )}
                </div>

                <DialogFooter className="gap-2 sm:gap-0">
                    <Button
                        type="button"
                        variant="outline"
                        onClick={() => onOpenChange(false)}
                        disabled={uploading}
                    >
                        Cancel
                    </Button>
                    <Button
                        type="button"
                        onClick={submit}
                        disabled={
                            uploading
                            || !file
                            || !displayName.trim()
                            || !evidenceCategory
                        }
                        className="inline-flex items-center gap-2"
                    >
                        {uploading && <Loader2 className="w-4 h-4 animate-spin" />}
                        {uploading ? 'Uploading…' : 'Upload'}
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
