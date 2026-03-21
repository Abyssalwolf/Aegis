'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { FileText, Trash2, Loader2, UploadCloud, Plus } from 'lucide-react';
import UploadModal, { CATEGORY_LABELS } from '@/components/cases/UploadModal';

interface DocumentItem {
    id: string;
    document_type: string;
    file_path: string;
    filename: string | null;
    display_name: string | null;
    evidence_category: string | null;
}

interface DocumentListProps {
    documents: DocumentItem[];
    caseId: string;
    token: string;
    canDelete: boolean;
    /** When true, show Documents header + upload button and the same metadata modal as RAG chat */
    enableUpload?: boolean;
}

export default function DocumentList({
    documents: initialDocs,
    caseId,
    token,
    canDelete,
    enableUpload = false,
}: DocumentListProps) {
    const router = useRouter();
    const [documents, setDocuments] = useState(initialDocs);
    const [deletingId, setDeletingId] = useState<string | null>(null);
    const [uploadModalOpen, setUploadModalOpen] = useState(false);
    const [uploading, setUploading] = useState(false);

    useEffect(() => {
        setDocuments(initialDocs);
    }, [initialDocs]);

    const handleUpload = async (data: {
        file: File;
        displayName: string;
        evidenceCategory: string;
        description: string;
    }) => {
        setUploading(true);
        const form = new FormData();
        form.append('file', data.file);
        form.append('document_type', data.file.type || 'application/octet-stream');
        form.append('display_name', data.displayName);
        form.append('evidence_category', data.evidenceCategory);
        if (data.description) form.append('description', data.description);

        try {
            const res = await fetch(`http://localhost:8000/api/v1/cases/${caseId}/documents`, {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}` },
                body: form,
            });

            if (res.ok) {
                const newDoc: DocumentItem = await res.json();
                setDocuments(prev => [newDoc, ...prev]);
                setUploadModalOpen(false);
                router.refresh();
            } else {
                const err = await res.json().catch(() => ({ detail: 'Upload failed.' }));
                alert(err.detail || 'Upload failed.');
            }
        } catch {
            alert('Could not reach the server.');
        } finally {
            setUploading(false);
        }
    };

    const handleDelete = async (docId: string) => {
        if (!confirm('Are you sure you want to delete this document? Its indexed data will also be removed.')) return;

        setDeletingId(docId);
        try {
            const res = await fetch(`http://localhost:8000/api/v1/cases/${caseId}/documents/${docId}`, {
                method: 'DELETE',
                headers: { Authorization: `Bearer ${token}` },
            });

            if (res.ok) {
                setDocuments(prev => prev.filter(d => d.id !== docId));
            } else {
                const err = await res.json().catch(() => ({ detail: 'Failed to delete document.' }));
                alert(err.detail || 'Failed to delete document.');
            }
        } catch {
            alert('Could not reach the server.');
        } finally {
            setDeletingId(null);
        }
    };

    return (
        <div>
            {enableUpload && (
                <div className="flex items-center justify-between mb-4">
                    <h2 className="font-semibold flex items-center gap-2 text-foreground">
                        <UploadCloud className="w-4 h-4 text-muted-foreground" />
                        Documents
                    </h2>
                    <button
                        type="button"
                        className="h-8 w-8 inline-flex items-center justify-center rounded-md hover:bg-muted transition-colors"
                        title="Upload document"
                        onClick={() => setUploadModalOpen(true)}
                        disabled={uploading}
                    >
                        {uploading ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                            <Plus className="w-4 h-4" />
                        )}
                    </button>
                </div>
            )}

        <div className="space-y-3">
            {documents.map((doc) => {
                const catLabel = doc.evidence_category
                    ? CATEGORY_LABELS[doc.evidence_category] || doc.evidence_category
                    : null;
                return (
                    <div
                        key={doc.id}
                        className="flex items-center justify-between p-3 rounded-lg border border-border bg-background hover:bg-muted/50 transition-colors group"
                    >
                        <div className="flex items-start gap-2 flex-1 min-w-0">
                            <FileText className="w-4 h-4 text-primary mt-0.5 shrink-0" />
                            <div className="flex flex-col min-w-0">
                                <span className="text-sm font-medium group-hover:text-primary transition-colors truncate">
                                    {doc.display_name || doc.filename || doc.document_type}
                                </span>
                                {catLabel && (
                                    <span className="inline-flex items-center gap-1 mt-0.5 text-[10px] text-muted-foreground">
                                        {catLabel}
                                    </span>
                                )}
                            </div>
                        </div>
                        {canDelete && (
                            <button
                                onClick={() => handleDelete(doc.id)}
                                disabled={deletingId === doc.id}
                                className="opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-md hover:bg-red-500/20 text-muted-foreground hover:text-red-400 shrink-0 ml-2"
                                title="Delete document"
                            >
                                {deletingId === doc.id ? (
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                ) : (
                                    <Trash2 className="w-4 h-4" />
                                )}
                            </button>
                        )}
                    </div>
                );
            })}
            {documents.length === 0 && (
                <div className="text-center p-4 border border-dashed border-border rounded-lg text-sm text-muted-foreground">
                    {enableUpload ? (
                        <button
                            type="button"
                            onClick={() => setUploadModalOpen(true)}
                            className="text-primary hover:underline"
                        >
                            Upload a document
                        </button>
                    ) : (
                        'No documents attached.'
                    )}
                </div>
            )}
        </div>

            {enableUpload && (
                <UploadModal
                    open={uploadModalOpen}
                    onOpenChange={setUploadModalOpen}
                    onUpload={handleUpload}
                    uploading={uploading}
                />
            )}
        </div>
    );
}
