'use client';

import { useState, useRef } from 'react';
import { UploadCloud, FileText, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
    Dialog, DialogContent, DialogHeader, DialogTitle,
    DialogDescription, DialogFooter,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

const EVIDENCE_CATEGORIES = [
    { value: 'case_diary', label: 'Case Diary' },
    { value: 'fir_file', label: 'FIR File' },
    { value: 'statement_file', label: 'Statement File' },
    { value: 'scene_of_crime', label: 'Scene of Crime File' },
    { value: 'forensic_evidence', label: 'Forensic / Evidence File' },
    { value: 'property_seizure', label: 'Property / Seizure File' },
    { value: 'arrest_remand', label: 'Arrest & Remand File' },
    { value: 'other', label: 'Other' },
] as const;

export type EvidenceCategory = typeof EVIDENCE_CATEGORIES[number]['value'];

export const CATEGORY_LABELS: Record<string, string> = Object.fromEntries(
    EVIDENCE_CATEGORIES.map(c => [c.value, c.label])
);

interface UploadModalProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    onUpload: (data: {
        file: File;
        displayName: string;
        evidenceCategory: EvidenceCategory;
        description: string;
    }) => void;
    uploading: boolean;
}

export default function UploadModal({ open, onOpenChange, onUpload, uploading }: UploadModalProps) {
    const [file, setFile] = useState<File | null>(null);
    const [displayName, setDisplayName] = useState('');
    const [evidenceCategory, setEvidenceCategory] = useState<EvidenceCategory>('fir_file');
    const [description, setDescription] = useState('');
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selected = e.target.files?.[0];
        if (selected) {
            setFile(selected);
            if (!displayName) {
                const nameWithoutExt = selected.name.replace(/\.[^.]+$/, '');
                setDisplayName(nameWithoutExt);
            }
        }
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!file || !displayName.trim()) return;
        onUpload({
            file,
            displayName: displayName.trim(),
            evidenceCategory,
            description: description.trim(),
        });
    };

    const resetForm = () => {
        setFile(null);
        setDisplayName('');
        setEvidenceCategory('fir_file');
        setDescription('');
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    const handleOpenChange = (v: boolean) => {
        if (!v) resetForm();
        onOpenChange(v);
    };

    return (
        <Dialog open={open} onOpenChange={handleOpenChange}>
            <DialogContent className="sm:max-w-md">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <UploadCloud className="w-5 h-5 text-primary" />
                        Upload Evidence File
                    </DialogTitle>
                    <DialogDescription>
                        Provide details about the file to help with identification and retrieval.
                    </DialogDescription>
                </DialogHeader>

                <form onSubmit={handleSubmit} className="space-y-4">
                    {/* File picker */}
                    <div className="space-y-2">
                        <Label htmlFor="upload-file">File</Label>
                        {file ? (
                            <div className="flex items-center gap-2 p-3 rounded-lg border border-border bg-muted/30">
                                <FileText className="w-4 h-4 text-primary shrink-0" />
                                <span className="text-sm truncate flex-1">{file.name}</span>
                                <button
                                    type="button"
                                    onClick={() => { setFile(null); if (fileInputRef.current) fileInputRef.current.value = ''; }}
                                    className="text-xs text-muted-foreground hover:text-foreground"
                                >
                                    Change
                                </button>
                            </div>
                        ) : (
                            <button
                                type="button"
                                onClick={() => fileInputRef.current?.click()}
                                className="w-full flex flex-col items-center justify-center p-6 border-2 border-dashed border-border rounded-lg text-center hover:border-primary/50 hover:bg-primary/5 transition-colors"
                            >
                                <UploadCloud className="w-6 h-6 text-muted-foreground mb-1.5" />
                                <span className="text-xs text-muted-foreground">Click to select a PDF or image file</span>
                            </button>
                        )}
                        <input
                            ref={fileInputRef}
                            id="upload-file"
                            type="file"
                            accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp,.webp"
                            className="hidden"
                            onChange={handleFileChange}
                        />
                    </div>

                    {/* Document Name */}
                    <div className="space-y-2">
                        <Label htmlFor="display-name">Document Name</Label>
                        <Input
                            id="display-name"
                            placeholder="e.g. FIR - Rajesh Kumar theft case"
                            value={displayName}
                            onChange={e => setDisplayName(e.target.value)}
                        />
                    </div>

                    {/* Evidence Category */}
                    <div className="space-y-2">
                        <Label htmlFor="evidence-category">File Type</Label>
                        <select
                            id="evidence-category"
                            value={evidenceCategory}
                            onChange={e => setEvidenceCategory(e.target.value as EvidenceCategory)}
                            className="w-full h-10 rounded-md border border-border bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50"
                        >
                            {EVIDENCE_CATEGORIES.map(cat => (
                                <option key={cat.value} value={cat.value}>
                                    {cat.label}
                                </option>
                            ))}
                        </select>
                    </div>

                    {/* Description */}
                    <div className="space-y-2">
                        <Label htmlFor="description">
                            Notes <span className="text-muted-foreground font-normal">(optional)</span>
                        </Label>
                        <textarea
                            id="description"
                            placeholder="Brief description or remarks about this document…"
                            value={description}
                            onChange={e => setDescription(e.target.value)}
                            rows={2}
                            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50 resize-none"
                        />
                    </div>

                    <DialogFooter>
                        <Button
                            type="button"
                            variant="ghost"
                            onClick={() => handleOpenChange(false)}
                            disabled={uploading}
                        >
                            Cancel
                        </Button>
                        <Button
                            type="submit"
                            disabled={!file || !displayName.trim() || uploading}
                            className="gap-2"
                        >
                            {uploading ? (
                                <>
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    Uploading…
                                </>
                            ) : (
                                <>
                                    <UploadCloud className="w-4 h-4" />
                                    Upload & Ingest
                                </>
                            )}
                        </Button>
                    </DialogFooter>
                </form>
            </DialogContent>
        </Dialog>
    );
}
