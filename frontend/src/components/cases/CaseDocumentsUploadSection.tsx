'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Plus } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { UploadEvidenceDialog } from '@/components/cases/UploadEvidenceDialog';

export function CaseDocumentsUploadSection({ caseId, token }: { caseId: string; token: string }) {
    const [open, setOpen] = useState(false);
    const router = useRouter();

    return (
        <>
            <Button
                type="button"
                variant="ghost"
                className="h-8 w-8 p-0"
                title="Upload document (RAG + Insights agents)"
                onClick={() => setOpen(true)}
            >
                <Plus className="w-4 h-4" />
            </Button>
            <UploadEvidenceDialog
                open={open}
                onOpenChange={setOpen}
                caseId={caseId}
                token={token}
                onUploaded={() => router.refresh()}
            />
        </>
    );
}
