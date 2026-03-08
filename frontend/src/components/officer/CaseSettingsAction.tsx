'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Settings } from 'lucide-react';
import { CaseSettingsModal } from '@/components/officer/CaseSettingsModal';

interface CaseSettingsActionProps {
    token: string;
    caseId: string;
    caseTitle: string;
}

export function CaseSettingsAction({ token, caseId, caseTitle }: CaseSettingsActionProps) {
    const [isModalOpen, setIsModalOpen] = useState(false);

    return (
        <>
            <Button variant="outline" className="gap-2 h-9 px-3 text-xs" onClick={() => setIsModalOpen(true)}>
                <Settings className="w-4 h-4" />
                Settings
            </Button>
            <CaseSettingsModal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                caseId={caseId}
                caseTitle={caseTitle}
                token={token}
            />
        </>
    );
}
