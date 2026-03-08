'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Users } from 'lucide-react';
import { ManageOfficersModal } from '@/components/officer/ManageOfficersModal';

interface ManageOfficersActionProps {
    token: string;
    caseId: string;
}

export function ManageOfficersAction({ token, caseId }: ManageOfficersActionProps) {
    const [isModalOpen, setIsModalOpen] = useState(false);

    return (
        <>
            <Button variant="outline" className="h-8 px-2 text-xs gap-1" onClick={() => setIsModalOpen(true)}>
                <Users className="w-3.5 h-3.5" />
                Manage
            </Button>
            <ManageOfficersModal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                caseId={caseId}
                token={token}
            />
        </>
    );
}
