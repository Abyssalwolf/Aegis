'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { CreateCaseModal } from '@/components/officer/CreateCaseModal';
import { useRouter } from 'next/navigation';

export function CreateCaseAction({ token }: { token: string }) {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const router = useRouter();

    const handleSave = () => {
        setIsModalOpen(false);
        router.refresh(); // Refresh the server component to fetch new cases
    };

    return (
        <>
            <Button className="h-full" onClick={() => setIsModalOpen(true)}>
                Create New Case
            </Button>
            <CreateCaseModal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                onSave={handleSave}
                token={token}
            />
        </>
    );
}
