'use client';

import { useState } from 'react';
import { Settings } from 'lucide-react';
import { ChangePasswordModal } from '@/components/auth/ChangePasswordModal';

export function UserProfileActions({ token }: { token: string }) {
    const [isPasswordModalOpen, setIsPasswordModalOpen] = useState(false);

    return (
        <>
            <button
                onClick={() => setIsPasswordModalOpen(true)}
                className="flex items-center gap-3 w-full px-3 py-2.5 rounded-md hover:bg-muted/50 text-sm font-medium transition-colors mb-2 text-foreground"
            >
                <Settings className="w-4 h-4 text-muted-foreground" />
                Change Password
            </button>
            <ChangePasswordModal
                isOpen={isPasswordModalOpen}
                onClose={() => setIsPasswordModalOpen(false)}
                token={token}
            />
        </>
    );
}
