'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { AlertTriangle, UserMinus } from 'lucide-react';
import { getApiV1Url } from '@/lib/api';

interface Officer {
    id: string;
    username: string;
    rank: string | null;
    clearance_level: number;
}

interface CaseSettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
    caseId: string;
    caseTitle: string;
    token: string;
}

export function CaseSettingsModal({ isOpen, onClose, caseId, caseTitle, token }: CaseSettingsModalProps) {
    const router = useRouter();
    const [error, setError] = useState<string | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Transfer State
    const [officers, setOfficers] = useState<Officer[]>([]);
    const [selectedOfficerId, setSelectedOfficerId] = useState<string>('');
    const [transferConfirmText, setTransferConfirmText] = useState('');

    // Delete State
    const [deleteConfirmText, setDeleteConfirmText] = useState('');

    useEffect(() => {
        if (isOpen && token) {
            fetchOfficers();
        }
    }, [isOpen, token]);

    const fetchOfficers = async () => {
        try {
            const res = await fetch(`${getApiV1Url()}/officer/list?limit=500`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            if (res.ok) {
                const data = await res.json();
                const list = Array.isArray(data) ? data : (data.items ?? []);
                // Filter officers to only those with clearance >= 4
                setOfficers(list.filter((o: Officer) => (o.clearance_level || 0) >= 4));
            }
        } catch (e) {
            console.error("Failed to fetch officers", e);
        }
    };

    const handleClose = () => {
        setError(null);
        setSelectedOfficerId('');
        setTransferConfirmText('');
        setDeleteConfirmText('');
        onClose();
    };

    const handleTransfer = async () => {
        if (transferConfirmText !== caseTitle) {
            setError("Case title does not match. Transfer cancelled.");
            return;
        }
        if (!selectedOfficerId) {
            setError("Please select an officer to transfer to.");
            return;
        }

        setIsSubmitting(true);
        setError(null);

        try {
            const res = await fetch(`${getApiV1Url()}/cases/${caseId}/transfer`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ new_owner_id: selectedOfficerId })
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Failed to transfer case');
            }

            handleClose();
            router.refresh();
        } catch (e: Error | any) {
            setError(e.message);
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleDelete = async () => {
        if (deleteConfirmText !== caseTitle) {
            setError("Case title does not match. Deletion cancelled.");
            return;
        }

        setIsSubmitting(true);
        setError(null);

        try {
            const res = await fetch(`${getApiV1Url()}/cases/${caseId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Failed to delete case');
            }

            handleClose();
            router.push('/officer/dashboard');
        } catch (e: Error | any) {
            setError(e.message);
            setIsSubmitting(false);
        }
    };

    return (
        <Dialog open={isOpen} onOpenChange={(open) => !open && handleClose()}>
            <DialogContent className="sm:max-w-[550px] max-h-[90vh] overflow-y-auto">
                <DialogHeader>
                    <DialogTitle>Case Settings</DialogTitle>
                    <DialogDescription>
                        Manage ownership and danger zone settings for this case.
                    </DialogDescription>
                </DialogHeader>

                {error && (
                    <div className="p-3 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-md mt-4">
                        {error}
                    </div>
                )}

                <div className="space-y-8 py-4">
                    {/* Transfer Section */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 text-primary border-b border-border pb-2">
                            <UserMinus className="w-5 h-5" />
                            <h3 className="font-semibold text-foreground">Transfer Ownership</h3>
                        </div>
                        <p className="text-sm text-muted-foreground">
                            Transfer this case to another officer. Only officers with Clearance Level 4 or higher can accept ownership.
                        </p>

                        <div className="grid gap-2">
                            <Label htmlFor="officer-select">Select New Owner</Label>
                            <select
                                id="officer-select"
                                value={selectedOfficerId}
                                onChange={(e) => setSelectedOfficerId(e.target.value)}
                                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                <option value="">Select an officer...</option>
                                {officers.map(o => (
                                    <option key={o.id} value={o.id}>{o.username} ({o.rank || 'Officer'}, Lvl {o.clearance_level})</option>
                                ))}
                            </select>
                        </div>

                        <div className="grid gap-2">
                            <Label htmlFor="transfer-confirm">Type <span className="font-mono text-primary font-bold">{caseTitle}</span> to confirm</Label>
                            <Input
                                id="transfer-confirm"
                                value={transferConfirmText}
                                onChange={(e) => setTransferConfirmText(e.target.value)}
                                placeholder={caseTitle}
                            />
                        </div>

                        <Button
                            onClick={handleTransfer}
                            disabled={isSubmitting || transferConfirmText !== caseTitle || !selectedOfficerId}
                            className="w-full"
                        >
                            {isSubmitting ? 'Processing...' : 'Transfer Ownership'}
                        </Button>
                    </div>

                    {/* Danger Zone */}
                    <div className="space-y-4 pt-4">
                        <div className="flex items-center gap-2 text-destructive border-b border-destructive/20 pb-2">
                            <AlertTriangle className="w-5 h-5" />
                            <h3 className="font-semibold">Danger Zone</h3>
                        </div>
                        <p className="text-sm text-muted-foreground">
                            Permanently delete this case and all associated assignments and logs. This action cannot be undone.
                        </p>

                        <div className="grid gap-2">
                            <Label htmlFor="delete-confirm" className="text-destructive">Type <span className="font-mono font-bold">{caseTitle}</span> to confirm deletion</Label>
                            <Input
                                id="delete-confirm"
                                className="border-destructive/50 focus-visible:ring-destructive/30"
                                value={deleteConfirmText}
                                onChange={(e) => setDeleteConfirmText(e.target.value)}
                                placeholder={caseTitle}
                            />
                        </div>

                        <Button
                            onClick={handleDelete}
                            disabled={isSubmitting || deleteConfirmText !== caseTitle}
                            className="w-full bg-destructive hover:bg-destructive/90 text-destructive-foreground"
                        >
                            {isSubmitting ? 'Processing...' : 'Permanently Delete Case'}
                        </Button>
                    </div>
                </div>

                <DialogFooter className="pt-2 border-t border-border sm:justify-start">
                    <Button type="button" variant="outline" onClick={handleClose} disabled={isSubmitting}>
                        Close Settings
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
