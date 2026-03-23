'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { UserMinus, UserPlus, Loader2 } from 'lucide-react';
import { getApiV1Url } from '@/lib/api';

interface Officer {
    id: string;
    username: string;
    rank: string | null;
    clearance_level: number;
}

interface ManageOfficersModalProps {
    isOpen: boolean;
    onClose: () => void;
    caseId: string;
    token: string;
}

export function ManageOfficersModal({ isOpen, onClose, caseId, token }: ManageOfficersModalProps) {
    const router = useRouter();
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    const [assignedOfficers, setAssignedOfficers] = useState<Officer[]>([]);
    const [allActiveOfficers, setAllActiveOfficers] = useState<Officer[]>([]);
    const [selectedOfficerId, setSelectedOfficerId] = useState<string>('');
    const [creatorId, setCreatorId] = useState<string>('');

    const fetchData = async () => {
        setIsLoading(true);
        setError(null);
        try {
            // Fetch case details to find creator
            const caseRes = await fetch(`${getApiV1Url()}/cases/${caseId}`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            if (caseRes.ok) {
                const caseData = await caseRes.json();
                setCreatorId(caseData.created_by);
            }

            // Fetch currently assigned officers
            const assignedRes = await fetch(`${getApiV1Url()}/cases/${caseId}/officers`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            if (assignedRes.ok) {
                const asgData = await assignedRes.json();
                setAssignedOfficers(asgData);
            }

            // Fetch all active officers to populate dropdown
            const rosterRes = await fetch(`${getApiV1Url()}/officer/list?limit=500`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            if (rosterRes.ok) {
                const rosterData = await rosterRes.json();
                const list = Array.isArray(rosterData) ? rosterData : (rosterData.items ?? []);
                setAllActiveOfficers(list);
            }
        } catch (e: any) {
            setError("Failed to fetch roster data.");
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        if (isOpen && token) {
            fetchData();
        }
    }, [isOpen, token, caseId]);

    const handleClose = () => {
        setError(null);
        setSelectedOfficerId('');
        onClose();
        router.refresh(); // Refresh the parent page to update the assigned personnel sidebar
    };

    const handleAssign = async () => {
        if (!selectedOfficerId) return;
        setIsLoading(true);
        setError(null);

        try {
            const res = await fetch(`${getApiV1Url()}/cases/${caseId}/officers`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ officer_id: selectedOfficerId })
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Failed to assign officer');
            }

            setSelectedOfficerId('');
            await fetchData(); // Refresh the lists in the modal
        } catch (e: any) {
            setError(e.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleRemove = async (officerId: string) => {
        setIsLoading(true);
        setError(null);

        try {
            const res = await fetch(`${getApiV1Url()}/cases/${caseId}/officers/${officerId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Failed to remove officer');
            }

            await fetchData(); // Refresh lists
        } catch (e: any) {
            setError(e.message);
        } finally {
            setIsLoading(false);
        }
    };

    // Filter available officers (exclude already assigned ones)
    const availableOfficers = allActiveOfficers.filter(
        ao => !assignedOfficers.find(asg => asg.id === ao.id)
    );

    return (
        <Dialog open={isOpen} onOpenChange={(open) => !open && handleClose()}>
            <DialogContent className="sm:max-w-[500px] max-h-[85vh] overflow-hidden flex flex-col">
                <DialogHeader>
                    <DialogTitle>Manage Assigned Personnel</DialogTitle>
                    <DialogDescription>
                        Add or remove officers assigned to working on this case.
                    </DialogDescription>
                </DialogHeader>

                {error && (
                    <div className="p-3 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-md">
                        {error}
                    </div>
                )}

                <div className="flex-1 overflow-y-auto pr-2 space-y-6">
                    {/* Add Officer Section */}
                    <div className="space-y-4 pt-4 border-t border-border">
                        <div className="flex items-center gap-2 text-primary">
                            <UserPlus className="w-5 h-5" />
                            <h3 className="font-semibold text-foreground">Assign New Officer</h3>
                        </div>
                        <div className="flex gap-2 items-end">
                            <div className="grid gap-2 flex-1">
                                <Label htmlFor="officer-select" className="sr-only">Select Officer</Label>
                                <select
                                    id="officer-select"
                                    value={selectedOfficerId}
                                    onChange={(e) => setSelectedOfficerId(e.target.value)}
                                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background disabled:cursor-not-allowed disabled:opacity-50"
                                    disabled={isLoading}
                                >
                                    <option value="">Select an officer to assign...</option>
                                    {availableOfficers.map(o => (
                                        <option key={o.id} value={o.id}>{o.username} ({o.rank || 'Officer'}, Lvl {o.clearance_level})</option>
                                    ))}
                                </select>
                            </div>
                            <Button
                                onClick={handleAssign}
                                disabled={!selectedOfficerId || isLoading}
                                className="w-24 gap-2"
                            >
                                {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Assign'}
                            </Button>
                        </div>
                    </div>

                    {/* Current Roster Section */}
                    <div className="space-y-4 pt-4 border-t border-border">
                        <div className="flex items-center gap-2 text-foreground">
                            <UserMinus className="w-5 h-5 text-muted-foreground" />
                            <h3 className="font-semibold">Currently Assigned ({assignedOfficers.length})</h3>
                        </div>

                        {isLoading && assignedOfficers.length === 0 ? (
                            <div className="flex justify-center py-4 text-muted-foreground">
                                <Loader2 className="w-5 h-5 animate-spin" />
                            </div>
                        ) : assignedOfficers.length === 0 ? (
                            <div className="text-sm text-muted-foreground p-4 text-center border border-dashed rounded-lg bg-muted/30">
                                No officers currently assigned to this case.
                            </div>
                        ) : (
                            <ul className="space-y-2">
                                {assignedOfficers.map(officer => (
                                    <li key={officer.id} className="flex items-center justify-between p-3 rounded-md border border-border bg-card/50">
                                        <div className="flex flex-col">
                                            <span className="text-sm font-medium">
                                                {officer.username}
                                                {officer.id === creatorId && <span className="ml-2 text-[10px] bg-primary/20 text-primary px-1.5 py-0.5 rounded uppercase tracking-wide">Creator</span>}
                                            </span>
                                            <span className="text-xs text-muted-foreground">{officer.rank || 'Officer'} · Level {officer.clearance_level}</span>
                                        </div>
                                        {officer.id !== creatorId && (
                                            <Button
                                                variant="outline"
                                                className="h-8 px-2 text-xs text-destructive hover:bg-destructive hover:text-destructive-foreground hover:border-destructive transition-colors"
                                                onClick={() => handleRemove(officer.id)}
                                                disabled={isLoading}
                                            >
                                                Remove
                                            </Button>
                                        )}
                                    </li>
                                ))}
                            </ul>
                        )}
                    </div>
                </div>

                <DialogFooter className="pt-4 border-t border-border sm:justify-start">
                    <Button type="button" variant="outline" onClick={handleClose}>
                        Done
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
