'use client';

import { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { getApiV1Url } from '@/lib/api';

const caseSchema = z.object({
    title: z.string().min(3, "Title must be at least 3 characters"),
    description: z.string().min(10, "Description must be at least 10 characters"),
    required_clearance_level: z.number().min(1).max(11),
});

interface Officer {
    id: string;
    username: string;
    rank: string | null;
}

interface CreateCaseModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSave: () => void;
    token: string;
}

export function CreateCaseModal({ isOpen, onClose, onSave, token }: CreateCaseModalProps) {
    const [error, setError] = useState<string | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Officer Assignment State
    const [officers, setOfficers] = useState<Officer[]>([]);
    const [selectedOfficerIds, setSelectedOfficerIds] = useState<string[]>([]);
    const [isLoadingOfficers, setIsLoadingOfficers] = useState(false);

    const { register, handleSubmit, reset, formState: { errors } } = useForm<z.infer<typeof caseSchema>>({
        resolver: zodResolver(caseSchema),
        defaultValues: {
            title: '',
            description: '',
            required_clearance_level: 1,
        }
    });

    useEffect(() => {
        if (isOpen && token) {
            fetchOfficers();
        }
    }, [isOpen, token]);

    const fetchOfficers = async () => {
        setIsLoadingOfficers(true);
        try {
            const res = await fetch(`${getApiV1Url()}/officer/list?limit=500`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            if (res.ok) {
                const data = await res.json();
                setOfficers(Array.isArray(data) ? data : (data.items ?? []));
            }
        } catch (e) {
            console.error("Failed to fetch officers", e);
        } finally {
            setIsLoadingOfficers(false);
        }
    };

    const handleClose = () => {
        reset();
        setError(null);
        setSelectedOfficerIds([]);
        onClose();
    };

    const toggleOfficerSelection = (id: string) => {
        setSelectedOfficerIds(prev =>
            prev.includes(id) ? prev.filter(oId => oId !== id) : [...prev, id]
        );
    };

    async function onSubmit(data: z.infer<typeof caseSchema>) {
        setIsSubmitting(true);
        setError(null);

        try {
            const res = await fetch(`${getApiV1Url()}/cases`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    title: data.title,
                    description: data.description,
                    required_clearance_level: data.required_clearance_level,
                    status: 'OPEN',
                    assigned_officer_ids: selectedOfficerIds,
                })
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Failed to create case');
            }

            onSave(); // Refresh list
            handleClose();
        } catch (e: Error | any) {
            setError(e.message);
        } finally {
            setIsSubmitting(false);
        }
    }

    return (
        <Dialog open={isOpen} onOpenChange={(open) => !open && handleClose()}>
            <DialogContent className="sm:max-w-[500px]">
                <DialogHeader>
                    <DialogTitle>Create New Case</DialogTitle>
                    <DialogDescription>
                        Spawn a new investigation. Provide the initial details below.
                    </DialogDescription>
                </DialogHeader>
                <div className="max-h-[70vh] overflow-y-auto px-1 -mx-1">
                    <form onSubmit={handleSubmit(onSubmit)} className="space-y-4 py-4">

                        {error && (
                            <div className="p-3 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-md">
                                {error}
                            </div>
                        )}

                        <div className="grid gap-2">
                            <Label htmlFor="title">Case Title</Label>
                            <Input id="title" placeholder="e.g. Operation Silk" {...register('title')} />
                            {errors.title && <p className="text-xs text-destructive">{errors.title.message as string}</p>}
                        </div>

                        <div className="grid gap-2">
                            <Label htmlFor="description">Initial Overview & Details</Label>
                            <textarea
                                id="description"
                                className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                                placeholder="Provide a detailed description of the case..."
                                {...register('description')}
                            />
                            {errors.description && <p className="text-xs text-destructive">{errors.description.message as string}</p>}
                        </div>

                        <div className="grid gap-2">
                            <Label htmlFor="required_clearance_level">Required Minimum Clearance Level</Label>
                            <Input
                                id="required_clearance_level"
                                type="number"
                                {...register('required_clearance_level', { valueAsNumber: true })}
                            />
                            {errors.required_clearance_level && <p className="text-xs text-destructive">{errors.required_clearance_level.message as string}</p>}
                        </div>

                        <div className="grid gap-2 mt-2">
                            <Label>Assign Initial Officers</Label>
                            {isLoadingOfficers ? (
                                <p className="text-sm text-muted-foreground italic">Loading officers...</p>
                            ) : officers.length === 0 ? (
                                <p className="text-sm text-muted-foreground italic">No officers available.</p>
                            ) : (
                                <div className="border border-border rounded-md max-h-[160px] overflow-y-auto p-2 space-y-1 bg-muted/20">
                                    {officers.map((officer) => (
                                        <div
                                            key={officer.id}
                                            onClick={() => toggleOfficerSelection(officer.id)}
                                            className={`flex items-center gap-3 p-2 rounded-md cursor-pointer transition-colors ${selectedOfficerIds.includes(officer.id) ? 'bg-primary/10 border border-primary/20' : 'hover:bg-muted border border-transparent'}`}
                                        >
                                            <div className={`w-4 h-4 rounded border flex items-center justify-center transition-colors ${selectedOfficerIds.includes(officer.id) ? 'bg-primary border-primary' : 'bg-background border-input'}`}>
                                                {selectedOfficerIds.includes(officer.id) && (
                                                    <svg width="10" height="8" viewBox="0 0 10 8" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                        <path d="M1 4L3.5 6.5L9 1" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                                    </svg>
                                                )}
                                            </div>
                                            <div className="flex flex-col">
                                                <span className="text-sm font-medium text-foreground leading-none">{officer.username}</span>
                                                <span className="text-xs text-muted-foreground mt-1">{officer.rank || 'Officer'}</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                            <p className="text-xs text-muted-foreground">Select one or more officers to assign to this case initially. You can assign more later.</p>
                        </div>

                        <DialogFooter className="pt-4 sticky bottom-0 bg-background">
                            <Button type="button" variant="outline" onClick={handleClose} disabled={isSubmitting}>
                                Cancel
                            </Button>
                            <Button type="submit" disabled={isSubmitting}>
                                {isSubmitting ? 'Creating...' : 'Create Case'}
                            </Button>
                        </DialogFooter>
                    </form>
                </div>
            </DialogContent>
        </Dialog>
    );
}
