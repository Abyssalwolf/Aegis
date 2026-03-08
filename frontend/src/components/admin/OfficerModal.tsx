'use client';

import { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

// Map of standard ranks to their clearance levels
const RANK_CLEARANCE_MAP: Record<string, number> = {
    'Constable': 1,
    'Head Constable': 2,
    'Assistant Sub Inspector (ASI)': 3,
    'Sub Inspector (SI)': 4,
    'Inspector': 5,
    'DSP / ACP': 6,
    'SP': 7,
    'DIG': 8,
    'IG': 9,
    'ADGP': 10,
    'DGP': 11,
};

const officerSchema = z.object({
    username: z.string().min(3, "Username must be at least 3 characters"),
    password: z.string().min(6, "Password must be at least 6 characters").optional(),
    rank: z.string().min(1, "Rank is required"),
    badge_number: z.string().optional(),
    station_name: z.string().min(1, "Station name is required"),
    clearance_level: z.number().min(1).max(11),
});

interface OfficerModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSave: () => void;
    officer?: {
        id: string;
        username: string;
        rank?: string;
        badge_number?: string;
        station_name?: string;
        clearance_level?: number;
    }; // If provided, we are in edit mode
    token: string;
}

export function OfficerModal({ isOpen, onClose, onSave, officer, token }: OfficerModalProps) {
    const [error, setError] = useState<string | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const isEditMode = !!officer;

    // We need to make the password optional if we are editing an officer
    const schema = isEditMode
        ? officerSchema
        : officerSchema.extend({ password: z.string().min(6, "Password must be at least 6 characters") });

    const { register, handleSubmit, setValue, watch, reset, formState: { errors } } = useForm<z.infer<typeof schema>>({
        resolver: zodResolver(schema),
        defaultValues: {
            username: '',
            password: '',
            rank: '',
            badge_number: '',
            station_name: '',
            clearance_level: 1,
        }
    });

    const selectedRank = watch('rank');

    // Auto-fill clearance level based on rank
    useEffect(() => {
        if (selectedRank && RANK_CLEARANCE_MAP[selectedRank]) {
            setValue('clearance_level', RANK_CLEARANCE_MAP[selectedRank]);
        }
    }, [selectedRank, setValue]);

    // Reset form when modal opens/closes or officer changes
    useEffect(() => {
        if (isOpen) {
            if (officer) {
                reset({
                    username: officer.username,
                    rank: officer.rank || '',
                    badge_number: officer.badge_number || '',
                    station_name: officer.station_name || '',
                    clearance_level: officer.clearance_level || 1,
                    password: '', // Don't show existing password
                });
            } else {
                reset({
                    username: '',
                    password: '',
                    rank: '',
                    badge_number: '',
                    station_name: '',
                    clearance_level: 1,
                });
            }
            setError(null);
        }
    }, [isOpen, officer, reset]);

    async function onSubmit(data: z.infer<typeof schema>) {
        setIsSubmitting(true);
        setError(null);

        const endpoint = isEditMode
            ? `http://localhost:8000/api/v1/admin/officers/${officer.id}`
            : 'http://localhost:8000/api/v1/admin/officers';

        const method = isEditMode ? 'PATCH' : 'POST';

        // Build payload. Omit password if in edit mode and left blank
        const payload: Record<string, string | number | boolean> = {
            role: 'OFFICER',
            username: data.username,
            rank: data.rank,
            clearance_level: data.clearance_level,
            badge_number: data.badge_number || '',
            station_name: data.station_name,
        };

        if (data.password) {
            payload.password = data.password;
        }

        if (!isEditMode) {
            payload.is_active = true;
        }

        try {
            const res = await fetch(endpoint, {
                method,
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Failed to save officer');
            }

            onSave(); // Refresh list and close
            onClose();
        } catch (e: Error | any) {
            setError(e.message);
        } finally {
            setIsSubmitting(false);
        }
    }

    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                    <DialogTitle>{isEditMode ? 'Edit Officer' : 'Add New Officer'}</DialogTitle>
                    <DialogDescription>
                        {isEditMode ? 'Update officer details here.' : 'Create a new officer account.'} Fill in the details below.
                    </DialogDescription>
                </DialogHeader>
                <form onSubmit={handleSubmit(onSubmit)} className="space-y-4 py-4">

                    {error && (
                        <div className="p-3 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-md">
                            {error}
                        </div>
                    )}

                    <div className="grid gap-2">
                        <Label htmlFor="username">Username / Full Name</Label>
                        <Input id="username" {...register('username')} disabled={isEditMode} placeholder="e.g. rahul.singh" />
                        {errors.username && <p className="text-xs text-destructive">{errors.username.message as string}</p>}
                    </div>

                    <div className="grid gap-2">
                        <Label htmlFor="password">{isEditMode ? 'New Password (leave blank to keep current)' : 'Temporary Password'}</Label>
                        <Input id="password" type="password" {...register('password')} placeholder="Password" />
                        {errors.password && <p className="text-xs text-destructive">{errors.password.message as string}</p>}
                    </div>

                    <div className="grid gap-2">
                        <Label htmlFor="rank">Rank</Label>
                        <select
                            id="rank"
                            {...register('rank')}
                            className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            <option value="">Select a rank...</option>
                            {Object.keys(RANK_CLEARANCE_MAP).map((rank) => (
                                <option key={rank} value={rank}>{rank}</option>
                            ))}
                        </select>
                        {errors.rank && <p className="text-xs text-destructive">{errors.rank.message as string}</p>}
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div className="grid gap-2">
                            <Label htmlFor="clearance_level">Clearance Level (1-11)</Label>
                            <Input
                                id="clearance_level"
                                type="number"
                                {...register('clearance_level', { valueAsNumber: true })}
                            />
                            {errors.clearance_level && <p className="text-xs text-destructive">{errors.clearance_level.message as string}</p>}
                        </div>

                        <div className="grid gap-2">
                            <Label htmlFor="badge_number">Badge Number</Label>
                            <Input id="badge_number" {...register('badge_number')} placeholder="e.g. KL-2312" />
                        </div>
                    </div>

                    <div className="grid gap-2">
                        <Label htmlFor="station_name">Station Name</Label>
                        <Input id="station_name" {...register('station_name')} placeholder="e.g. Kochi Central" />
                        {errors.station_name && <p className="text-xs text-destructive">{errors.station_name.message as string}</p>}
                    </div>

                    <DialogFooter className="pt-4">
                        <Button type="button" variant="outline" onClick={onClose} disabled={isSubmitting}>
                            Cancel
                        </Button>
                        <Button type="submit" disabled={isSubmitting}>
                            {isSubmitting ? 'Saving...' : 'Save Officer'}
                        </Button>
                    </DialogFooter>
                </form>
            </DialogContent>
        </Dialog>
    );
}
