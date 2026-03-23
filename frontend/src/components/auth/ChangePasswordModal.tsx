'use client';

import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { logoutAction } from '@/lib/auth';
import { getApiV1Url } from '@/lib/api';

const passwordSchema = z.object({
    current_password: z.string().min(1, "Current password is required"),
    new_password: z.string().min(6, "Password must be at least 6 characters"),
    confirm_password: z.string().min(1, "Please confirm your password"),
}).refine((data) => data.new_password === data.confirm_password, {
    message: "Passwords do not match",
    path: ["confirm_password"],
});

interface ChangePasswordModalProps {
    isOpen: boolean;
    onClose: () => void;
    token: string;
}

export function ChangePasswordModal({ isOpen, onClose, token }: ChangePasswordModalProps) {
    const [error, setError] = useState<string | null>(null);
    const [successMsg, setSuccessMsg] = useState<string | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const { register, handleSubmit, reset, formState: { errors } } = useForm<z.infer<typeof passwordSchema>>({
        resolver: zodResolver(passwordSchema),
        defaultValues: {
            current_password: '',
            new_password: '',
            confirm_password: '',
        }
    });

    const handleClose = () => {
        reset();
        setError(null);
        setSuccessMsg(null);
        onClose();
    };

    async function onSubmit(data: z.infer<typeof passwordSchema>) {
        setIsSubmitting(true);
        setError(null);
        setSuccessMsg(null);

        try {
            const res = await fetch(`${getApiV1Url()}/auth/change-password`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    current_password: data.current_password,
                    new_password: data.new_password
                })
            });

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || 'Failed to change password');
            }

            setSuccessMsg("Password changed successfully! Please sign in again.");
            setTimeout(() => {
                logoutAction();
            }, 2000);

        } catch (e: any) {
            setError(e.message);
        } finally {
            setIsSubmitting(false);
        }
    }

    return (
        <Dialog open={isOpen} onOpenChange={(open) => !open && handleClose()}>
            <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                    <DialogTitle>Change Password</DialogTitle>
                    <DialogDescription>
                        Update your account password securely. You will be asked to sign in again after a successful update.
                    </DialogDescription>
                </DialogHeader>
                <form onSubmit={handleSubmit(onSubmit)} className="space-y-4 py-4">

                    {error && (
                        <div className="p-3 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-md">
                            {error}
                        </div>
                    )}

                    {successMsg && (
                        <div className="p-3 text-sm text-green-500 bg-green-500/10 border border-green-500/20 rounded-md">
                            {successMsg}
                        </div>
                    )}

                    <div className="grid gap-2">
                        <Label htmlFor="current_password">Current Password</Label>
                        <Input id="current_password" type="password" {...register('current_password')} />
                        {errors.current_password && <p className="text-xs text-destructive">{errors.current_password.message as string}</p>}
                    </div>

                    <div className="grid gap-2">
                        <Label htmlFor="new_password">New Password</Label>
                        <Input id="new_password" type="password" {...register('new_password')} />
                        {errors.new_password && <p className="text-xs text-destructive">{errors.new_password.message as string}</p>}
                    </div>

                    <div className="grid gap-2">
                        <Label htmlFor="confirm_password">Confirm New Password</Label>
                        <Input id="confirm_password" type="password" {...register('confirm_password')} />
                        {errors.confirm_password && <p className="text-xs text-destructive">{errors.confirm_password.message as string}</p>}
                    </div>

                    <DialogFooter className="pt-4">
                        <Button type="button" variant="outline" onClick={handleClose} disabled={isSubmitting || !!successMsg}>
                            Cancel
                        </Button>
                        <Button type="submit" disabled={isSubmitting || !!successMsg}>
                            {isSubmitting ? 'Updating...' : 'Change Password'}
                        </Button>
                    </DialogFooter>
                </form>
            </DialogContent>
        </Dialog>
    );
}
