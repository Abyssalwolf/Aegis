'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { UserPlus, Settings, ShieldAlert, Trash2, PowerOff, Power } from 'lucide-react';
import { OfficerModal } from '@/components/admin/OfficerModal';
import { getAccessToken } from '@/lib/auth';

export default function AdminDashboard() {
    const [officers, setOfficers] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [token, setToken] = useState<string>('');

    // Modal state
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [editingOfficer, setEditingOfficer] = useState<any | null>(null);

    const fetchOfficers = async () => {
        setLoading(true);
        try {
            const currentToken = await getAccessToken();
            if (!currentToken) throw new Error("No access token found");
            setToken(currentToken);

            const res = await fetch('http://localhost:8000/api/v1/admin/officers', {
                headers: { Authorization: `Bearer ${currentToken}` },
                cache: 'no-store'
            });

            if (!res.ok) {
                if (res.status === 403) {
                    setError("Access Restricted: Only administrators can view this dashboard.");
                } else {
                    setError("Failed to load officers. Is backend running?");
                }
            } else {
                const data = await res.json();
                setOfficers(data);
                setError(null);
            }
        } catch (err: any) {
            setError(err.message || 'An unexpected error occurred');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchOfficers();
    }, []);

    const handleAddOfficer = () => {
        setEditingOfficer(null);
        setIsModalOpen(true);
    };

    const handleEditOfficer = (officer: any) => {
        setEditingOfficer(officer);
        setIsModalOpen(true);
    };

    const handleToggleStatus = async (officerId: string, currentStatus: boolean) => {
        try {
            const res = await fetch(`http://localhost:8000/api/v1/admin/officers/${officerId}/status?is_active=${!currentStatus}`, {
                method: 'PATCH',
                headers: { Authorization: `Bearer ${token}` }
            });
            if (res.ok) {
                fetchOfficers();
            } else {
                console.error("Failed to toggle status");
            }
        } catch (e) {
            console.error(e);
        }
    };

    const handleDeleteOfficer = async (officerId: string) => {
        if (!confirm("Are you sure you want to permanently delete this officer?")) return;

        try {
            const res = await fetch(`http://localhost:8000/api/v1/admin/officers/${officerId}`, {
                method: 'DELETE',
                headers: { Authorization: `Bearer ${token}` }
            });
            if (res.ok) {
                fetchOfficers();
            } else {
                console.error("Failed to delete officer");
            }
        } catch (e) {
            console.error(e);
        }
    };

    if (error && error.includes('Restricted')) {
        return (
            <div className="flex flex-col items-center justify-center p-24 text-center">
                <ShieldAlert className="w-16 h-16 text-destructive mb-4" />
                <h2 className="text-2xl font-bold">Access Restricted</h2>
                <p className="text-muted-foreground mt-2">{error}</p>
            </div>
        );
    }

    return (
        <>
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Admin Operations</h1>
                    <p className="text-muted-foreground">Manage police officers, roles, and clearance levels.</p>
                </div>
                <Button className="gap-2" onClick={handleAddOfficer}>
                    <UserPlus className="w-4 h-4" />
                    New Officer
                </Button>
            </div>

            {error && !error.includes('Restricted') && (
                <div className="mt-8 p-4 text-destructive bg-destructive/10 rounded-lg">
                    {error}
                </div>
            )}

            <div className="bg-card border border-border rounded-xl shadow-sm overflow-hidden mt-8">
                <div className="overflow-x-auto relative min-h-[400px]">
                    {loading && (
                        <div className="absolute inset-0 z-10 bg-background/50 flex items-center justify-center backdrop-blur-sm">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                        </div>
                    )}
                    <table className="w-full text-sm text-left">
                        <thead className="bg-muted/50 text-muted-foreground uppercase text-xs font-semibold">
                            <tr>
                                <th className="px-6 py-4">Officer</th>
                                <th className="px-6 py-4">Role & Rank</th>
                                <th className="px-6 py-4">Clearance</th>
                                <th className="px-6 py-4">Status</th>
                                <th className="px-6 py-4 text-right">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                            {officers.map((officer: any) => (
                                <tr key={officer.id} className={`hover:bg-muted/20 transition-colors ${!officer.is_active ? 'opacity-50' : ''}`}>
                                    <td className="px-6 py-4 font-medium text-foreground">
                                        {officer.username}
                                        <div className="text-xs text-muted-foreground font-normal mt-1">{officer.badge_number || 'No Badge'} • {officer.station_name}</div>
                                    </td>
                                    <td className="px-6 py-4">
                                        <span className="inline-flex items-center px-2 py-1 rounded-md bg-secondary text-secondary-foreground text-xs font-medium">
                                            {officer.rank || officer.role}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4">
                                        <div className="flex items-center gap-1.5">
                                            <ShieldAlert className="w-3.5 h-3.5 text-primary" />
                                            <span className="font-semibold text-primary">Level {officer.clearance_level || 0}</span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4">
                                        <span className={`inline-flex h-2 w-2 rounded-full mr-2 ${officer.is_active ? 'bg-green-500' : 'bg-destructive'}`}></span>
                                        {officer.is_active ? 'Active' : 'Inactive'}
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                        <div className="flex justify-end gap-2">
                                            <Button variant="ghost" className="h-8 w-8 p-0 hover:bg-white/10" onClick={() => handleEditOfficer(officer)} title="Edit Officer">
                                                <Settings className="w-4 h-4 text-muted-foreground hover:text-primary" />
                                            </Button>
                                            <Button variant="ghost" className="h-8 w-8 p-0 hover:bg-white/10" onClick={() => handleToggleStatus(officer.id, officer.is_active)} title={officer.is_active ? "Disable Officer" : "Enable Officer"}>
                                                {officer.is_active ? <PowerOff className="w-4 h-4 text-destructive hover:text-destructive/80" /> : <Power className="w-4 h-4 text-green-500 hover:text-green-400" />}
                                            </Button>
                                            <Button variant="ghost" className="h-8 w-8 p-0 hover:bg-white/10" onClick={() => handleDeleteOfficer(officer.id)} title="Delete Officer">
                                                <Trash2 className="w-4 h-4 text-destructive hover:text-destructive/80" />
                                            </Button>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                            {!loading && officers.length === 0 && (
                                <tr>
                                    <td colSpan={5} className="px-6 py-12 text-center text-muted-foreground">
                                        No officers found in the system. Click &quot;New Officer&quot; to add one.
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div >

            <OfficerModal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                onSave={fetchOfficers}
                officer={editingOfficer}
                token={token}
            />
        </>
    );
}
