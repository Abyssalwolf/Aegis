'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import {
    UserPlus, Settings, ShieldAlert, Trash2, PowerOff, Power,
    ChevronDown, ChevronUp, FolderOpen, UserCheck, X, Loader2
} from 'lucide-react';
import { OfficerModal } from '@/components/admin/OfficerModal';
import { getAccessToken } from '@/lib/auth';
import { getApiV1Url } from '@/lib/api';

const API = getApiV1Url();

const STATUS_COLORS: Record<string, string> = {
    OPEN: 'text-blue-400 bg-blue-400/10',
    UNDER_INVESTIGATION: 'text-yellow-400 bg-yellow-400/10',
    CLOSED: 'text-muted-foreground bg-muted/30',
};

interface Officer {
    id: string;
    username: string;
    rank: string | null;
    role: string;
    clearance_level: number | null;
    badge_number: string | null;
    station_name: string | null;
    is_active: boolean;
}

interface Case {
    id: string;
    title: string;
    status: string;
    required_clearance_level: number;
    created_by: string;
}

interface AssignPanelState {
    caseId: string;
    caseTitle: string;
    requiredClearance: number;
}

export default function AdminDashboard() {
    const [officers, setOfficers] = useState<Officer[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [token, setToken] = useState<string>('');

    // Officer modal
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [editingOfficer, setEditingOfficer] = useState<Officer | null>(null);

    // Expanded cases per officer: officerId → Case[] | null (null = loading)
    const [expandedOfficer, setExpandedOfficer] = useState<string | null>(null);
    const [officerCases, setOfficerCases] = useState<Record<string, Case[]>>({});
    const [casesLoading, setCasesLoading] = useState(false);

    // Assign officer panel
    const [assignPanel, setAssignPanel] = useState<AssignPanelState | null>(null);
    const [eligibleOfficers, setEligibleOfficers] = useState<Officer[]>([]);
    const [assignLoading, setAssignLoading] = useState(false);

    const fetchOfficers = useCallback(async () => {
        setLoading(true);
        try {
            const currentToken = await getAccessToken();
            if (!currentToken) throw new Error('No access token found');
            setToken(currentToken);

            const res = await fetch(`${API}/admin/officers?limit=500`, {
                headers: { Authorization: `Bearer ${currentToken}` },
                cache: 'no-store',
            });

            if (!res.ok) {
                setError(res.status === 403
                    ? 'Access Restricted: Only administrators can view this dashboard.'
                    : 'Failed to load officers. Is backend running?');
            } else {
                const data = await res.json();
                setOfficers(Array.isArray(data) ? data : (data.items ?? []));
                setError(null);
            }
        } catch (err: any) {
            setError(err.message || 'An unexpected error occurred');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { fetchOfficers(); }, [fetchOfficers]);

    // ── Officer actions ───────────────────────────────────────────────────────

    const handleToggleStatus = async (officerId: string, currentStatus: boolean) => {
        try {
            const res = await fetch(`${API}/admin/officers/${officerId}/status?is_active=${!currentStatus}`, {
                method: 'PATCH',
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) { fetchOfficers(); }
            else {
                const data = await res.json().catch(() => ({}));
                setError(data.detail || `Status update failed (${res.status})`);
            }
        } catch (e: any) { setError(e.message || 'Network error'); }
    };

    const handleDeleteOfficer = async (officerId: string) => {
        if (!confirm('Are you sure you want to permanently delete this officer?')) return;
        try {
            const res = await fetch(`${API}/admin/officers/${officerId}`, {
                method: 'DELETE',
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) {
                if (expandedOfficer === officerId) setExpandedOfficer(null);
                fetchOfficers();
            } else {
                const data = await res.json().catch(() => ({}));
                setError(data.detail || `Delete failed (${res.status})`);
            }
        } catch (e: any) { setError(e.message || 'Network error'); }
    };

    // ── Case panel ────────────────────────────────────────────────────────────

    const toggleCases = async (officerId: string) => {
        if (expandedOfficer === officerId) {
            setExpandedOfficer(null);
            return;
        }
        setExpandedOfficer(officerId);
        if (officerCases[officerId]) return; // already fetched

        setCasesLoading(true);
        try {
            const res = await fetch(`${API}/admin/cases?officer_id=${officerId}`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) {
                const data: Case[] = await res.json();
                setOfficerCases(prev => ({ ...prev, [officerId]: data }));
            } else {
                setOfficerCases(prev => ({ ...prev, [officerId]: [] }));
            }
        } catch {
            setOfficerCases(prev => ({ ...prev, [officerId]: [] }));
        } finally {
            setCasesLoading(false);
        }
    };

    const handleDeleteCase = async (officerId: string, caseId: string, caseTitle: string) => {
        if (!confirm(`Delete case "${caseTitle}"? This cannot be undone.`)) return;
        try {
            const res = await fetch(`${API}/admin/cases/${caseId}`, {
                method: 'DELETE',
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) {
                setOfficerCases(prev => ({
                    ...prev,
                    [officerId]: prev[officerId].filter(c => c.id !== caseId),
                }));
            } else {
                const data = await res.json().catch(() => ({}));
                setError(data.detail || `Case delete failed (${res.status})`);
            }
        } catch (e: any) { setError(e.message || 'Network error'); }
    };

    // ── Assign officer panel ──────────────────────────────────────────────────

    const openAssignPanel = async (c: Case) => {
        setAssignPanel({ caseId: c.id, caseTitle: c.title, requiredClearance: c.required_clearance_level });
        setAssignLoading(true);
        try {
            const res = await fetch(`${API}/admin/officers`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (res.ok) {
                const all: Officer[] = await res.json();
                setEligibleOfficers(all.filter(o => o.is_active && (o.clearance_level || 0) >= c.required_clearance_level));
            }
        } finally {
            setAssignLoading(false);
        }
    };

    const handleAssignOfficer = async (officerId: string) => {
        if (!assignPanel) return;
        try {
            const res = await fetch(`${API}/admin/cases/${assignPanel.caseId}/officers`, {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ officer_id: officerId }),
            });
            if (res.ok) {
                setAssignPanel(null);
            } else {
                const data = await res.json().catch(() => ({}));
                setError(data.detail || `Assign failed (${res.status})`);
            }
        } catch (e: any) { setError(e.message || 'Network error'); }
    };

    // ── Render ────────────────────────────────────────────────────────────────

    if (error?.includes('Restricted')) {
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
                <Button className="gap-2" onClick={() => { setEditingOfficer(null); setIsModalOpen(true); }}>
                    <UserPlus className="w-4 h-4" />
                    New Officer
                </Button>
            </div>

            {error && !error.includes('Restricted') && (
                <div className="mt-4 p-4 text-destructive bg-destructive/10 rounded-lg flex items-center justify-between">
                    <span>{error}</span>
                    <button onClick={() => setError(null)}><X className="w-4 h-4" /></button>
                </div>
            )}

            <div className="bg-card border border-border rounded-xl shadow-sm overflow-hidden mt-8">
                <div className="overflow-x-auto relative min-h-[400px]">
                    {loading && (
                        <div className="absolute inset-0 z-10 bg-background/50 flex items-center justify-center backdrop-blur-sm">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
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
                            {officers.map((officer) => (
                                <React.Fragment key={officer.id}>
                                    {/* Officer row */}
                                    <tr
                                        className={`hover:bg-muted/20 transition-colors ${!officer.is_active ? 'opacity-50' : ''}`}
                                    >
                                        <td className="px-6 py-4 font-medium text-foreground">
                                            {officer.username}
                                            <div className="text-xs text-muted-foreground font-normal mt-1">
                                                {officer.badge_number || 'No Badge'} • {officer.station_name}
                                            </div>
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
                                            <span className={`inline-flex h-2 w-2 rounded-full mr-2 ${officer.is_active ? 'bg-green-500' : 'bg-destructive'}`} />
                                            {officer.is_active ? 'Active' : 'Inactive'}
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            <div className="flex justify-end gap-2">
                                                <Button
                                                    variant="ghost" size="sm"
                                                    className="h-8 px-2 gap-1 text-xs text-muted-foreground hover:text-primary hover:bg-white/10"
                                                    onClick={() => toggleCases(officer.id)}
                                                    title="View Cases"
                                                >
                                                    <FolderOpen className="w-3.5 h-3.5" />
                                                    Cases
                                                    {expandedOfficer === officer.id
                                                        ? <ChevronUp className="w-3 h-3" />
                                                        : <ChevronDown className="w-3 h-3" />}
                                                </Button>
                                                <Button variant="ghost" className="h-8 w-8 p-0 hover:bg-white/10" onClick={() => { setEditingOfficer(officer); setIsModalOpen(true); }} title="Edit Officer">
                                                    <Settings className="w-4 h-4 text-muted-foreground hover:text-primary" />
                                                </Button>
                                                <Button variant="ghost" className="h-8 w-8 p-0 hover:bg-white/10" onClick={() => handleToggleStatus(officer.id, officer.is_active)} title={officer.is_active ? 'Disable Officer' : 'Enable Officer'}>
                                                    {officer.is_active
                                                        ? <PowerOff className="w-4 h-4 text-destructive" />
                                                        : <Power className="w-4 h-4 text-green-500" />}
                                                </Button>
                                                <Button variant="ghost" className="h-8 w-8 p-0 hover:bg-white/10" onClick={() => handleDeleteOfficer(officer.id)} title="Delete Officer">
                                                    <Trash2 className="w-4 h-4 text-destructive" />
                                                </Button>
                                            </div>
                                        </td>
                                    </tr>

                                    {/* Expandable cases sub-panel */}
                                    {expandedOfficer === officer.id && (
                                        <tr className="bg-muted/10">
                                            <td colSpan={5} className="px-6 py-4">
                                                <div className="space-y-2">
                                                    <p className="text-xs font-semibold uppercase text-muted-foreground tracking-wider mb-3">
                                                        Cases created by {officer.username}
                                                    </p>

                                                    {casesLoading && !officerCases[officer.id] ? (
                                                        <div className="flex items-center gap-2 text-muted-foreground text-sm py-2">
                                                            <Loader2 className="w-4 h-4 animate-spin" /> Loading cases…
                                                        </div>
                                                    ) : officerCases[officer.id]?.length === 0 ? (
                                                        <p className="text-sm text-muted-foreground py-2">No cases created by this officer.</p>
                                                    ) : (
                                                        <div className="rounded-lg border border-border overflow-hidden">
                                                            <table className="w-full text-sm">
                                                                <thead className="bg-muted/40 text-muted-foreground text-xs uppercase">
                                                                    <tr>
                                                                        <th className="px-4 py-2 text-left">Case Title</th>
                                                                        <th className="px-4 py-2 text-left">Status</th>
                                                                        <th className="px-4 py-2 text-left">Required Clearance</th>
                                                                        <th className="px-4 py-2 text-right">Actions</th>
                                                                    </tr>
                                                                </thead>
                                                                <tbody className="divide-y divide-border/50">
                                                                    {(officerCases[officer.id] || []).map(c => (
                                                                        <tr key={c.id} className="hover:bg-muted/20 transition-colors">
                                                                            <td className="px-4 py-2.5 font-medium">{c.title}</td>
                                                                            <td className="px-4 py-2.5">
                                                                                <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${STATUS_COLORS[c.status] ?? 'text-muted-foreground bg-muted/30'}`}>
                                                                                    {c.status.replace(/_/g, ' ')}
                                                                                </span>
                                                                            </td>
                                                                            <td className="px-4 py-2.5">
                                                                                <div className="flex items-center gap-1 text-primary">
                                                                                    <ShieldAlert className="w-3 h-3" />
                                                                                    <span className="font-semibold text-xs">Level {c.required_clearance_level}</span>
                                                                                </div>
                                                                            </td>
                                                                            <td className="px-4 py-2.5 text-right">
                                                                                <div className="flex justify-end gap-2">
                                                                                    <Button
                                                                                        variant="ghost" size="sm"
                                                                                        className="h-7 px-2 gap-1 text-xs hover:bg-white/10 text-muted-foreground hover:text-primary"
                                                                                        onClick={() => openAssignPanel(c)}
                                                                                        title="Assign officer to this case"
                                                                                    >
                                                                                        <UserCheck className="w-3.5 h-3.5" />
                                                                                        Assign Officer
                                                                                    </Button>
                                                                                    <Button
                                                                                        variant="ghost" size="sm"
                                                                                        className="h-7 w-7 p-0 hover:bg-destructive/10"
                                                                                        onClick={() => handleDeleteCase(officer.id, c.id, c.title)}
                                                                                        title="Delete case"
                                                                                    >
                                                                                        <Trash2 className="w-3.5 h-3.5 text-destructive" />
                                                                                    </Button>
                                                                                </div>
                                                                            </td>
                                                                        </tr>
                                                                    ))}
                                                                </tbody>
                                                            </table>
                                                        </div>
                                                    )}
                                                </div>
                                            </td>
                                        </tr>
                                    )}
                                </React.Fragment>
                            ))}
                            {!loading && officers.length === 0 && (
                                <tr>
                                    <td colSpan={5} className="px-6 py-12 text-center text-muted-foreground">
                                        No officers found. Click &quot;New Officer&quot; to add one.
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Assign Officer Modal */}
            {assignPanel && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
                    <div className="bg-card border border-border rounded-xl shadow-2xl w-full max-w-md p-6 space-y-4">
                        <div className="flex items-center justify-between">
                            <div>
                                <h3 className="font-semibold text-lg">Assign Officer</h3>
                                <p className="text-xs text-muted-foreground mt-0.5">
                                    Case: <span className="text-foreground font-medium">{assignPanel.caseTitle}</span>
                                </p>
                                <p className="text-xs text-muted-foreground">
                                    Minimum clearance required: <span className="text-primary font-semibold">Level {assignPanel.requiredClearance}</span>
                                </p>
                            </div>
                            <button onClick={() => setAssignPanel(null)} className="text-muted-foreground hover:text-foreground">
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {assignLoading ? (
                            <div className="flex items-center gap-2 text-muted-foreground py-4 justify-center">
                                <Loader2 className="w-4 h-4 animate-spin" /> Loading eligible officers…
                            </div>
                        ) : eligibleOfficers.length === 0 ? (
                            <p className="text-sm text-muted-foreground py-4 text-center">
                                No active officers meet the clearance requirement for this case.
                            </p>
                        ) : (
                            <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
                                {eligibleOfficers.map(o => (
                                    <button
                                        key={o.id}
                                        onClick={() => handleAssignOfficer(o.id)}
                                        className="w-full flex items-center justify-between px-3 py-2.5 rounded-lg hover:bg-muted/50 border border-transparent hover:border-border transition-colors text-left"
                                    >
                                        <div>
                                            <p className="font-medium text-sm">{o.username}</p>
                                            <p className="text-xs text-muted-foreground">{o.rank || 'Officer'} • {o.station_name || '—'}</p>
                                        </div>
                                        <div className="flex items-center gap-1 text-primary text-xs font-semibold">
                                            <ShieldAlert className="w-3 h-3" />
                                            Level {o.clearance_level}
                                        </div>
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            )}

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
