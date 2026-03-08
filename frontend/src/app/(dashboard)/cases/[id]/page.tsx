import { getAccessToken } from '@/lib/auth';
import { Shield, FileText, CheckCircle2, UploadCloud, Activity, Plus, ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import Link from 'next/link';
import { CaseSettingsAction } from '@/components/officer/CaseSettingsAction';
import { ManageOfficersAction } from '@/components/officer/ManageOfficersAction';

export default async function CaseDetailPage({ params }: { params: { id: string } }) {
    const token = await getAccessToken();

    // Fetch case details
    const caseRes = await fetch(`http://localhost:8000/api/v1/cases/${params.id}`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store'
    });

    if (!caseRes.ok) {
        return <div className="p-12 text-center text-destructive">Failed to load case details. You may not have clearance.</div>;
    }

    const caseData = await caseRes.json();

    // Fetch current user details
    const meRes = await fetch('http://localhost:8000/api/v1/officer/me', {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store'
    });
    const meData = meRes.ok ? await meRes.json() : null;

    // Fetch creator details
    const creatorRes = await fetch(`http://localhost:8000/api/v1/officer/${caseData.created_by}`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store'
    });
    const creatorData = creatorRes.ok ? await creatorRes.json() : null;

    const isOwner = meData && meData.id === caseData.created_by;

    // Fetch documents
    const docsRes = await fetch(`http://localhost:8000/api/v1/cases/${params.id}/documents`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store'
    });
    const documents = docsRes.ok ? await docsRes.json() : [];

    // Fetch assigned officers
    const assignedRes = await fetch(`http://localhost:8000/api/v1/cases/${params.id}/officers`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store'
    });
    const assignedOfficers = assignedRes.ok ? await assignedRes.json() : [];

    // Logic: can manage if owner OR (assigned AND higher clearance than owner)
    const isAssigned = meData && assignedOfficers.some((officer: any) => officer.id === meData.id);
    const creatorClearance = creatorData ? (creatorData.clearance_level || 0) : 0;
    const myClearance = meData ? (meData.clearance_level || 0) : 0;
    const canManage = isOwner || (isAssigned && myClearance > creatorClearance);

    return (
        <div className="space-y-6">
            <Link
                href="/officer/dashboard"
                className="inline-flex items-center text-sm font-medium text-muted-foreground hover:text-foreground transition-colors mb-2 gap-2"
            >
                <ArrowLeft className="w-4 h-4" />
                Back to Dashboard
            </Link>

            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-border pb-6">
                <div>
                    <div className="flex items-center gap-3 mb-2">
                        <span className="inline-flex items-center px-2 py-1 rounded text-xs font-bold bg-primary/20 text-primary uppercase tracking-widest cursor-default border border-primary/20">
                            Case ID: {caseData.id.substring(0, 8)}
                        </span>
                        <span className={`inline-flex items-center gap-1.5 text-xs font-medium ${caseData.status === 'OPEN' ? 'text-primary' : 'text-emerald-500'}`}>
                            <CheckCircle2 className="w-4 h-4" />
                            {caseData.status}
                        </span>
                    </div>
                    <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-foreground">{caseData.title}</h1>
                </div>
                {isOwner && (
                    <div className="flex items-center gap-3">
                        <CaseSettingsAction token={token || ''} caseId={caseData.id} caseTitle={caseData.title} />
                    </div>
                )}
            </div>

            <div className="pt-8 grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Main Column */}
                <div className="col-span-1 lg:col-span-2 space-y-8">
                    <section className="bg-card/50 backdrop-blur-md border border-border rounded-xl p-6 shadow-sm">
                        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                            <FileText className="w-5 h-5 text-primary" />
                            Overview
                        </h2>
                        <div className="prose prose-invert max-w-none text-muted-foreground">
                            <p className="whitespace-pre-wrap leading-relaxed">{caseData.description}</p>
                        </div>
                        <div className="mt-6 flex gap-4 text-sm text-muted-foreground border-t border-border pt-4">
                            <div className="flex flex-col">
                                <span className="font-medium text-foreground">Required Clearance</span>
                                <span>Level {caseData.required_clearance_level}</span>
                            </div>
                            <div className="flex flex-col">
                                <span className="font-medium text-foreground">Created At</span>
                                <span>{new Date(caseData.created_at).toLocaleDateString()}</span>
                            </div>
                            {creatorData && (
                                <div className="flex flex-col ml-8 border-l border-border pl-4">
                                    <span className="font-medium text-foreground">Lead Investigator</span>
                                    <span>{creatorData.username} (Lvl {creatorData.clearance_level})</span>
                                </div>
                            )}
                        </div>
                    </section>

                    <section className="bg-card/50 backdrop-blur-md border border-border rounded-xl p-6 shadow-sm">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-xl font-semibold flex items-center gap-2">
                                <Activity className="w-5 h-5 text-primary" />
                                Activity Log
                            </h2>
                        </div>

                        <div className="space-y-4 relative before:absolute before:inset-y-0 before:left-2.5 before:w-0.5 before:bg-border">
                            <div className="relative pl-8 bg-transparent">
                                <div className="absolute left-0 top-1 w-5 h-5 rounded-full bg-primary/20 border-2 border-primary flex items-center justify-center">
                                    <div className="w-1.5 h-1.5 rounded-full bg-primary" />
                                </div>
                                <p className="text-sm font-medium">Case Created</p>
                                <p className="text-xs text-muted-foreground mt-0.5">{new Date(caseData.created_at).toLocaleString()}</p>
                            </div>
                            {documents.map((doc: any, i: number) => (
                                <div key={doc.id} className="relative pl-8 bg-transparent">
                                    <div className="absolute left-0 top-1 w-5 h-5 rounded-full bg-muted border-2 border-border flex items-center justify-center" />
                                    <p className="text-sm font-medium">Document Uploaded: {doc.document_type}</p>
                                    <p className="text-xs text-muted-foreground mt-0.5">{new Date(doc.created_at).toLocaleString()}</p>
                                </div>
                            ))}
                        </div>
                    </section>
                </div>

                {/* Sidebar Column */}
                <div className="col-span-1 space-y-6">
                    <section className="bg-card/50 backdrop-blur-md border border-border rounded-xl p-6 shadow-sm">
                        <div className="flex items-center justify-between mb-4">
                            <h2 className="font-semibold flex items-center gap-2 text-foreground">
                                <UploadCloud className="w-4 h-4 text-muted-foreground" />
                                Documents
                            </h2>
                            <Button variant="ghost" className="h-8 w-8 p-0" title="Upload Document">
                                <Plus className="w-4 h-4" />
                            </Button>
                        </div>
                        <div className="space-y-3">
                            {documents.map((doc: any) => (
                                <div key={doc.id} className="flex items-center justify-between p-3 rounded-lg border border-border bg-background hover:bg-muted/50 transition-colors cursor-pointer group">
                                    <div className="flex flex-col">
                                        <span className="text-sm font-medium group-hover:text-primary transition-colors">{doc.document_type}</span>
                                        <span className="text-xs text-muted-foreground mt-1 truncate max-w-[150px]">{doc.file_path}</span>
                                    </div>
                                </div>
                            ))}
                            {documents.length === 0 && (
                                <div className="text-center p-4 border border-dashed border-border rounded-lg text-sm text-muted-foreground">
                                    No documents attached.
                                </div>
                            )}
                        </div>
                    </section>

                    <section className="bg-card/50 backdrop-blur-md border border-border rounded-xl p-6 shadow-sm">
                        <div className="flex items-center justify-between mb-4">
                            <h2 className="font-semibold flex items-center gap-2 text-foreground">
                                <Shield className="w-4 h-4 text-muted-foreground" />
                                Assigned Personnel
                            </h2>
                            {canManage && (
                                <ManageOfficersAction token={token || ''} caseId={caseData.id} />
                            )}
                        </div>
                        <div className="space-y-3">
                            {assignedOfficers.map((doc: any) => (
                                <div key={doc.id} className="flex items-center justify-between p-3 rounded-lg border border-border bg-background hover:bg-muted/50 transition-colors">
                                    <div className="flex flex-col">
                                        <span className="text-sm font-medium">{doc.username}</span>
                                        <span className="text-xs text-muted-foreground mt-0.5">{doc.rank || 'Officer'} · Level {doc.clearance_level}</span>
                                    </div>
                                </div>
                            ))}
                            {assignedOfficers.length === 0 && (
                                <div className="text-center p-4 border border-dashed border-border rounded-lg text-sm text-muted-foreground">
                                    No personnel assigned.
                                </div>
                            )}
                        </div>
                    </section>
                </div>
            </div>
        </div>
    )
}
