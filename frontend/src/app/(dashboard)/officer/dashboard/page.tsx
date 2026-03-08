import { getAccessToken } from '@/lib/auth';
import { Shield, FileText, CheckCircle2 } from 'lucide-react';
import Link from 'next/link';
import { redirect } from 'next/navigation';
import { CreateCaseAction } from '@/components/officer/CreateCaseAction';

export default async function OfficerDashboard() {
    const token = await getAccessToken();

    const profileRes = await fetch('http://localhost:8000/api/v1/officer/me', {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store'
    });

    if (!profileRes.ok) {
        if (profileRes.status === 403) redirect('/admin/dashboard');
        return <div className="text-destructive">Failed to load officer profile</div>;
    }

    const casesRes = await fetch('http://localhost:8000/api/v1/officer/cases', {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store'
    });

    const profile = await profileRes.json();
    const cases = casesRes.ok ? await casesRes.json() : [];

    return (
        <>
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Officer Dashboard</h1>
                    <p className="text-muted-foreground mt-1">Welcome back, {profile.rank || 'Officer'} <span className="text-foreground font-medium">{profile.username}</span>.</p>
                </div>

                <div className="flex flex-col sm:flex-row gap-3">
                    <div className="px-4 py-2 rounded-lg bg-primary/10 border border-primary/20 flex flex-col items-center">
                        <span className="text-xs text-primary font-semibold uppercase opacity-80">Clearance Level</span>
                        <span className="text-xl font-bold text-primary flex items-center gap-2">
                            <Shield className="w-5 h-5 fill-primary/20" />
                            Level {profile.clearance_level || 0}
                        </span>
                    </div>
                    {(profile.clearance_level || 0) >= 4 && (
                        <CreateCaseAction token={token || ''} />
                    )}
                </div>
            </div>

            <div className="mt-12">
                <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
                    <FileText className="w-5 h-5 text-primary" />
                    Accessible Cases Dashboard
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                    {cases.map((c: { id: string, title: string, description: string, status: string, required_clearance_level: number }) => (
                        <Link href={`/cases/${c.id}`} key={c.id} className="group">
                            <div className="bg-card/50 backdrop-blur-sm border border-border hover:border-primary/50 transition-all duration-300 rounded-xl p-5 shadow-sm hover:shadow-primary/10 hover:-translate-y-1 h-full flex flex-col relative overflow-hidden">
                                <div className="absolute top-0 left-0 w-1 h-full bg-primary opacity-0 group-hover:opacity-100 transition-opacity" />
                                <div className="flex justify-between items-start mb-4">
                                    <span className="inline-flex items-center px-2 py-1 rounded text-[10px] font-bold bg-secondary text-secondary-foreground uppercase tracking-widest cursor-default">
                                        Clearance L{c.required_clearance_level}
                                    </span>
                                    <span className={`inline-flex items-center gap-1.5 text-xs font-medium ${c.status === 'OPEN' ? 'text-primary' : 'text-emerald-500'}`}>
                                        <CheckCircle2 className="w-3.5 h-3.5" />
                                        {c.status}
                                    </span>
                                </div>
                                <h3 className="font-bold text-lg leading-tight mb-2 group-hover:text-primary transition-colors text-foreground">{c.title}</h3>
                                <p className="text-sm text-muted-foreground line-clamp-2 flex-grow">{c.description}</p>

                                <div className="mt-5 pt-4 border-t border-border flex justify-between items-center text-xs text-muted-foreground font-medium">
                                    <span className="font-mono">ID: {c.id.substring(0, 8).toUpperCase()}</span>
                                    <span className="text-primary opacity-0 group-hover:opacity-100 transition-opacity translate-x-1 group-hover:translate-x-0">View Details &rarr;</span>
                                </div>
                            </div>
                        </Link>
                    ))}
                </div>
                {cases.length === 0 && (
                    <div className="p-16 text-center bg-card/30 border border-dashed border-border/50 rounded-2xl">
                        <Shield className="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-10" />
                        <h3 className="text-lg font-medium text-foreground">No Accessible Cases</h3>
                        <p className="text-sm text-muted-foreground mt-2 max-w-sm mx-auto">You have not been assigned any cases, or your clearance limit is too low to view restricted operations.</p>
                    </div>
                )}
            </div>
        </>
    )
}
