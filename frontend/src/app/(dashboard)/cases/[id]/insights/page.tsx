import { getAccessToken } from '@/lib/auth';
import { BrainCircuit, Construction } from 'lucide-react';

export default async function InsightsPage() {
    return (
        <div className="flex flex-col h-[80vh]">
            <div className="mb-6">
                <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3 text-foreground">
                    <BrainCircuit className="w-8 h-8 text-primary" />
                    Insights (Multi-Agent System)
                </h1>
                <p className="text-muted-foreground mt-1 text-lg">AI agents analyzing case files and establishing connections.</p>
            </div>

            <div className="flex-1 flex flex-col items-center justify-center p-12 text-center bg-card/30 backdrop-blur-md border border-border/50 rounded-xl border-dashed">
                <div className="p-4 bg-muted rounded-full mb-6">
                    <Construction className="w-10 h-10 text-muted-foreground" />
                </div>
                <h2 className="text-2xl font-semibold mb-2 text-foreground">Under Construction</h2>
                <p className="text-muted-foreground max-w-md">This module will eventually house a collaborative multi-agent architecture to cross-reference data across all active investigations.</p>
            </div>
        </div>
    );
}
