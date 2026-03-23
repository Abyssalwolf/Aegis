import { getAccessToken } from '@/lib/auth';
import { getApiV1Url } from '@/lib/api';
import InsightsPanel from '@/components/cases/InsightsPanel';

export default async function InsightsPage({ params }: { params: Promise<{ id: string }> }) {
    const { id } = await params;
    const token = await getAccessToken();

    const caseRes = await fetch(`${getApiV1Url()}/cases/${id}`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store',
    });

    if (!caseRes.ok) {
        return (
            <div className="p-12 text-center text-destructive">
                Failed to load case. You may not have access.
            </div>
        );
    }

    const caseData = await caseRes.json();

    return (
        <InsightsPanel caseId={id} token={token || ''} caseTitle={caseData.title} />
    );
}
