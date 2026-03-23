import { redirect } from 'next/navigation';

/** Alias: /cases/[id]/investigation → RAG chat at /cases/[id]/chat */
export default async function CaseInvestigationAliasPage({
    params,
}: {
    params: Promise<{ id: string }>;
}) {
    const { id } = await params;
    redirect(`/cases/${id}/chat`);
}
