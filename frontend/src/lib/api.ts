/**
 * Public API base for browser + server.
 * Set NEXT_PUBLIC_API_URL in .env.local (no trailing slash), e.g. http://192.168.1.5:8000
 */
export function getPublicApiBaseUrl(): string {
    const raw = typeof process !== 'undefined' ? process.env.NEXT_PUBLIC_API_URL : undefined;
    if (raw && raw.trim()) {
        return raw.trim().replace(/\/$/, '');
    }
    return 'http://localhost:8000';
}

export function getApiV1Url(): string {
    return `${getPublicApiBaseUrl()}/api/v1`;
}
