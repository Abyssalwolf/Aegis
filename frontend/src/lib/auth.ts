'use server';

import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';

import { getApiV1Url } from '@/lib/api';

export async function loginAction(formData: FormData) {
    const username = formData.get('username');
    const password = formData.get('password');

    if (!username || !password) {
        return { error: 'Username and password are required' };
    }

    try {
        const res = await fetch(`${getApiV1Url()}/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                username: username.toString(),
                password: password.toString(),
            }),
        });

        if (!res.ok) {
            const data = await res.json();
            return { error: data.detail || 'Login failed' };
        }

        const tokens = await res.json();

        const cookieStore = await cookies();

        cookieStore.set('access_token', tokens.access_token, {
            httpOnly: true,
            secure: process.env.NODE_ENV === 'production',
            maxAge: 60 * 60 * 24 * 7,
            path: '/',
        });

        cookieStore.set('refresh_token', tokens.refresh_token, {
            httpOnly: true,
            secure: process.env.NODE_ENV === 'production',
            maxAge: 60 * 60 * 24 * 30,
            path: '/',
        });

        const meRes = await fetch(`${getApiV1Url()}/officer/me`, {
            headers: { Authorization: `Bearer ${tokens.access_token}` }
        });

        if (meRes.ok) {
            redirect('/officer/dashboard');
        } else {
            redirect('/admin/dashboard');
        }

    } catch (error) {
        if ((error as any).digest?.startsWith('NEXT_REDIRECT')) {
            throw error;
        }
        return { error: 'Service is temporarily unavailable' };
    }
}

export async function logoutAction() {
    const cookieStore = await cookies();
    cookieStore.delete('access_token');
    cookieStore.delete('refresh_token');
    redirect('/');
}

export async function getAccessToken() {
    const cookieStore = await cookies();
    const token = cookieStore.get('access_token');
    return token?.value;
}
