'use server';

import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';

export async function loginAction(formData: FormData) {
    const username = formData.get('username');
    const password = formData.get('password');

    if (!username || !password) {
        return { error: 'Username and password are required' };
    }

    try {
        const res = await fetch('http://localhost:8000/api/v1/auth/login', {
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

        // Set cookies
        cookies().set('access_token', tokens.access_token, {
            httpOnly: true,
            secure: process.env.NODE_ENV === 'production',
            maxAge: 60 * 60 * 24 * 7, // 7 days
            path: '/',
        });

        cookies().set('refresh_token', tokens.refresh_token, {
            httpOnly: true,
            secure: process.env.NODE_ENV === 'production',
            maxAge: 60 * 60 * 24 * 30, // 30 days
            path: '/',
        });

        // We need to decode the token to know if admin or officer to route properly
        // Oh wait, JWT token payload only has "sub" right now, no role.
        // I will fetch `/officer/me` to get role. Wait, `/officer/me` is protected by `get_current_active_officer`. Let's fetch the user role directly if we have an endpoint, or just try to fetch both.
        // Wait! Let's update `app/api/endpoints/auth.py` to include role in token, it's easier and saves a network call. Wait, I can't do that now without breaking flow. Let me just route based on /admin/officers or similar.
        // A better way is to make an API call to a `/me` endpoint but we don't have a shared one. We only have `/officer/me`. If it fails with 403, we must be an admin.

        const meRes = await fetch('http://localhost:8000/api/v1/officer/me', {
            headers: { Authorization: `Bearer ${tokens.access_token}` }
        });

        if (meRes.ok) {
            redirect('/officer/dashboard');
        } else {
            redirect('/admin/dashboard');
        }

    } catch (error) {
        if ((error as Error).message === 'NEXT_REDIRECT') {
            throw error;
        }
        return { error: 'Service is temporarily unavailable' };
    }
}

export async function logoutAction() {
    cookies().delete('access_token');
    cookies().delete('refresh_token');
    redirect('/');
}

export async function getAccessToken() {
    const token = cookies().get('access_token');
    return token?.value;
}
