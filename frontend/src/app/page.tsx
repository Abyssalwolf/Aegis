'use client';

import { Shield } from 'lucide-react';
import { loginAction } from '@/lib/auth';
import { useState } from 'react';

export default function LoginPage() {
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(formData: FormData) {
    setLoading(true);
    setError(null);
    try {
      const result = await loginAction(formData);
      if (result?.error) {
        setError(result.error);
      }
    } catch (e: any) {
      if (e.message !== 'NEXT_REDIRECT') {
        setError('An unexpected error occurred.');
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen w-full flex bg-background text-foreground selection:bg-primary/30">
      {/* Decorative Left Side */}
      <div className="hidden lg:flex w-1/2 relative bg-secondary overflow-hidden items-center justify-center p-12 lg:p-24 flex-col">
        {/* Abstract Background Shapes */}
        <div className="absolute inset-0 z-0">
          <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-primary/20 blur-[120px] rounded-full animate-pulse-glow" />
          <div className="absolute bottom-[-10%] right-[-10%] w-[60%] h-[60%] bg-blue-900/20 blur-[120px] rounded-full animate-pulse-glow" style={{ animationDelay: '1s' }} />
        </div>

        <div className="relative z-10 flex flex-col items-center max-w-lg text-center gap-6 animate-fade-in-up">
          <div className="p-4 rounded-2xl bg-background/10 backdrop-blur-md border border-white/10 shadow-2xl">
            <Shield className="w-16 h-16 text-primary" />
          </div>
          <h1 className="text-4xl lg:text-5xl font-bold tracking-tight text-white leading-tight">
            AEGIS
            <span className="block text-2xl font-medium text-white/70 mt-4">AI Police Assistance System</span>
          </h1>
          <p className="text-lg text-white/50 leading-relaxed max-w-md">
            Secure, case-centric intelligence management for modern police departments.
          </p>
        </div>
      </div>

      {/* Right side login form */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-8">
        <div className="w-full max-w-sm space-y-8 animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
          <div className="flex flex-col space-y-2 text-center">
            <div className="lg:hidden mx-auto p-3 rounded-2xl bg-secondary mb-4">
              <Shield className="w-10 h-10 text-primary" />
            </div>
            <h2 className="text-3xl font-semibold tracking-tight">Access Portal</h2>
            <p className="text-sm text-muted-foreground">
              Enter your credentials to securely login.
            </p>
          </div>

          <form action={handleSubmit} className="space-y-6">
            {error && (
              <div className="p-3 text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-md text-center">
                {error}
              </div>
            )}
            <div className="space-y-4">
              <div className="space-y-2 relative">
                <input
                  name="username"
                  type="text"
                  required
                  id="username"
                  className="peer flex h-11 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-transparent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 transition-colors hover:border-muted-foreground/30 focus:placeholder:text-muted-foreground"
                  placeholder="Username"
                />
                <label
                  htmlFor="username"
                  className="absolute left-3 -top-2.5 bg-background px-1 text-xs font-medium text-muted-foreground transition-all peer-placeholder-shown:top-2.5 peer-placeholder-shown:text-sm peer-focus:-top-2.5 peer-focus:text-xs peer-focus:text-primary"
                >
                  Username
                </label>
              </div>
              <div className="space-y-2 relative">
                <input
                  name="password"
                  type="password"
                  required
                  id="password"
                  className="peer flex h-11 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-transparent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 transition-colors hover:border-muted-foreground/30 focus:placeholder:text-muted-foreground"
                  placeholder="Password"
                />
                <label
                  htmlFor="password"
                  className="absolute left-3 -top-2.5 bg-background px-1 text-xs font-medium text-muted-foreground transition-all peer-placeholder-shown:top-2.5 peer-placeholder-shown:text-sm peer-focus:-top-2.5 peer-focus:text-xs peer-focus:text-primary"
                >
                  Password
                </label>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full inline-flex h-11 items-center justify-center rounded-lg bg-primary px-8 text-sm font-medium text-primary-foreground transition-all hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 shadow-lg hover:shadow-primary/20 hover:-translate-y-0.5"
            >
              {loading ? 'Authenticating...' : 'Sign In'}
            </button>
          </form>

          <div className="text-center">
            <p className="text-xs text-muted-foreground">Unauthorized access is strictly prohibited and monitored.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
