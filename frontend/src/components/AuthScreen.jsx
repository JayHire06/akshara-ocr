import React, { useState } from 'react';
import { UserPlus, LogIn, ArrowRight } from 'lucide-react';
import { api, setAuthToken } from '../services/api';

export default function AuthScreen({ onLoginSuccess }) {
    const [isLogin, setIsLogin] = useState(true);
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [errorMsg, setErrorMsg] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setErrorMsg('');
        setLoading(true);

        try {
            if (isLogin) {
                const res = await api.login(username, password);
                setAuthToken(res.access_token);
                onLoginSuccess();
            } else {
                await api.register(username, password);
                const res = await api.login(username, password);
                setAuthToken(res.access_token);
                onLoginSuccess();
            }
        } catch (err) {
            setErrorMsg(err.message || 'Authentication failed. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="auth-container flex-col items-center justify-center animate-fade-in" style={{ minHeight: '80vh', padding: 'var(--space-4)' }}>
            <div className="card" style={{ width: '100%', maxWidth: '400px', padding: 'var(--space-5) var(--space-4)' }}>
                <h2 style={{ textAlign: 'center', marginBottom: 'var(--space-1)' }}>
                    {isLogin ? 'Welcome Back' : 'Create Account'}
                </h2>
                <p style={{ textAlign: 'center', color: 'var(--color-muted)', marginBottom: 'var(--space-4)' }}>
                    {isLogin ? 'Sign in to digitize your documents.' : 'Sign up to start extracting text.'}
                </p>

                {errorMsg && (
                    <div style={{ color: 'var(--color-error)', marginBottom: 'var(--space-3)', padding: 'var(--space-2)', backgroundColor: 'rgba(255, 74, 74, 0.1)', borderRadius: 'var(--radius-chip)', textAlign: 'center', fontSize: '0.9rem' }}>
                        {errorMsg}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="flex-col gap-3">
                    <div className="input-group flex-col gap-1">
                        <label style={{ fontSize: '0.9rem', fontWeight: '500' }}>Username</label>
                        <input
                            type="text"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            required
                            placeholder="Enter your username"
                            style={{
                                width: '100%',
                                padding: '12px 16px',
                                borderRadius: 'var(--radius-button)',
                                border: '1px solid var(--color-border)',
                                backgroundColor: 'var(--color-surface)',
                                outline: 'none',
                                transition: 'border-color 0.2s',
                                color: 'var(--color-text)'
                            }}
                            onFocus={(e) => e.target.style.borderColor = 'var(--color-accent)'}
                            onBlur={(e) => e.target.style.borderColor = 'var(--color-border)'}
                        />
                    </div>

                    <div className="input-group flex-col gap-1">
                        <label style={{ fontSize: '0.9rem', fontWeight: '500' }}>Password</label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            placeholder="Enter your password"
                            style={{
                                width: '100%',
                                padding: '12px 16px',
                                borderRadius: 'var(--radius-button)',
                                border: '1px solid var(--color-border)',
                                backgroundColor: 'var(--color-surface)',
                                outline: 'none',
                                transition: 'border-color 0.2s',
                                color: 'var(--color-text)'
                            }}
                            onFocus={(e) => e.target.style.borderColor = 'var(--color-accent)'}
                            onBlur={(e) => e.target.style.borderColor = 'var(--color-border)'}
                        />
                    </div>

                    <button
                        type="submit"
                        className="btn-primary"
                        disabled={loading || !username || !password}
                        style={{ width: '100%', marginTop: 'var(--space-2)', padding: '12px' }}
                    >
                        {loading ? 'Processing...' : (
                            <>
                                {isLogin ? <LogIn size={18} /> : <UserPlus size={18} />}
                                {isLogin ? 'Sign In' : 'Sign Up'}
                            </>
                        )}
                    </button>
                </form>

                <div style={{ textAlign: 'center', marginTop: 'var(--space-4)', fontSize: '0.9rem', color: 'var(--color-muted)' }}>
                    {isLogin ? "Don't have an account? " : "Already have an account? "}
                    <button
                        type="button"
                        onClick={() => { setIsLogin(!isLogin); setErrorMsg(''); }}
                        style={{ color: 'var(--color-accent)', fontWeight: '600', textDecoration: 'underline' }}
                    >
                        {isLogin ? 'Sign up' : 'Sign in'}
                        <ArrowRight size={14} style={{ display: 'inline-block', verticalAlign: 'middle', marginLeft: '4px' }} />
                    </button>
                </div>
            </div>
        </div>
    );
}
