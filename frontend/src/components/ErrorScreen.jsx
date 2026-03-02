import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

export default function ErrorScreen({ message, onRetry }) {
    return (
        <div className="error-container flex-col items-center justify-center animate-fade-in" style={{ minHeight: '60vh', gap: 'var(--space-3)' }}>
            <div style={{ color: 'var(--color-error)' }}>
                <AlertCircle size={64} />
            </div>
            <div style={{ textAlign: 'center', maxWidth: '400px' }}>
                <h2 style={{ marginBottom: 'var(--space-1)' }}>Extract Failed</h2>
                <p style={{ color: 'var(--color-muted)' }}>
                    {message || 'An unexpected error occurred while processing your document.'}
                </p>
            </div>
            <button className="btn-primary" onClick={onRetry} style={{ marginTop: 'var(--space-2)' }}>
                <RefreshCw size={18} />
                Try Again
            </button>
        </div>
    );
}
