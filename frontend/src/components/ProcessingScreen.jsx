import React from 'react';
import { Loader2, XCircle } from 'lucide-react';

export default function ProcessingScreen({ onCancel }) {
    return (
        <div className="processing-container flex-col items-center justify-center animate-fade-in" style={{ minHeight: '60vh', gap: 'var(--space-4)' }}>
            <div className="processing-spinner" style={{ color: 'var(--color-accent)' }}>
                <Loader2 size={64} className="animate-spin" />
            </div>
            <div className="text-center" style={{ textAlign: 'center' }}>
                <h2 style={{ marginBottom: 'var(--space-1)' }}>Extracting Text...</h2>
                <p style={{ color: 'var(--color-muted)' }}>This usually takes a few seconds.</p>
            </div>

            <div className="progress-bar-container" style={{
                width: '100%',
                maxWidth: '400px',
                height: '8px',
                backgroundColor: 'var(--color-surface)',
                borderRadius: '4px',
                overflow: 'hidden',
                marginTop: 'var(--space-2)'
            }}>
                <div className="progress-bar-fill" style={{
                    width: '50%', // Indeterminate animation could be added here
                    height: '100%',
                    backgroundColor: 'var(--color-accent)',
                    borderRadius: '4px',
                    animation: 'pulse 1.5s infinite ease-in-out alternate'
                }}></div>
            </div>
            <style>{`
        @keyframes pulse {
          0% { width: 10%; opacity: 0.7; }
          100% { width: 90%; opacity: 1; }
        }
      `}</style>

            {onCancel && (
                <button
                    className="btn-secondary"
                    onClick={onCancel}
                    style={{ marginTop: 'var(--space-4)', color: 'var(--color-error)', borderColor: 'rgba(255, 74, 74, 0.3)' }}
                >
                    <XCircle size={18} /> Cancel Extraction
                </button>
            )}
        </div>
    );
}
