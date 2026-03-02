import React from 'react';
import { FileText, Zap, Globe, Shield } from 'lucide-react';

export default function LandingScreen({ onStart }) {
    const features = [
        {
            icon: <FileText size={24} className="feature-icon" />,
            title: 'High Accuracy',
            desc: 'State-of-the-art models trained specifically on Indian scripts.'
        },
        {
            icon: <Globe size={24} className="feature-icon" />,
            title: 'Multi-lingual Support',
            desc: 'Extract text from Devanagari, Tamil, Bengali, and more.'
        },
        {
            icon: <Zap size={24} className="feature-icon" />,
            title: 'Lightning Fast',
            desc: 'Get your text extracted in seconds, not minutes.'
        },
        {
            icon: <Shield size={24} className="feature-icon" />,
            title: 'Secure & Private',
            desc: 'Your documents are never stored permanently.'
        }
    ];

    return (
        <div className="landing-container flex-col items-center justify-center animate-fade-in" style={{ padding: 'var(--space-8) var(--space-3)' }}>
            <div className="hero-section flex-col items-center" style={{ textAlign: 'center', maxWidth: '800px', margin: '0 auto', gap: 'var(--space-3)' }}>
                <h1 style={{ fontSize: '3rem', marginBottom: 'var(--space-2)' }}>
                    Preserve your heritage.<br />
                    <span style={{ color: 'var(--color-accent)' }}>Digitize with Akshara.</span>
                </h1>
                <p style={{ color: 'var(--color-muted)', fontSize: '1.2rem', marginBottom: 'var(--space-4)', maxWidth: '600px' }}>
                    An advanced Optical Character Recognition tool designed precisely for complex Indian typography.
                </p>
                <button className="btn-primary" onClick={onStart} style={{ fontSize: '1.1rem', padding: 'var(--space-2) var(--space-4)' }}>
                    Start Extracting Text
                </button>
            </div>

            <div className="features-grid" style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
                gap: 'var(--space-4)',
                maxWidth: '1000px',
                width: '100%',
                marginTop: 'var(--space-8)'
            }}>
                {features.map((f, i) => (
                    <div key={i} className="card flex-col items-center" style={{ textAlign: 'center', gap: 'var(--space-2)' }}>
                        <div style={{ color: 'var(--color-accent)', marginBottom: 'var(--space-1)' }}>
                            {f.icon}
                        </div>
                        <h3 style={{ fontSize: '1.25rem' }}>{f.title}</h3>
                        <p style={{ color: 'var(--color-muted)', fontSize: '0.9rem' }}>{f.desc}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}
