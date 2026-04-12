import React from 'react';
import { FileText, Zap, Globe, Shield, Play } from 'lucide-react';

export default function LandingScreen({ onStart, demoCases = [] }) {
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
                <h1 style={{ fontSize: '3.5rem', marginBottom: 'var(--space-2)' }}>
                    Preserve your heritage.<br />
                    <span style={{ color: 'var(--color-accent)' }}>Digitize with Akshara.</span>
                </h1>
                <p style={{ color: 'var(--color-muted)', fontSize: '1.2rem', marginBottom: 'var(--space-4)', maxWidth: '600px' }}>
                    An advanced Optical Character Recognition tool designed precisely for complex Indian typography.
                </p>
                <button className="btn-primary" onClick={onStart} style={{ fontSize: '1.15rem', padding: 'var(--space-2) var(--space-4)', boxShadow: '0 4px 14px 0 rgba(198, 241, 53, 0.39)' }}>
                    Start Extracting Text
                </button>
            </div>

            <div className="demo-preview" style={{
                width: '100%',
                maxWidth: '900px',
                height: '400px',
                backgroundColor: '#0a0a0a',
                borderRadius: 'var(--radius-card)',
                marginTop: 'var(--space-6)',
                border: '1px solid var(--color-border)',
                boxShadow: '0 20px 40px rgba(0,0,0,0.5)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative',
                overflow: 'hidden'
            }}>
                <div style={{ position: 'absolute', inset: 0, backgroundImage: 'url(https://images.unsplash.com/photo-1544812852-628d712ce627?q=80&w=2070&auto=format&fit=crop)', backgroundSize: 'cover', opacity: 0.3, filter: 'blur(2px)' }}></div>
                <button style={{
                    width: '64px',
                    height: '64px',
                    borderRadius: '50%',
                    backgroundColor: 'var(--color-accent)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#000',
                    zIndex: 1,
                    boxShadow: '0 4px 14px 0 rgba(198, 241, 53, 0.39)',
                    cursor: 'pointer'
                }}>
                    <Play fill="#000" size={24} style={{ marginLeft: '4px' }} />
                </button>
            </div>

            <div className="demo-case-grid" style={{ width: '100%', maxWidth: '1100px', marginTop: 'var(--space-6)' }}>
                {demoCases.map((demoCase) => (
                    <article key={demoCase.id} className="demo-case-card">
                        <img src={demoCase.assetPath} alt={demoCase.title} className="demo-case-image" />
                        <div className="demo-case-body">
                            <div className="demo-case-meta">
                                <span>{demoCase.script}</span>
                                <span>{demoCase.languageLabel}</span>
                            </div>
                            <h3>{demoCase.title}</h3>
                            <p>{demoCase.note}</p>
                            <pre className="demo-case-output">{demoCase.expectedText}</pre>
                        </div>
                    </article>
                ))}
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
