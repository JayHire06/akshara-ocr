import React from 'react';
import { Settings as SettingsIcon, Database, Zap, ShieldCheck } from 'lucide-react';

export default function SettingsScreen() {
    return (
        <div className="settings-container container animate-fade-in" style={{ padding: 'var(--space-4)', maxWidth: '600px' }}>
            <div className="flex items-center gap-2" style={{ marginBottom: 'var(--space-4)' }}>
                <SettingsIcon size={28} color="var(--color-accent)" />
                <h2 style={{ margin: 0 }}>System Status</h2>
            </div>

            <div className="card flex-col gap-4" style={{ marginBottom: 'var(--space-4)', padding: 'var(--space-4)' }}>
                <div>
                    <h3 style={{ marginBottom: 'var(--space-1)', fontSize: '1.2rem' }}>On-Device Engine</h3>
                    <p style={{ color: 'var(--color-muted)', fontSize: '0.9rem', marginBottom: 'var(--space-2)' }}>
                        Performance is optimized for your local hardware.
                    </p>
                    <div className="flex-col gap-2">
                        <div className="flex items-center gap-2" style={{ color: 'var(--color-accent)', fontSize: '0.9rem' }}>
                            <Zap size={16} /> Inference: <strong>Accelerated WebGPU</strong>
                        </div>
                        <div className="flex items-center gap-2" style={{ color: 'var(--color-muted)', fontSize: '0.9rem' }}>
                            <Database size={16} /> Privacy: <strong>100% Local Storage</strong>
                        </div>
                    </div>
                </div>

                <div style={{ borderTop: '1px solid var(--color-border)', paddingTop: 'var(--space-3)', marginTop: 'var(--space-2)' }}>
                   <div className="flex items-center gap-3" style={{ padding: 'var(--space-2)', borderRadius: 'var(--radius-button)', border: '1px solid var(--color-border)' }}>
                       <ShieldCheck size={24} color="var(--color-accent)" />
                       <div>
                           <div style={{ fontWeight: '600', fontSize: '1rem' }}>Data Sovereignty</div>
                           <div style={{ fontSize: '0.85rem', color: 'var(--color-muted)' }}>No data leaves this device.</div>
                       </div>
                   </div>
                </div>
            </div>

            <div className="card" style={{ padding: 'var(--space-4)', backgroundColor: 'rgba(198, 241, 53, 0.05)' }}>
                <h3 style={{ marginBottom: 'var(--space-1)', fontSize: '1.1rem' }}>Model Information</h3>
                <p style={{ color: 'var(--color-muted)', fontSize: '0.9rem', margin: 0 }}>
                    Current Model: <strong>Akshara V6 Edge (Hindi)</strong>. Optimized for Devanagari script recognition and multi-line paragraph segmentation.
                </p>
            </div>
        </div>
    );
}
