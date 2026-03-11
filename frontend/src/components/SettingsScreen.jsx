import React, { useState } from 'react';
import { Settings as SettingsIcon, LogOut, Check } from 'lucide-react';

export default function SettingsScreen({ onLogout }) {
    const [defaultLang, setDefaultLang] = useState(localStorage.getItem('default_lang') || 'hin');
    const [saved, setSaved] = useState(false);

    const handleSave = () => {
        localStorage.setItem('default_lang', defaultLang);
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
    };

    return (
        <div className="settings-container container animate-fade-in" style={{ padding: 'var(--space-4) 0', maxWidth: '600px' }}>
            <div className="flex items-center gap-2" style={{ marginBottom: 'var(--space-4)' }}>
                <SettingsIcon size={28} color="var(--color-accent)" />
                <h2 style={{ margin: 0 }}>Preferences</h2>
            </div>

            <div className="card flex-col gap-4" style={{ marginBottom: 'var(--space-4)' }}>
                <div>
                    <h3 style={{ marginBottom: 'var(--space-1)', fontSize: '1.1rem' }}>Default Language</h3>
                    <p style={{ color: 'var(--color-muted)', fontSize: '0.9rem', marginBottom: 'var(--space-2)' }}>
                        Select the language to be selected by default when you upload a document.
                    </p>
                    <select
                        value={defaultLang}
                        onChange={(e) => setDefaultLang(e.target.value)}
                        style={{
                            padding: '10px 16px',
                            borderRadius: 'var(--radius-button)',
                            border: '1px solid var(--color-border)',
                            backgroundColor: 'var(--color-surface)',
                            color: 'var(--color-text)',
                            width: '100%',
                            maxWidth: '300px',
                            outline: 'none',
                            fontFamily: 'inherit'
                        }}
                    >
                        <option value="hin">Hindi (हिन्दी)</option>
                        <option value="tam">Tamil (தமிழ்)</option>
                        <option value="ben">Bengali (বাংলা)</option>
                        <option value="mar">Marathi (मराठी)</option>
                        <option value="nep">Nepali (नेपाली)</option>
                        <option value="eng">English</option>
                    </select>
                </div>

                <div style={{ borderTop: '1px solid var(--color-border)', paddingTop: 'var(--space-3)' }}>
                    <button className="btn-primary" onClick={handleSave}>
                        {saved ? <><Check size={18} /> Saved</> : 'Save Preferences'}
                    </button>
                </div>
            </div>

            <div className="card" style={{ borderColor: 'rgba(255, 74, 74, 0.3)' }}>
                <h3 style={{ marginBottom: 'var(--space-1)', fontSize: '1.1rem', color: 'var(--color-error)' }}>Account Actions</h3>
                <p style={{ color: 'var(--color-muted)', fontSize: '0.9rem', marginBottom: 'var(--space-3)' }}>
                    Sign out of your Akshara OCR account on this device.
                </p>
                <button
                    className="btn-secondary"
                    onClick={onLogout}
                    style={{ color: 'var(--color-error)', borderColor: 'rgba(255, 74, 74, 0.3)' }}
                >
                    <LogOut size={18} /> Sign Out
                </button>
            </div>
        </div>
    );
}
