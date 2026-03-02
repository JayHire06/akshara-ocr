import React, { useState } from 'react';
import { Copy, Download, Check, ArrowLeft, CheckCircle2 } from 'lucide-react';

export default function ResultsScreen({ result, selectedLanguage, onBack }) {
    const [copied, setCopied] = useState(false);

    // Helper to map UI language codes to CSS fonts
    const getFontFamily = (langCode) => {
        switch (langCode) {
            case 'hin':
            case 'mar':
            case 'nep':
                return 'var(--font-devanagari)';
            case 'tam':
                return 'var(--font-tamil)';
            case 'ben':
                return 'var(--font-bengali)';
            default:
                return 'var(--font-body)';
        }
    };

    const handleCopy = () => {
        navigator.clipboard.writeText(result.text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const handleDownload = () => {
        const blob = new Blob([result.text], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `extracted_text_${Date.now()}.txt`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

    const charCount = result.text.length;
    const lineCount = result.text.split('\n').filter(l => l.trim().length > 0).length;

    return (
        <div className="results-container container animate-fade-in" style={{ padding: 'var(--space-4) 0', maxWidth: '900px' }}>
            <button
                onClick={onBack}
                style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-1)', color: 'var(--color-muted)', marginBottom: 'var(--space-3)' }}
            >
                <ArrowLeft size={16} /> Back to Upload
            </button>

            <div className="flex justify-between items-center" style={{ marginBottom: 'var(--space-3)' }}>
                <h2 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                    Extraction Complete
                    <CheckCircle2 size={24} color="var(--color-accent)" />
                </h2>
                {result.confidence && (
                    <div style={{
                        backgroundColor: 'rgba(198, 241, 53, 0.1)',
                        color: 'var(--color-accent)',
                        padding: '4px 12px',
                        borderRadius: 'var(--radius-chip)',
                        fontWeight: '600',
                        fontSize: '0.9rem'
                    }}>
                        {Math.round(result.confidence * 100)}% Confidence
                    </div>
                )}
            </div>

            <div className="stats-row flex" style={{ gap: 'var(--space-4)', marginBottom: 'var(--space-3)', color: 'var(--color-muted)', fontSize: '0.9rem' }}>
                <div><strong>{result.wordCount}</strong> Words</div>
                <div><strong>{charCount}</strong> Characters</div>
                <div><strong>{lineCount}</strong> Lines</div>
                {result.processingTimeMs && <div><strong>{(result.processingTimeMs / 1000).toFixed(2)}s</strong> Processing time</div>}
            </div>

            <div className="card" style={{ padding: '0', display: 'flex', flexDirection: 'column', height: '500px' }}>
                <div style={{
                    padding: 'var(--space-2) var(--space-3)',
                    borderBottom: '1px solid var(--color-border)',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    backgroundColor: 'var(--color-surface)',
                    borderTopLeftRadius: 'var(--radius-card)',
                    borderTopRightRadius: 'var(--radius-card)'
                }}>
                    <span style={{ fontWeight: '500', color: 'var(--color-muted)', textTransform: 'uppercase', fontSize: '0.8rem', letterSpacing: '0.5px' }}>
                        Result Text
                    </span>
                    <div className="flex gap-2">
                        <button className="btn-secondary" onClick={handleCopy} style={{ padding: '4px 12px', fontSize: '0.9rem' }}>
                            {copied ? <Check size={16} color="var(--color-accent)" /> : <Copy size={16} />}
                            {copied ? 'Copied!' : 'Copy'}
                        </button>
                        <button className="btn-primary" onClick={handleDownload} style={{ padding: '4px 12px', fontSize: '0.9rem' }}>
                            <Download size={16} /> Download
                        </button>
                    </div>
                </div>

                <div style={{
                    padding: 'var(--space-3)',
                    flex: 1,
                    overflowY: 'auto',
                    fontFamily: getFontFamily(selectedLanguage),
                    fontSize: '1.2rem',
                    lineHeight: '1.8',
                    whiteSpace: 'pre-wrap'
                }}>
                    {result.text}
                </div>
            </div>
        </div>
    );
}
