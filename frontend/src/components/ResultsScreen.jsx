import React, { useState, useEffect } from 'react';
import { Copy, Download, Check, ArrowLeft, CheckCircle2, FileText, Image as ImageIcon } from 'lucide-react';

export default function ResultsScreen({ result, selectedLanguage, selectedFile, onBack }) {
    const [copied, setCopied] = useState(false);
    const [imagePreview, setImagePreview] = useState(null);

    useEffect(() => {
        if (selectedFile) {
            const objectUrl = URL.createObjectURL(selectedFile);
            setImagePreview(objectUrl);
            return () => URL.revokeObjectURL(objectUrl);
        }
    }, [selectedFile]);

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

    const handleDownloadTxt = () => {
        const blob = new Blob([result.text], { type: 'text/plain;charset=utf-8' });
        triggerDownload(blob, `extracted_text_${Date.now()}.txt`);
    };

    const handleDownloadDocx = () => {
        const htmlContent = `
            <html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
            <head><meta charset='utf-8'></head><body><p style="font-size: 14pt;">${result.text.replace(/\n/g, '<br>')}</p></body></html>
        `;
        const blob = new Blob(['\ufeff', htmlContent], { type: 'application/msword' });
        triggerDownload(blob, `extracted_text_${Date.now()}.doc`);
    };

    const handleDownloadPdf = () => {
        // Native print dialog allows saving as PDF easily without large third-party libraries
        window.print();
    };

    const triggerDownload = (blob, filename) => {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
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

            <div className="card" style={{ padding: '0', display: 'flex', flexDirection: 'column', height: '600px', '@media print': { height: 'auto', border: 'none', boxShadow: 'none' } }}>
                <div style={{
                    padding: 'var(--space-2) var(--space-3)',
                    borderBottom: '1px solid var(--color-border)',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    backgroundColor: 'var(--color-surface)',
                    borderTopLeftRadius: 'var(--radius-card)',
                    borderTopRightRadius: 'var(--radius-card)'
                }} className="print-hidden">
                    <span style={{ fontWeight: '500', color: 'var(--color-muted)', textTransform: 'uppercase', fontSize: '0.8rem', letterSpacing: '0.5px' }}>
                        Result Actions
                    </span>
                    <div className="flex gap-2">
                        <button className="btn-secondary" onClick={handleCopy} style={{ padding: '4px 12px', fontSize: '0.9rem' }}>
                            {copied ? <Check size={16} color="var(--color-accent)" /> : <Copy size={16} />}
                            {copied ? 'Copied!' : 'Copy'}
                        </button>
                        <button className="btn-secondary" onClick={handleDownloadTxt} style={{ padding: '4px 12px', fontSize: '0.9rem' }}>
                            <FileText size={16} /> TXT
                        </button>
                        <button className="btn-secondary" onClick={handleDownloadDocx} style={{ padding: '4px 12px', fontSize: '0.9rem' }}>
                            <FileText size={16} /> DOCX
                        </button>
                        <button className="btn-primary" onClick={handleDownloadPdf} style={{ padding: '4px 12px', fontSize: '0.9rem' }}>
                            <Download size={16} /> PDF
                        </button>
                    </div>
                </div>

                <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
                    {imagePreview && (
                        <div style={{
                            flex: 1,
                            borderRight: '1px solid var(--color-border)',
                            backgroundColor: '#000',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            position: 'relative',
                            overflow: 'hidden'
                        }} className="print-hidden">
                            <span style={{ position: 'absolute', top: '10px', left: '10px', backgroundColor: 'rgba(0,0,0,0.6)', padding: '4px 8px', borderRadius: '4px', fontSize: '0.8rem', color: '#fff', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                <ImageIcon size={14} /> Original Source
                            </span>
                            <img src={imagePreview} alt="Original Document" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
                        </div>
                    )}
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

            <style>{`
                @media print {
                    body * {
                        visibility: hidden;
                    }
                    .results-container, .results-container * {
                        visibility: visible;
                    }
                    .results-container {
                        position: absolute;
                        left: 0;
                        top: 0;
                        width: 100%;
                    }
                    .print-hidden {
                        display: none !important;
                    }
                }
            `}</style>
        </div>
    );
}
