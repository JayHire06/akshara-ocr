import React, { useState, useRef } from 'react';
import { UploadCloud, File as FileIcon } from 'lucide-react';

export default function UploadScreen({ onUploadSubmit, demoCases = [] }) {
    const [dragActive, setDragActive] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);
    const [loadingDemoId, setLoadingDemoId] = useState('');
    const [selectedDemoCase, setSelectedDemoCase] = useState(null);
    const [errorMsg, setErrorMsg] = useState('');

    const fileInputRef = useRef(null);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            handleFileSelection(e.target.files[0]);
        }
    };

    const handleFileSelection = (file) => {
        setErrorMsg('');
        setSelectedDemoCase(null);
        const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            setErrorMsg('Please upload a valid image (JPEG, PNG, WEBP).');
            return;
        }
        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            setErrorMsg('File size must be less than 10MB.');
            return;
        }
        setSelectedFile(file);
    };

    const handleDemoSelection = async (demoCase) => {
        setErrorMsg('');
        setLoadingDemoId(demoCase.id);

        try {
            const response = await fetch(demoCase.assetPath);
            if (!response.ok) {
                throw new Error('Unable to load the bundled sample case.');
            }

            const blob = await response.blob();
            const extension = blob.type.includes('png') ? 'png' : 'jpg';
            const file = new window.File([blob], `${demoCase.id}.${extension}`, { type: blob.type || 'image/png' });

            setSelectedFile(file);
            setSelectedDemoCase(demoCase);
        } catch (err) {
            setErrorMsg(err.message || 'Unable to load the sample case.');
        } finally {
            setLoadingDemoId('');
        }
    };

    const onSubmit = () => {
        if (!selectedFile) {
            setErrorMsg('Please select an image to extract text from.');
            return;
        }
        // Hardcoded to Hindi as per user requirement
        onUploadSubmit(selectedFile, 'hin');
    };

    return (
        <div className="upload-container container animate-fade-in" style={{ padding: 'var(--space-4)', maxWidth: '800px' }}>
            <h2 style={{ marginBottom: 'var(--space-1)' }}>Instant Hindi OCR</h2>
            <p style={{ color: 'var(--color-muted)', marginBottom: 'var(--space-4)' }}>
                Powered by high-performance on-device WebGPU inference.
            </p>

            {demoCases.length > 0 && (
                <section style={{ marginBottom: 'var(--space-4)' }}>
                    <h3 style={{ marginBottom: 'var(--space-2)' }}>Try Sample Documents</h3>
                    <div className="demo-case-grid compact">
                        {demoCases.map((demoCase) => (
                            <article key={demoCase.id} 
                                className={`demo-case-card compact ${selectedDemoCase?.id === demoCase.id ? 'selected' : ''}`}
                                onClick={() => handleDemoSelection(demoCase)}
                                style={{ cursor: 'pointer' }}
                            >
                                <img src={demoCase.assetPath} alt={demoCase.title} className="demo-case-image" />
                                <div className="demo-case-body">
                                    <div className="demo-case-meta">
                                        <span>{demoCase.script}</span>
                                        <span>{demoCase.languageLabel}</span>
                                    </div>
                                    <h3>{demoCase.title}</h3>
                                </div>
                            </article>
                        ))}
                    </div>
                </section>
            )}

            {/* Drag & Drop Zone */}
            <div
                className={`drag-drop-zone ${dragActive ? 'active' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                style={{
                    border: `2px dashed ${dragActive ? 'var(--color-accent)' : 'var(--color-border)'}`,
                    borderRadius: 'var(--radius-card)',
                    padding: 'var(--space-6) var(--space-4)',
                    textAlign: 'center',
                    backgroundColor: dragActive ? 'rgba(198, 241, 53, 0.05)' : 'var(--color-surface)',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    marginBottom: 'var(--space-4)'
                }}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/jpeg, image/png, image/webp"
                    onChange={handleChange}
                    style={{ display: 'none' }}
                />

                {!selectedFile ? (
                    <div className="flex-col items-center justify-center gap-2">
                        <UploadCloud size={48} color="var(--color-muted)" />
                        <h3>Tap to browse or Drop Image</h3>
                        <p style={{ color: 'var(--color-muted)' }}>JPG, PNG, WEBP (Max 10MB)</p>
                    </div>
                ) : (
                    <div className="flex-col items-center justify-center gap-2">
                        <FileIcon size={48} color="var(--color-accent)" />
                        <h3 style={{ marginTop: 'var(--space-2)' }}>{selectedFile.name}</h3>
                        <p style={{ color: 'var(--color-muted)' }}>{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                        <button
                            className="btn-secondary"
                            style={{ marginTop: 'var(--space-2)' }}
                            onClick={(e) => {
                                e.stopPropagation();
                                setSelectedFile(null);
                                setSelectedDemoCase(null);
                                if (fileInputRef.current) fileInputRef.current.value = '';
                            }}
                        >
                            Remove
                        </button>
                    </div>
                )}
            </div>

            {errorMsg && (
                <div style={{ color: 'var(--color-error)', marginBottom: 'var(--space-3)', padding: 'var(--space-2)', backgroundColor: 'rgba(255, 74, 74, 0.1)', borderRadius: 'var(--radius-chip)' }}>
                    {errorMsg}
                </div>
            )}

            <div style={{ padding: 'var(--space-4)', backgroundColor: 'rgba(198, 241, 53, 0.05)', borderRadius: 'var(--radius-card)', marginBottom: 'var(--space-4)', border: '1px solid var(--color-border)' }}>
                <h3 style={{ fontSize: '1.1rem', marginBottom: '4px' }}>Multi-line Hindi Support</h3>
                <p style={{ color: 'var(--color-muted)', fontSize: '0.9rem' }}>
                    The engine automatically detects and processes multiple lines of Devanagari text.
                </p>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', borderTop: '1px solid var(--color-border)', paddingTop: 'var(--space-4)', marginTop: 'var(--space-2)' }}>
                <button
                    className="btn-primary"
                    onClick={onSubmit}
                    disabled={!selectedFile}
                    style={{ fontSize: '1.2rem', padding: 'var(--space-2) var(--space-6)', width: '100%' }}
                >
                    Extract Hindi Text
                </button>
            </div>
        </div>
    );
}
