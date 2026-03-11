import React, { useState, useEffect, useRef } from 'react';
import { UploadCloud, File, Languages, CheckCircle } from 'lucide-react';
import { api } from '../services/api';

export default function UploadScreen({ onUploadSubmit }) {
    const [dragActive, setDragActive] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);
    const [selectedLanguage, setSelectedLanguage] = useState('');
    const [docType, setDocType] = useState('printed');
    const [languages, setLanguages] = useState([]);
    const [loadingLangs, setLoadingLangs] = useState(true);
    const [errorMsg, setErrorMsg] = useState('');

    const fileInputRef = useRef(null);

    useEffect(() => {
        // Fetch languages
        api.getLanguages()
            .then(res => {
                setLanguages(res);
                const defaultLg = localStorage.getItem('default_lang') || 'auto';
                setSelectedLanguage(defaultLg);
                setLoadingLangs(false);
            })
            .catch(err => {
                console.error('Failed to load languages', err);
                // Fallback for UI if API fails
                const fallback = [
                    { code: 'hin', name: 'Hindi/Devanagari', native_name: 'हिन्दी' },
                    { code: 'tam', name: 'Tamil', native_name: 'தமிழ்' },
                    { code: 'ben', name: 'Bengali', native_name: 'বাংলা' },
                    { code: 'eng', name: 'English', native_name: 'English' }
                ];
                setLanguages(fallback);
                setSelectedLanguage('auto');
                setLoadingLangs(false);
            });
    }, []);

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
        const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'application/pdf'];
        if (!validTypes.includes(file.type)) {
            setErrorMsg('Please upload a valid image (JPEG, PNG, WEBP) or PDF.');
            return;
        }
        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            setErrorMsg('File size must be less than 10MB.');
            return;
        }
        setSelectedFile(file);
    };

    const onSubmit = () => {
        if (!selectedFile) {
            setErrorMsg('Please select a file to extract text from.');
            return;
        }
        if (!selectedLanguage) {
            setErrorMsg('Please select a language.');
            return;
        }
        onUploadSubmit(selectedFile, selectedLanguage);
    };

    return (
        <div className="upload-container container animate-fade-in" style={{ padding: 'var(--space-4) 0', maxWidth: '800px' }}>
            <h2 style={{ marginBottom: 'var(--space-1)' }}>Upload Document</h2>
            <p style={{ color: 'var(--color-muted)', marginBottom: 'var(--space-4)' }}>
                Select a document containing the regional text you wish to extract.
            </p>

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
                    accept="image/jpeg, image/png, image/webp, application/pdf"
                    onChange={handleChange}
                    style={{ display: 'none' }}
                />

                {!selectedFile ? (
                    <div className="flex-col items-center justify-center gap-2">
                        <UploadCloud size={48} color="var(--color-muted)" />
                        <h3 style={{ marginTop: 'var(--space-2)' }}>Drag & drop a file here</h3>
                        <p style={{ color: 'var(--color-muted)' }}>or click to browse from your computer</p>
                        <p style={{ fontSize: '0.8rem', color: 'var(--color-muted)', marginTop: 'var(--space-1)' }}>
                            Supports JPG, PNG, WEBP, PDF (Max 10MB)
                        </p>
                    </div>
                ) : (
                    <div className="flex-col items-center justify-center gap-2">
                        <File size={48} color="var(--color-accent)" />
                        <h3 style={{ marginTop: 'var(--space-2)' }}>{selectedFile.name}</h3>
                        <p style={{ color: 'var(--color-muted)' }}>{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                        <button
                            className="btn-secondary"
                            style={{ marginTop: 'var(--space-2)' }}
                            onClick={(e) => {
                                e.stopPropagation();
                                setSelectedFile(null);
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

            {/* Language Selection */}
            <h3 style={{ marginBottom: 'var(--space-2)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Languages size={20} /> Select Primary Language
            </h3>

            {loadingLangs ? (
                <p style={{ color: 'var(--color-muted)' }}>Loading languages...</p>
            ) : (
                <div className="language-grid" style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))',
                    gap: 'var(--space-2)',
                    marginBottom: 'var(--space-4)'
                }}>
                    <div
                        className={`lang-card ${selectedLanguage === 'auto' ? 'selected' : ''}`}
                        onClick={() => setSelectedLanguage('auto')}
                        style={{
                            border: `1px solid ${selectedLanguage === 'auto' ? 'var(--color-accent)' : 'var(--color-border)'}`,
                            backgroundColor: selectedLanguage === 'auto' ? 'rgba(198, 241, 53, 0.1)' : 'var(--color-surface)',
                            padding: 'var(--space-2)',
                            borderRadius: 'var(--radius-button)',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            transition: 'all 0.2s',
                            gridColumn: '1 / -1',
                            fontWeight: '600',
                            color: selectedLanguage === 'auto' ? 'var(--color-accent)' : 'var(--color-text)'
                        }}
                    >
                        <div>Auto-Detect Language</div>
                        {selectedLanguage === 'auto' && <CheckCircle size={18} color="var(--color-accent)" style={{ marginLeft: '8px' }} />}
                    </div>
                    {languages.map((lang) => (
                        <div
                            key={lang.code}
                            className={`lang-card ${selectedLanguage === lang.code ? 'selected' : ''}`}
                            onClick={() => setSelectedLanguage(lang.code)}
                            style={{
                                border: `1px solid ${selectedLanguage === lang.code ? 'var(--color-accent)' : 'var(--color-border)'}`,
                                backgroundColor: selectedLanguage === lang.code ? 'rgba(198, 241, 53, 0.1)' : 'var(--color-surface)',
                                padding: 'var(--space-2)',
                                borderRadius: 'var(--radius-button)',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'space-between',
                                transition: 'all 0.2s',
                                transform: selectedLanguage === lang.code ? 'scale(1.02)' : 'scale(1)'
                            }}
                        >
                            <div>
                                <div style={{ fontWeight: '500' }}>{lang.name}</div>
                                {lang.native_name && <div style={{ fontSize: '0.8rem', color: 'var(--color-muted)' }}>{lang.native_name}</div>}
                            </div>
                            {selectedLanguage === lang.code && <CheckCircle size={18} color="var(--color-accent)" />}
                        </div>
                    ))}  </div>
            )}

            {/* Document Type Selection */}
            <div style={{ marginBottom: 'var(--space-4)' }}>
                <h3 style={{ marginBottom: 'var(--space-2)' }}>Document Type</h3>
                <div className="flex gap-2">
                    {['printed', 'handwritten'].map(type => (
                        <button
                            key={type}
                            onClick={() => setDocType(type)}
                            style={{
                                padding: 'var(--space-1) var(--space-3)',
                                borderRadius: 'var(--radius-chip)',
                                border: `1px solid ${docType === type ? 'var(--color-accent)' : 'var(--color-border)'}`,
                                backgroundColor: docType === type ? 'var(--color-accent)' : 'transparent',
                                color: docType === type ? '#000' : 'var(--color-text)',
                                textTransform: 'capitalize',
                                fontWeight: '500'
                            }}
                        >
                            {type}
                        </button>
                    ))}
                </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', borderTop: '1px solid var(--color-border)', paddingTop: 'var(--space-4)' }}>
                <button
                    className="btn-primary"
                    onClick={onSubmit}
                    disabled={!selectedFile || !selectedLanguage}
                    style={{ fontSize: '1.1rem', padding: 'var(--space-2) var(--space-4)' }}
                >
                    Extract Text
                </button>
            </div>
        </div>
    );
}
