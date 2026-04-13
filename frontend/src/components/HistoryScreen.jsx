import React, { useState, useEffect } from 'react';
import { Clock, ChevronLeft, ChevronRight, FileText, Trash2 } from 'lucide-react';
import { localHistory } from '../services/localHistory';

export default function HistoryScreen() {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadData();
    }, []);

    const loadData = () => {
        setLoading(true);
        const data = localHistory.getHistory();
        setHistory(data);
        setLoading(false);
    };

    const handleClear = () => {
        if (window.confirm("Clear all locally stored extraction history?")) {
            localHistory.clearHistory();
            setHistory([]);
        }
    };

    const formatDate = (isoString) => {
        const d = new Date(isoString);
        return new Intl.DateTimeFormat('en-IN', {
            day: 'numeric', month: 'short', year: 'numeric',
            hour: '2-digit', minute: '2-digit'
        }).format(d);
    };

    return (
        <div className="history-container container animate-fade-in" style={{ padding: 'var(--space-4)', maxWidth: '800px' }}>
            <div className="flex items-center justify-between" style={{ marginBottom: 'var(--space-4)' }}>
                <div className="flex items-center gap-2">
                    <Clock size={28} color="var(--color-accent)" />
                    <h2 style={{ margin: 0 }}>Offline History</h2>
                </div>
                {history.length > 0 && (
                    <button className="btn-secondary" onClick={handleClear} style={{ color: '#ff4d4d', borderColor: '#ff4d4d' }}>
                        <Trash2 size={16} style={{ marginRight: '6px' }} /> Clear All
                    </button>
                )}
            </div>

            {loading ? (
                <div style={{ padding: 'var(--space-4)', textAlign: 'center', color: 'var(--color-muted)' }}>
                    Accessing local storage...
                </div>
            ) : history.length === 0 ? (
                <div className="card" style={{ textAlign: 'center', padding: 'var(--space-6)' }}>
                    <FileText size={48} color="var(--color-border)" style={{ margin: '0 auto var(--space-2)' }} />
                    <h3>No Local Data</h3>
                    <p style={{ color: 'var(--color-muted)' }}>Extracts are stored securely in your browser's private storage.</p>
                </div>
            ) : (
                <div className="history-list flex-col gap-3">
                    {history.map((item) => (
                        <div key={item.id} className="card flex items-center justify-between" style={{ padding: 'var(--space-4)', transition: 'transform 0.2s', cursor: 'pointer' }}>
                            <div className="flex-col gap-1" style={{ flex: 1, overflow: 'hidden' }}>
                                <div className="flex items-center gap-2">
                                    <span style={{
                                        backgroundColor: 'rgba(198, 241, 53, 0.1)',
                                        color: 'var(--color-accent)',
                                        padding: '2px 8px',
                                        borderRadius: 'var(--radius-chip)',
                                        fontSize: '0.8rem',
                                        fontWeight: '600'
                                    }}>
                                        {item.language}
                                    </span>
                                    <span style={{ color: 'var(--color-muted)', fontSize: '0.85rem' }}>
                                        {formatDate(item.timestamp)}
                                    </span>
                                </div>
                                <p style={{
                                    whiteSpace: 'nowrap',
                                    overflow: 'hidden',
                                    textOverflow: 'ellipsis',
                                    margin: 0,
                                    fontSize: '1.2rem',
                                    color: 'var(--color-text)'
                                }}>
                                    {item.text}
                                </p>
                                <div style={{ color: 'var(--color-muted)', fontSize: '0.8rem' }}>
                                    {item.fileName} • {item.processingTimeMs}ms
                                </div>
                            </div>
                            <ChevronRight size={20} color="var(--color-muted)" />
                        </div>
                    ))}
                    <p style={{ textAlign: 'center', color: 'var(--color-muted)', fontSize: '0.8rem', marginTop: 'var(--space-4)' }}>
                        Last 50 entries stored locally.
                    </p>
                </div>
            )}
        </div>
    );
}
