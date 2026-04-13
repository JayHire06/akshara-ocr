import React from 'react';
import { ArrowRight, Cpu, WifiOff, Gauge, Layers } from 'lucide-react';

const METRICS = [
    { icon: <Gauge size={18} />, value: '79.37%', label: 'v8 best CRR',       sub: 'Staged curriculum' },
    { icon: <WifiOff size={18} />, value: '100%',  label: 'On-device',        sub: 'No server round-trip' },
    { icon: <Cpu size={18} />,    value: '~380ms', label: 'Median inference', sub: 'WebGPU / ONNX' },
    { icon: <Layers size={18} />, value: '8',      label: 'Model generations', sub: 'v1 → v8 shipped' }
];

// Mirrors scripts/inference/evaluate_all_versions.py output shape.
// Replace with JSON fetch once outputs/logs/cross_version_eval.log is parsed.
const BENCHMARK_ROWS = [
    { name: 'v1', label: 'Baseline',              cer: '—',  wer: '—',  note: 'Generic CRNN' },
    { name: 'v2', label: 'Early Aug.',            cer: '—',  wer: '—',  note: 'First augmentations' },
    { name: 'v3', label: '200K Extended',         cer: '—',  wer: '—',  note: 'Vocab expansion' },
    { name: 'v4', label: 'Realistic',             cer: '—',  wer: '—',  note: 'Document pool' },
    { name: 'v5', label: 'Prod Candidate',        cer: '—',  wer: '—',  note: 'Current production' },
    { name: 'v6', label: 'Edge STN',              cer: '—',  wer: '—',  note: 'MobileNet + STN' },
    { name: 'v7', label: 'NLP Reranker',          cer: '—',  wer: '—',  note: 'v6 + beam rerank' },
    { name: 'v8', label: 'Staged Curriculum',     cer: '—',  wer: '—',  note: 'Synth → printed → handwritten' },
    { name: 'v9', label: 'Transformer Encoder',   cer: '—',  wer: '—',  note: 'STN + MobileNet + 6L Transformer + EMA', highlight: true }
];

const STAGES = [
    { id: 'capture',   title: '01 · Capture',    desc: 'Any Devanagari image — scan, photo, screenshot.' },
    { id: 'preprocess',title: '02 · Preprocess', desc: 'Deskew, binarize, vertical-projection line segmentation.' },
    { id: 'infer',     title: '03 · Infer',      desc: 'CRNNv6 on-device via WebGPU ONNX, focal-CTC decoded.' },
    { id: 'rerank',    title: '04 · Rerank',     desc: 'Spelling beam + NLP rerank against Devanagari lexicon.' }
];

export default function LandingScreen({ onStart, demoCases = [] }) {
    return (
        <div className="lab-landing animate-fade-in">

            {/* ── HERO ────────────────────────────────────────────── */}
            <section className="lab-hero">
                <div className="lab-hero-copy">
                    <span className="lab-eyebrow">
                        <span className="lab-dot" /> On-device OCR · Devanagari focus
                    </span>
                    <h1 className="lab-title">
                        A benchmarked OCR stack<br />
                        <span className="lab-title-accent">built for Indic scripts.</span>
                    </h1>
                    <p className="lab-sub">
                        Eight model generations, one reproducible pipeline. Preprocessing to
                        decoding runs entirely in your browser — no uploads, no API keys, no
                        telemetry.
                    </p>
                    <div className="lab-cta-row">
                        <button className="btn-primary" onClick={onStart}>
                            Run a live extraction <ArrowRight size={18} />
                        </button>
                        <a
                            className="btn-secondary"
                            href="https://github.com/RandomArtist22/akshara-ocr"
                            target="_blank"
                            rel="noreferrer"
                        >
                            View source
                        </a>
                    </div>
                </div>

                <div className="lab-hero-metrics">
                    {METRICS.map((m) => (
                        <div className="lab-metric" key={m.label}>
                            <div className="lab-metric-icon">{m.icon}</div>
                            <div className="lab-metric-value">{m.value}</div>
                            <div className="lab-metric-label">{m.label}</div>
                            <div className="lab-metric-sub">{m.sub}</div>
                        </div>
                    ))}
                </div>
            </section>

            {/* ── PIPELINE STRIP ───────────────────────────────────── */}
            <section className="lab-section">
                <header className="lab-section-head">
                    <h2>Pipeline</h2>
                    <p>Four deterministic stages. Every one unit-tested.</p>
                </header>
                <div className="lab-pipeline">
                    {STAGES.map((s, i) => (
                        <div className="lab-pipeline-step" key={s.id}>
                            <div className="lab-pipeline-marker">{String(i + 1).padStart(2, '0')}</div>
                            <h3>{s.title.split('·')[1].trim()}</h3>
                            <p>{s.desc}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* ── DEMO CASES ───────────────────────────────────────── */}
            <section className="lab-section">
                <header className="lab-section-head">
                    <h2>Stress tests</h2>
                    <p>Each demo targets a known weakness in Devanagari recognition.</p>
                </header>
                <div className="demo-case-grid">
                    {demoCases.map((dc) => (
                        <article key={dc.id} className="demo-case-card">
                            <img src={dc.assetPath} alt={dc.title} className="demo-case-image" />
                            <div className="demo-case-body">
                                <div className="demo-case-meta">
                                    <span>{dc.script}</span>
                                    {dc.difficulty && (
                                        <span className={`difficulty difficulty-${dc.difficulty.toLowerCase()}`}>
                                            {dc.difficulty}
                                        </span>
                                    )}
                                </div>
                                <h3>{dc.title}</h3>
                                {dc.testsFor && (
                                    <div className="demo-case-tests">Tests: {dc.testsFor}</div>
                                )}
                                <p>{dc.note}</p>
                                <pre className="demo-case-output">{dc.expectedText}</pre>
                            </div>
                        </article>
                    ))}
                </div>
            </section>

            {/* ── BENCHMARK TABLE ──────────────────────────────────── */}
            <section className="lab-section">
                <header className="lab-section-head">
                    <h2>Cross-version benchmark</h2>
                    <p>
                        Every checkpoint evaluated on the same <code>data/combined/val</code> set
                        with identical preprocessing. Numbers populate after
                        <code> scripts/inference/evaluate_all_versions.py</code> completes.
                    </p>
                </header>
                <div className="lab-table-wrap">
                    <table className="lab-table">
                        <thead>
                            <tr>
                                <th>Version</th>
                                <th>Architecture</th>
                                <th className="num">CER</th>
                                <th className="num">WER</th>
                                <th>Notes</th>
                            </tr>
                        </thead>
                        <tbody>
                            {BENCHMARK_ROWS.map((row) => (
                                <tr key={row.name} className={row.highlight ? 'highlight' : ''}>
                                    <td className="mono">{row.name}</td>
                                    <td>{row.label}</td>
                                    <td className="num mono">{row.cer}</td>
                                    <td className="num mono">{row.wer}</td>
                                    <td className="muted">{row.note}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </section>

            {/* ── FOOTER CTA ──────────────────────────────────────── */}
            <section className="lab-footer-cta">
                <h2>Try it on your own document.</h2>
                <p>Processed in-browser. Nothing leaves your device.</p>
                <button className="btn-primary" onClick={onStart}>
                    Open the extractor <ArrowRight size={18} />
                </button>
            </section>
        </div>
    );
}
