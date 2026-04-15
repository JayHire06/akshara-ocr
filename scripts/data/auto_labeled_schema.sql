-- Auto-labeled data pipeline work queue schema.
-- See docs/superpowers/specs/2026-04-16-external-auto-labeled-data-design.md §3.1

CREATE TABLE IF NOT EXISTS pages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL CHECK (source IN ('wikipedia','wikisource','gov_pdf','archive_org')),
    uri          TEXT NOT NULL,
    local_path   TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('pending','labeled','fetch_failed','label_failed')),
    error        TEXT
);

CREATE INDEX IF NOT EXISTS idx_pages_status ON pages(status);
CREATE INDEX IF NOT EXISTS idx_pages_source ON pages(source);

CREATE TABLE IF NOT EXISTS words (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id         INTEGER NOT NULL REFERENCES pages(id),
    x               INTEGER NOT NULL,
    y               INTEGER NOT NULL,
    w               INTEGER NOT NULL,
    h               INTEGER NOT NULL,
    text_nfc        TEXT NOT NULL,
    conf            REAL NOT NULL,
    provider        TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN (
                        'labeled','kept','exported_qa_candidate',
                        'filtered_low_conf','filtered_oov','filtered_invalid_bbox',
                        'filtered_dup','filtered_text_leak'
                    )),
    filter_reason   TEXT,
    qa_decision     TEXT CHECK (qa_decision IN ('accept','edit','reject')),
    qa_edited_text  TEXT,
    qa_decided_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_words_status ON words(status);
CREATE INDEX IF NOT EXISTS idx_words_text_nfc ON words(text_nfc);
CREATE INDEX IF NOT EXISTS idx_words_page_id ON words(page_id);

CREATE TABLE IF NOT EXISTS manifest (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
