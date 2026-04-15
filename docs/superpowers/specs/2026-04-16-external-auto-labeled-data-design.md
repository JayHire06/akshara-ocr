# External Auto-Labeled Data Pipeline — Design Spec

**Date:** 2026-04-16
**Status:** Design — awaiting implementation plan
**Author:** Drafted via brainstorming session, 2026-04-16
**Related docs:**
- `docs/v8-data-preparation-plan.md` (v8 data conventions — this pipeline inherits them)
- `docs/v9-design.md` (D2 in §3 names "Hindi Wikipedia screenshots, auto-labelled" as deferred work; this spec is the fuller version of that item)
- `scripts/data/rebuild_text_disjoint_split.py` (the v9 post-mortem fix that motivates §3 below)

---

## 1. Motivation

The v1–v8 leaderboard shows inconsistent accuracies across model versions, and the v9 honest-leap post-mortem identified the root causes: a 79.6% text-leaky val split, orphaned CRNN checkpoints, and silent `strict=False` loading. All three are fixed. The remaining weakness is **training-data diversity** — the current training pool relies heavily on synthetic generation plus a handful of IIIT datasets, and is thin on real printed-document variety.

`docs/v9-design.md` already lists `D2: "Clean Hindi Wikipedia screenshots, auto-labelled via v8 + QA"` as a deferred v9.1/v9.2 item. That framing has a subtle flaw: using v8 itself as the teacher reinforces v8's own blind spots. An **external** teacher (a cloud OCR API) breaks that self-reinforcement loop and gives the student a genuinely independent supervision signal.

This spec designs the full pipeline: scraping public-domain Hindi printed content, labeling it with a cloud OCR teacher, filtering it under a strict text-disjoint policy that prevents v9-style benchmark contamination, and producing both a new training stage and a human-verified test suite.

## 2. Goals and non-goals

**In scope:**
- Scrape printed Hindi content from four public-domain sources: Wikipedia, Wikisource, Government-of-India PDFs, and archive.org Hindi book scans.
- Pseudo-label the scraped word crops using a cloud OCR API (Azure Read as default, pluggable).
- Apply strict confidence, OOV, and text-disjoint filtering.
- Produce a v8-compatible training stage of ~150K–200K pseudo-labeled word crops.
- Produce a 2,000-row human-verified test suite via an extended in-browser QA tool.
- Wire the outputs into the v9 training curriculum and the benchmark harness.

**Out of scope (explicitly deferred):**
- Handwriting scraping. Azure Read is weak on handwriting; the existing IIIT-HW-Words and IIIT-HW-UC datasets already cover this domain.
- Scene text scraping. Different failure mode; `v8-data-preparation-plan.md` scopes scene text as a separate Tier-3 concern.
- Tier-2 sources (news sites, blogs). License restrictions prevent redistribution, and scraping complexity is high for an unclear return.
- Line-level or page-level recognition. Pipeline stays word-level to match the existing vocab, dataloader, CRNN/v9 Transformer, and benchmark builder.
- Automatic retraining or scheduled reruns. The pipeline is manual per v9 iteration.
- Multimodal-LLM teacher (GPT-4o, Gemini, Claude). Provider interface is abstract, so this is a future additive change, not v1.
- A web dashboard. Manifest and terminal stats are the only observability.

## 3. Architecture overview

### 3.1 Data flow

```
scrape_external_pages.py  ──▶  work.db (pages)  ──▶  label_with_cloud_ocr.py
                                                             │
                                                             ▼
                                                      work.db (words)
                                                             │
                                                             ▼
                                               filter_and_crop_words.py
                                                             │
                            ┌────────────────────────────────┼─────────────────────────────┐
                            ▼                                ▼                             ▼
                 normalized/train_labels.txt     normalized/val_labels.txt      normalized/qa_candidates.txt
                            │                                                             │
                            ▼                                                             ▼
                  build_auto_labeled_stage.py                                   QA verification tool
                            │                                                             │
                            ▼                                                             ▼
              data/v8/stages/stage_auto_labeled_printed/                 normalized/verified_test_labels.txt
```

### 3.2 On-disk layout

```
data/external/auto_labeled/
├── raw/
│   ├── wikipedia/
│   ├── wikisource/
│   ├── gov_pdf/
│   └── archive_org/
├── work.db
├── normalized/
│   ├── crops/
│   ├── train_labels.txt
│   ├── val_labels.txt
│   ├── qa_candidates.txt
│   └── verified_test_labels.txt
├── manifests/
│   └── manifest.json
└── README.md
```

### 3.3 Flow invariants

1. The SQLite work queue (`work.db`) is the single source of truth during a pipeline run. Flat files under `normalized/` are regenerated from the DB; they are never hand-edited.
2. Each stage is resumable at the row level: killing the labeler mid-run and restarting it continues from where it stopped, because row status transitions are committed per-row.
3. The QA tool is the only stage that consumes human time. Every upstream stage runs unattended.
4. `verified_test_labels.txt` is frozen once QA completes. Its SHA256 is recorded in the manifest. It never re-enters the training pool on subsequent runs.

## 4. Components

### 4.1 `scripts/data/scrape_external_pages.py`

**Purpose:** Fetch pages from the four source tiers and record them as `pending` rows in `work.db`.

**CLI:**
- `--source {wikipedia,wikisource,gov_pdf,archive_org}`
- `--limit N` (hard cap on pages per run)
- `--db data/external/auto_labeled/work.db`

**Per-source fetch strategy:**

| Source | Acquisition | Rendering |
|---|---|---|
| Wikipedia | MediaWiki API → random article by namespace | Playwright headless screenshot, full height |
| Wikisource | MediaWiki API → ProofreadPage transcriptions | Playwright headless screenshot |
| Gov PDFs | Static seed URL list (data.gov.in, egazette, judgments) → `httpx` download | `pdf2image` rasterize at 300 DPI |
| archive.org | Advanced Search API (language=hin, subject=text) → download `_jp2.zip` | Extract and rasterize selected pages |

**Why Playwright for Wikipedia/Wikisource:** Hindi rendering depends on the browser's text shaper. `requests.get` returns raw HTML, not shaped glyphs. Playwright produces exactly what a human would see, which is what we want the student model to learn.

**Failure handling:**
- Network retries via `tenacity` with exponential backoff, max 5 attempts, then `status='fetch_failed'` in DB.
- Blank-render detection via minimum non-blank pixel threshold before saving a screenshot.
- Per-page timeout on PDF rasterization to prevent wedging.

### 4.2 `scripts/data/label_with_cloud_ocr.py`

**Purpose:** Drain `pages` rows with `status='pending'`, label each page with the teacher provider, write `words` rows.

**CLI:**
- `--db data/external/auto_labeled/work.db`
- `--provider {azure,google,aws}` (default `azure`, decided at runtime by available credentials)
- `--concurrency N` (default 8, capped by provider rate limit)

**Outputs:**
- `words` rows: `(page_id, x, y, w, h, text_nfc, teacher_confidence, teacher_provider, status='labeled')`.
- `pages.status` transitions to `labeled` or `label_failed`.

**Internals:**
- `asyncio.Semaphore(N)` guards concurrent outbound requests.
- Each request wrapped in `tenacity.retry(stop_after_attempt=3, wait=exponential(2, 30))`.
- Per-call timeout 60s.
- NFC normalization applied at write time. The DB only ever stores NFC text.
- Provider interface is abstract (`label_page(path) -> list[WordBox]`), so the fake-provider test double and future providers share the same contract.

**Empty-response handling:** An Azure response with zero word boxes on a visibly populated page is recorded as `status='labeled'` with no child `words` rows. A manifest counter tracks empty-response rate; the filter stage prints a loud warning if it exceeds 5% of labeled pages.

### 4.3 `scripts/data/filter_and_crop_words.py`

**Purpose:** Apply confidence, OOV, and text-disjoint filtering; crop kept word-boxes to PNG; emit normalized label files; select QA candidates.

**CLI:**
- `--db data/external/auto_labeled/work.db`
- `--min-conf 0.85`
- `--vocab data/vocab.json`
- `--existing-val data/combined/val_labels.txt`
- `--existing-test data/combined/test_labels.txt`
- `--existing-benchmarks data/benchmarks/*/labels.txt`

**Outputs:**
- `normalized/crops/<word_id>.png` (height-normalized to 32 px).
- `normalized/train_labels.txt` (pipe-format, training pool).
- `normalized/val_labels.txt` (10% stratified-by-source holdout for automated val).
- `normalized/qa_candidates.txt` (2,000 stratified-equal rows, 500 per source).
- `words.status` transitions to one of: `kept`, `filtered_low_conf`, `filtered_oov`, `filtered_text_leak`, `filtered_dup`, `filtered_invalid_bbox`, `exported_qa_candidate`.

**Filter ordering (performance and correctness both benefit):**
1. `labeled → filtered_low_conf` if `teacher_confidence < --min-conf`.
2. `labeled → filtered_oov` if `text_nfc` contains any character not in `vocab.json`.
3. `labeled → filtered_invalid_bbox` if `(x, y, w, h)` falls outside the page image after clamping.
4. `labeled → filtered_dup` if an identical `(page_id, text_nfc, x, y)` row already exists.
5. `labeled → filtered_text_leak` (text-disjoint policy B — see §5 below).
6. `labeled → kept` for survivors.
7. Second pass: promote 2,000 kept rows (500 per source) to `exported_qa_candidate`, selecting only rows whose `text_nfc` is unique within the `kept` set (see §5.3 for the SQL). A third, defense-in-depth pass re-filters any remaining `kept` rows whose `text_nfc` appears in the QA candidate set — in a correct run this pass is a no-op.
8. Crop kept rows to `normalized/crops/`. Emit label files.

### 4.4 `scripts/data/build_auto_labeled_stage.py`

**Purpose:** Produce a v8-compatible stage file from the filter output. Delegates to the existing helper in `data/v8_dataset_prep.py` so stage-file format is shared.

**CLI:**
- `--stage-name stage_auto_labeled_printed`
- `--train-labels data/external/auto_labeled/normalized/train_labels.txt`
- `--val-labels data/external/auto_labeled/normalized/val_labels.txt`
- `--notes "Auto-labeled Tier-1 + Tier-4 printed Hindi, teacher={provider}, conf≥{threshold}"`

**Output:** `data/v8/stages/stage_auto_labeled_printed/` with the same structure as existing v8 stages.

### 4.5 `data/labeling_tool/` — extended in place

See §6 for the QA tool design. New endpoints: `GET /qa`, `POST /qa/decide`, `GET /qa/stats`. The existing `GET /` and `POST /submit` for manual-from-scratch labeling remain untouched.

### 4.6 `scripts/data/run_auto_labeled_pipeline.sh`

Shell orchestrator chaining 4.1 → 4.2 → 4.3 → 4.4 with sensible defaults. Primary purpose: one-command smoke test.

## 5. Text-disjoint policy enforcement (Policy B)

This is the most load-bearing section, because the v9 post-mortem showed 79.6% text leakage between train and val in a prior pipeline. Getting this wrong re-introduces that bug in a new form.

### 5.1 Policy B, formally

Let **P** = the new pseudo-labeled pool from this pipeline.
Let **E** = the union of `data/combined/val_labels.txt`, `data/combined/test_labels.txt`, and all `data/benchmarks/*/labels.txt` text-strings.
Let **P_train** = P's training partition (rows ending at `kept`).
Let **P_test** = P's verified test partition (rows ending at `exported_qa_candidate` that are accepted or edited by QA).

Required:
1. `text-strings(P_train) ∩ E = ∅`
2. `text-strings(P_test) ∩ (E ∪ text-strings(P_train)) = ∅`

Rule 1 prevents new training data from polluting the existing benchmarks. Rule 2 prevents the new verified test suite from being polluted by either the existing benchmarks OR the new training pool (symmetric).

**Not required:** disjointness between `P_train` and the existing `data/combined/train_labels.txt`. Repeated text-strings across training pools is how typeface diversity happens, and the v9 bug was specifically about train/val overlap, not about shared training-side vocabulary.

### 5.2 Normalization before comparison

All strings are compared after:

```python
s_norm = unicodedata.normalize("NFC", s).strip()
```

No case-folding (Devanagari has no case), no punctuation stripping, no ZWJ/ZWNJ removal. NFC is the canonical form; this matches `vocab.json` and `rebuild_text_disjoint_split.py`.

### 5.3 SQL implementation (filter stage)

Populate three temp tables from the existing flat files, each with a `text_nfc` column and a primary-key index:

```sql
CREATE TEMP TABLE existing_val_strings   (text_nfc TEXT PRIMARY KEY);
CREATE TEMP TABLE existing_test_strings  (text_nfc TEXT PRIMARY KEY);
CREATE TEMP TABLE existing_bench_strings (text_nfc TEXT PRIMARY KEY);
```

Each row is NFC-normalized on the Python side before `INSERT`.

Rule 1 enforcement:

```sql
UPDATE words
SET status = 'filtered_text_leak',
    filter_reason = 'leak_vs_existing_val_test_bench'
WHERE status = 'labeled'
  AND text_nfc IN (
      SELECT text_nfc FROM existing_val_strings
      UNION
      SELECT text_nfc FROM existing_test_strings
      UNION
      SELECT text_nfc FROM existing_bench_strings
  );
```

QA candidate selection (rule 2). The key constraint is that the candidate's `text_nfc` must appear **exactly once** in the current `kept` set. Picking from unique-text rows makes the symmetric eviction step a no-op by construction: a candidate's text can't collide with any other training row because no other training row shares that text.

```sql
WITH unique_text_kept AS (
    SELECT id, text_nfc, page_id
    FROM words
    WHERE status = 'kept'
      AND text_nfc IN (
          SELECT text_nfc FROM words WHERE status = 'kept'
          GROUP BY text_nfc HAVING COUNT(*) = 1
      )
      AND text_nfc NOT IN (SELECT text_nfc FROM existing_val_strings)
      AND text_nfc NOT IN (SELECT text_nfc FROM existing_test_strings)
      AND text_nfc NOT IN (SELECT text_nfc FROM existing_bench_strings)
),
ranked AS (
    SELECT u.id, u.text_nfc, p.source,
           ROW_NUMBER() OVER (PARTITION BY p.source ORDER BY RANDOM()) AS rn
    FROM unique_text_kept u JOIN pages p ON p.id = u.page_id
)
UPDATE words
SET status = 'exported_qa_candidate'
WHERE id IN (SELECT id FROM ranked WHERE rn <= 500);
```

This requires SQLite 3.25+ (for window functions). The repo's existing `scripts/db/` setup uses a recent enough SQLite — no new dependency.

**Yield assumption:** The `HAVING COUNT(*) = 1` predicate restricts candidates to text-strings that appear exactly once in `kept`. With ~150K–200K kept rows and typical Hindi word-frequency distributions, there are reliably far more than 2,000 unique-text rows per source, so hitting 500/source is easy. If a pathological run ever falls short, the fallback is to relax the predicate to `COUNT(*) ≤ K` and accept the eviction cost — documented in the implementation plan.

Rule 2 symmetric enforcement (second pass — belt-and-suspenders). Because candidates were selected from unique-text rows, this UPDATE finds zero matches in a correct run. It exists as a defense-in-depth check and as an explicit enforcement statement that future code can't accidentally weaken:

```sql
UPDATE words
SET status = 'filtered_text_leak',
    filter_reason = 'leak_vs_qa_candidate'
WHERE status = 'kept'
  AND text_nfc IN (
      SELECT DISTINCT text_nfc FROM words WHERE status = 'exported_qa_candidate'
  );
```

If this UPDATE ever affects a nonzero row count, the filter stage aborts with a loud error — that means the unique-text predicate was violated, which is a bug worth investigating immediately.

### 5.4 Verification audits

Two independent correctness checks run at the end of the filter stage. Both must pass or the stage aborts with a loud error.

**Audit 1: Count-based.** Four queries that must each return exactly zero:

```sql
-- Existing val vs new training.
SELECT COUNT(*) FROM words
 WHERE status='kept' AND text_nfc IN (SELECT text_nfc FROM existing_val_strings);
-- Existing test vs new training.
SELECT COUNT(*) FROM words
 WHERE status='kept' AND text_nfc IN (SELECT text_nfc FROM existing_test_strings);
-- Existing benchmarks vs new training.
SELECT COUNT(*) FROM words
 WHERE status='kept' AND text_nfc IN (SELECT text_nfc FROM existing_bench_strings);
-- New training vs new verified test (symmetric).
SELECT COUNT(*) FROM words w1
 WHERE w1.status='kept'
   AND w1.text_nfc IN (SELECT text_nfc FROM words WHERE status='exported_qa_candidate');
```

**Audit 2: Random-sample Levenshtein-1.** Draw 100 random `(kept_row, existing_val_row)` pairs where `Levenshtein(kept_row.text_nfc, existing_val_row.text_nfc) ≤ 1`. Emit them to `manifests/levenshtein_audit.txt` for human spot-check. This catches silent normalization bugs (e.g., trailing-space differences) that the count-based audit would miss.

### 5.5 Edge cases explicitly handled

| Edge case | Handling |
|---|---|
| Non-NFC text from Azure | Normalized at write time in 4.2. DB stores only NFC. |
| Whitespace-only strings | Rejected at write time; never enter `words`. |
| Empty text | Same. |
| ZWJ / ZWNJ | Preserved (load-bearing for Devanagari conjuncts). Compared as-is. |
| Non-Devanagari characters (Latin, digits, punctuation) | Kept if in `vocab.json`, else `filtered_oov`. |
| Stale non-NFC entries in existing label files | Re-normalized on load into temp tables. Source files are not rewritten. |
| Bounding box outside page dimensions | Clamp to page bounds; if clamped box is empty, `filtered_invalid_bbox`. |

## 6. QA verification tool

### 6.1 Starting point and insight

The existing `data/labeling_tool/main.py` (62 lines, FastAPI + Jinja) is a manual-from-scratch labeler with no keyboard shortcuts, no teacher-prediction concept, no progress tracking, and no stratification. Using it as-is for 2,000 QA samples would take 16+ hours and almost certainly not finish.

Pre-filled QA is a fundamentally different task from scratch labeling: you're mostly confirming, occasionally catching errors. The tool must optimize the accept path until it's nearly free, and only make the edit and reject paths comfortable. Done right, 2,000 samples take 60–90 minutes in a single focused session.

### 6.2 New endpoints (extension of existing app)

| Endpoint | Method | Purpose |
|---|---|---|
| `/qa` | GET | Render next unverified candidate |
| `/qa/decide` | POST | Accept/edit/reject, advance to next |
| `/qa/stats` | GET | JSON blob with per-source progress counts |

Existing `/` and `/submit` are not modified.

### 6.3 Template shows, at 4× zoom

- Cropped word image, prominent.
- Teacher prediction, large, selectable for copy.
- Source tier badge (Wikipedia / Wikisource / gov_pdf / archive_org).
- Teacher confidence score.
- Progress bar keyed to current source bucket.

### 6.4 Keyboard bindings

| Key | Action |
|---|---|
| `Enter` or `Space` | Accept teacher prediction, advance |
| `e` | Open inline edit box pre-filled with prediction |
| `Enter` (in edit box) | Submit edit |
| `Esc` (in edit box) | Cancel edit |
| `r` | Reject sample entirely |
| `b` | Go back to previous candidate |
| `?` | Toggle shortcuts overlay |

### 6.5 Stratification

The `/qa` endpoint picks the next candidate in a round-robin over the four source buckets, so the human hits roughly equal samples from each tier as they progress. This prevents verifying 1,200 Wikipedia samples and then quitting with gov_pdf under-verified.

### 6.6 Session resumability

The work DB is the source of truth. Quitting the browser mid-session and returning later resumes exactly where you left off. No local state, no cookies, no "save your work" button.

### 6.7 Export

When all 2,000 candidates have a decision, the tool shows a "complete" screen with a button to emit `normalized/verified_test_labels.txt`. Clicking runs an export routine that:

1. Filters rows to `accept` or `edit` decisions only (rejects are dropped entirely — **Decision A from §9**).
2. Writes the pipe-format file atomically (temp file + `os.replace`).
3. Computes SHA256 of the final file.
4. Writes hash plus split counts to the manifest.

### 6.8 What the QA tool deliberately does not do

- No editing already-accepted samples without explicit back navigation.
- No auto-submit on idle, no auto-accept by confidence, no skipping.
- No modification of rows outside the `exported_qa_candidate` status set. Training partition is untouchable.
- No difficulty-based ordering (would introduce selection bias into the verified test).
- No rejected-samples diagnostic file. Rejects are dropped cleanly.

## 7. Error handling and failure modes

### 7.1 Per-stage failure modes

**Scrape stage:** Network flakes, rate-limit responses, malformed pages, PDFs with corrupt metadata. Handled by retry decorators, blank-render detection, per-page timeouts.

**Label stage:** Transient HTTP errors, 429 rate limits, suspiciously empty provider responses. Handled by tenacity retries, worker-count caps below rate limit, empty-response counter with 5% warning threshold.

**Filter stage:** Silent text-disjoint bugs (caught by §5.4 audits), out-of-bounds bounding boxes (clamped, else `filtered_invalid_bbox`).

**QA stage:** Human error (`b` shortcut), session loss (DB is source of truth), tab crash (per-click commits).

### 7.2 Global invariants

1. Work DB writes are transactional; no halfway-commit state survives a crash.
2. Flat files under `normalized/` are written atomically via temp-file + `os.replace`.
3. Manifests are append-only within a run, regenerated between runs. The last successful run can always be reconstructed from the DB.

## 8. Testing strategy

### 8.1 Unit-level

Fixtures under `scripts/data/tests/` (new directory). Each test sets up a temp SQLite DB with hand-crafted rows covering the §5.5 edge cases, runs one pure function, asserts on the resulting statuses. No mocking frameworks.

Coverage targets:
- NFC normalization and comparison.
- Confidence threshold filtering.
- OOV filtering against `vocab.json`.
- Bounding-box clamping.
- The full §5.3 SQL under each edge case.

### 8.2 Fake provider for labeler

The labeler's `label_page` interface is abstract. The test double returns canned responses from a JSON fixture, enabling tests for retry logic, rate-limit handling, and empty-response handling without touching the network.

### 8.3 End-to-end smoke test

`scripts/data/tests/test_pipeline_smoke.py` runs the full pipeline against 3 local fixture pages and the fake provider, asserts `normalized/train_labels.txt` contains the expected rows. Runs in under 5 seconds, included in CI.

### 8.4 Manual real-API integration check

`bash scripts/data/run_auto_labeled_pipeline.sh --source wikipedia --limit 5` pulls 5 real pages through real Azure. **Not part of CI.** Run manually before full 200K ingests.

### 8.5 Deliberately untested

- Azure Read's OCR correctness. Teacher's problem, not pipeline's contract.
- The actual CER improvement from using the pipeline. That's the v9 training run's job, not this pipeline's tests.
- Cross-browser rendering of Playwright. Playwright version is pinned; empty-response counter catches regressions.

## 9. Integration with existing pipeline

### 9.1 Training integration

The pipeline produces `data/v8/stages/stage_auto_labeled_printed/` with the same shape as existing v8 stages. The v9 curriculum gains one entry pointing at that stage, inserted **between Stage 1 (base synthetic pretraining) and Stage 2 (Mozhi printed adaptation)**. Rationale: Wikipedia + gov PDFs + archive.org span a wider quality band than Mozhi-Hindi's 600 DPI scans, so it's a gentler broadening step before targeted Mozhi adaptation, preserving the v8 plan's "hardest real source last" principle.

### 9.2 Benchmark integration

`verified_test_labels.txt` is wired into `model/benchmark_eval.py` as a new suite named `auto_labeled_verified_printed`. It is a **fourth** benchmark alongside `synthetic_only`, `synthetic_morphed_plus_real`, and `real_only`, not a replacement. Per-source breakdown is reported so Wikipedia / Wikisource / gov_pdf / archive_org CERs are individually visible in the leaderboard.

### 9.3 Manifest

The pipeline's `manifests/manifest.json` inherits v8's manifest spec (dataset name, source, modality, language, license, splits, normalization rules, etc.) and adds:

- Teacher provider and model version.
- Confidence threshold used.
- Four text-disjoint audit counts (all must be zero — in the manifest precisely so a reviewer can verify enforcement ran).
- Path to the Levenshtein-audit sample file.
- SHA256 of `verified_test_labels.txt`.

Manifest is committed to git. Crops and label files go through DVC alongside `data/external/mozhi_hindi` etc.

## 10. Success criteria

The pipeline ships when **all six** are simultaneously true:

1. **Unit tests pass.** Every test under `scripts/data/tests/` runs green in CI — text-disjoint enforcement, NFC normalization, bbox clamp, fake-provider labeler.
2. **Smoke test produces well-formed output.** The manual `--source wikipedia --limit 5` run with real Azure credentials produces ≥1 non-empty row in `normalized/train_labels.txt` and a manifest with zero audit violations.
3. **Full-scale run hits target counts.** ≥150,000 kept training rows and exactly 2,000 QA candidates, with stratified-equal source distribution (≈500 per tier, tolerance ±25).
4. **Human QA completed.** All 2,000 candidates have a decision recorded; `verified_test_labels.txt` exists with SHA256 in the manifest.
5. **Text-disjoint audits clean.** All four §5.4 counts exactly zero; Levenshtein-1 audit spot-checked by human, no near-duplicates flagged.
6. **Integration wired.** New stage appears in v9 curriculum config, new benchmark appears in `model/benchmark_eval.py`, `bash scripts/run_benchmarks.sh` produces an `auto_labeled_verified_printed` row without crashing.

Criteria 1, 2, 5, 6 are automatable. Criteria 3, 4 are human-gated. The pipeline is not shipped until all six are green.

**Explicitly not required:** a specific CER improvement from the student. That's evaluated after training, separately.

## 11. Rollback

Rollback is mechanically simple because the pipeline lives in isolated locations:

1. Delete `data/external/auto_labeled/` and `data/v8/stages/stage_auto_labeled_printed/`.
2. Remove the stage entry from the v9 training config.
3. Remove the `auto_labeled_verified_printed` suite from `model/benchmark_eval.py`.
4. Revert the small edits in `scripts/data/run_auto_labeled_pipeline.sh` and the shell orchestrators.

No v1–v8 data, no v9 model code, no existing benchmark suites are touched. Rollback takes ≈10 minutes of manual git work.

The only genuinely hard rollback is a *training run* that already consumed the pipeline's data — that checkpoint permanently reflects the pipeline's labels. The remedy is "train a new checkpoint without the stage", which is the standard story for any training-data decision.

## 12. Decision log

Decisions locked during the 2026-04-16 brainstorming session, in chronological order:

| # | Decision | Alternative considered | Reason |
|---|---|---|---|
| 1 | Test set: pseudo-label candidates, **human verify** | Train-only auto-labeling; noisy-teacher benchmark | User preference for verification over automation |
| 2 | Teacher: **cloud OCR API** (Azure Read default) | Multimodal LLM; open-source local OCR; ensemble | Best cost/quality/box-output tradeoff for printed Devanagari |
| 3 | Granularity: **word-level** | Line, page | Matches existing vocab, dataloader, benchmark builder |
| 4 | Sources: **Tier 1 + Tier 4** (Wiki + Wikisource + gov PDFs + archive.org) | +Tier 2 news sites | License cleanliness + typeface diversity without scraping complexity |
| 5 | Scale: **Medium** (150–200K train, 2,000 test) | Small (50K/500); Large (500K/5,000) | Statistical power without QA burnout |
| 6 | Test allocation: **stratified equal** (500/source) | Proportional; difficulty-stratified | Per-slice diagnostic visibility |
| 7 | Disjointness: **Policy B** (symmetric P_test vs. P_train + E) | Policy A (one-way); Policy C (all-vs-all) | Closes v9 leakage bug class without over-pruning training |
| 8 | Orchestration: **Approach 2** (staged scripts + SQLite work queue) | Approach 1 (flat files only); Approach 3 (full DVC) | Resumability + SQL dedup + matches `scripts/db/` convention |
| 9 | QA rejects: **dropped from test, no diagnostic file** | Kept as negative probes | Cleaner, forcing function against later contamination |
| 10 | Real-API smoke test: **manual only** | CI-gated with `--real-api` flag | Avoids Azure flakes polluting CI health |
| 11 | Stage placement in curriculum: **between Stage 1 synthetic and Stage 2 Mozhi** | Alongside Mozhi; after Mozhi | Preserves "hardest real source last" v8 principle |

## 13. Open items (for the implementation plan, not this spec)

- Final provider choice (Azure vs. Google vs. AWS) depends on which credentials are available. Plan should include a credential-audit step early.
- Exact seed URL list for Tier-3 gov PDFs. Needs research; plan should allocate time for this.
- Exact Wikipedia article-selection strategy (random, featured-only, some category filter). Plan should prototype at `--limit 10` and inspect before committing.
- Whether the labeler's manifest fields need schema validation (e.g., `pydantic`). Plan should decide based on what's already in the repo.
