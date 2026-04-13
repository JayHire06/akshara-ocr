# V8 Data Preparation Plan

This document defines the data strategy for `v8`. It is intentionally explicit about **what to add**, **what to keep out of training**, and **why**. The goal is to improve real-world Hindi OCR quality without repeating the main weakness of the earlier pipeline: strong synthetic results with weak evidence on independent real data.

## 1. Objectives

`v8` should improve on three fronts at the same time:

1. **Real printed documents**
   Reason: current synthetic-heavy training can hide failures on scanned books, printed forms, and phone captures of clean printed text.
2. **Real handwritten / camera-captured words**
   Reason: this is the highest-variance regime and the hardest to simulate well with synthetic augmentation alone.
3. **Evaluation credibility**
   Reason: a model is only meaningfully better if it improves on fixed holdout suites that were never used during training.

## 1.1 Implemented Utilities

The repo now includes the first `v8` data-preparation entrypoints:

1. `scripts/data/import_nltm_word_dataset.py`
   Reason: normalizes IIIT / NLTM-style word datasets whose labels are stored as `filename<TAB>text`.
2. `scripts/data/import_pipe_word_dataset.py`
   Reason: normalizes any existing dataset that is already in `image_path|text` format, including private data and previously prepared repo datasets.
3. `scripts/data/build_v8_stage_labels.py`
   Reason: merges normalized datasets into a named `v8` training stage without hand-editing label files.
4. `scripts/data/download_v8_datasets.py`
   Reason: downloads the official `Mozhi-Hindi`, `IIIT-INDIC-HW-WORDS-Hindi`, and `IIIT-Synthetic-R-Hindi` archives directly into the normalized `v8` external dataset layout.

The shared implementation lives in:

- `data/v8_dataset_prep.py`

### Example: Import Mozhi-Hindi

```bash
python scripts/data/import_nltm_word_dataset.py \
  --dataset-name "Mozhi-Hindi" \
  --dataset-slug mozhi_hindi \
  --language hi \
  --modality printed \
  --source-url https://ilocr.iiit.ac.in/dataset/7/ \
  --train-images-dir /path/to/mozhi/train/images \
  --train-gt /path/to/mozhi/train_gt.txt \
  --val-images-dir /path/to/mozhi/val/images \
  --val-gt /path/to/mozhi/val_gt.txt \
  --test-images-dir /path/to/mozhi/test/images \
  --test-gt /path/to/mozhi/test_gt.txt
```

### Example: Download and Normalize Official V8 Sources

```bash
python scripts/data/download_v8_datasets.py \
  --datasets mozhi_hindi iiit_hw_words_hindi iiit_synth_r_hindi
```

### Example: Build a V8 Base Stage

```bash
python scripts/data/build_v8_stage_labels.py \
  --stage-name stage1_base \
  --train-labels \
    data/external/iiit_synth_r_hindi/normalized/train_labels.txt \
    data/external/repo_synthetic/normalized/train_labels.txt \
  --val-labels \
    data/external/iiit_synth_r_hindi/normalized/val_labels.txt \
    data/external/repo_synthetic/normalized/val_labels.txt \
  --notes "Base synthetic pretraining pool for v8"
```

### Example: Import Existing Pipe Labels

```bash
python scripts/data/import_pipe_word_dataset.py \
  --dataset-name "Private Real Hindi" \
  --dataset-slug private_real_hindi \
  --language hi \
  --modality phone_captured \
  --source-url "internal collection" \
  --train-labels /path/to/private/train_labels.txt \
  --val-labels /path/to/private/val_labels.txt \
  --test-labels /path/to/private/test_labels.txt
```

These commands write manifests on purpose. The reason is simple: the most common failure in long OCR projects is losing track of exactly which samples and splits were used.

## 2. Core Principles

1. **Never train on benchmark data**
   Official `test` splits stay frozen. If a dataset has `train`, `val`, and `test`, use `test` only for final benchmarking.
2. **Prefer real data over more synthetic data once the script coverage is adequate**
   Synthetic data is excellent for pretraining and vocabulary coverage, but not sufficient evidence of production robustness.
3. **Use synthetic data for scale, real data for alignment**
   Synthetic data should stabilize optimization; real data should determine the final model behavior.
4. **Keep benchmark suites distribution-specific**
   One mixed validation file is not enough. We need separate synthetic-only, mixed, and real-only measurement.
5. **Deduplicate aggressively**
   A near-duplicate between training and benchmark sets makes scores look better than the model really is.

## 3. Recommended Data Additions

These are the recommended additions for `v8`, ordered by impact.

## 3.1 Tier 1: Must Add

### A. IIIT-Indic-HW-UC

- Type: real, handwritten, camera-captured
- Role: main real-photo handwriting source for training and benchmarking
- Source: https://cvit.iiit.ac.in/usodi/ucciohd.php
- Why it matters:
  - It is much closer to phone-photo reality than flat synthetic rendering.
  - The dataset description says it includes **200K word images per language**, with mobile-camera capture conditions and official train/val/test splits of **150K / 20K / 30K** for each language.
  - It introduces blur, exposure variation, perspective distortion, background clutter, low resolution, and shadows that synthetic text usually misses.
- `v8` decision:
  - Use official `train` and `val` for training development.
  - Keep official `test` fully frozen for the `real_only` benchmark.

### B. Mozhi-Hindi

- Type: real, printed, scanned books
- Role: main real printed-document source
- Source: https://ilocr.iiit.ac.in/dataset/7/
- Why it matters:
  - It gives **100,049** cropped Hindi word images from book pages scanned at **600 DPI**.
  - It covers printed document structure better than synthetic rendering and should improve printed-text OCR materially.
  - Official splits are **79,762 / 10,114 / 10,173** train/val/test.
- `v8` decision:
  - Use train + val for printed-domain training.
  - Keep test for benchmark only.

### C. IIIT-Synthetic-R-Hindi

- Type: synthetic, printed
- Role: base pretraining and vocabulary/font coverage
- Source: https://ilocr.iiit.ac.in/dataset/49/
- Why it matters:
  - It provides **726,750** synthetic printed Hindi word images with **1,115 fonts**.
  - It is large enough to pretrain a recognizer before moving into expensive real-domain fine-tuning.
  - Official splits are **508,725 / 72,675 / 145,350** train/val/test.
- `v8` decision:
  - Use train heavily during early training.
  - Use val for synthetic monitoring if needed.
  - Keep official test out of training and reserve it for `synthetic_only` evaluation.

## 3.2 Tier 2: Strongly Recommended

### D. IIIT-INDIC-HW-WORDS-Hindi

- Type: real, handwritten, scanned forms
- Role: clean handwriting supervision before harder camera-captured fine-tuning
- Source: https://ilocr.iiit.ac.in/dataset/18/
- Why it matters:
  - It contains **95,431** handwritten Hindi word images.
  - Official splits are **69,853 / 12,708 / 12,870** train/val/test.
  - It is cleaner than camera-captured handwriting and is useful as an intermediate stage between synthetic pretraining and harder real capture conditions.
- `v8` decision:
  - Use train + val during handwriting training.
  - Keep test frozen for the `real_only` benchmark.

### E. Your Own Real Photos

- Type: real, printed and handwritten, phone-captured
- Role: highest-priority private domain alignment
- Why it matters:
  - Public Hindi OCR datasets still underrepresent the exact failure conditions you care about.
  - A few thousand carefully labeled real crops from your target domain can move the model more than another large synthetic batch.
- `v8` decision:
  - Collect this continuously.
  - Split by **document/page/writer**, not by random crop, to avoid leakage.
  - Use a small frozen subset for the highest-trust benchmark.

## 3.3 Tier 3: Optional, Domain-Specific

### F. IndicSTR12-Hindi

- Type: real, scene text
- Role: hard benchmark and optional fine-tuning supplement for outdoor text
- Source: https://ilocr.iiit.ac.in/dataset/30/
- Why it matters:
  - It is small: **1,083** word images with **812 / 271** train/test.
  - It is not a general document dataset, but it is a good stress test for signs, boards, and naturally captured text.
- `v8` decision:
  - Prefer to use it as a hard benchmark or very light fine-tuning source.
  - Do not let it dominate training.

### G. IIIT-Synthetic-IndicSTR-Hindi

- Type: synthetic, scene text
- Role: scene-text pretraining if outdoor / signboard OCR matters
- Source: https://ilocr.iiit.ac.in/dataset/60/
- Why it matters:
  - It provides **2M** synthetic Hindi scene-text word images with **1.5M / 0.5M / 0.5M** train/val/test splits.
  - It is useful if `v8` is expected to handle signboards, street photos, posters, or storefront text.
- `v8` decision:
  - Use only if scene text is a real target requirement.
  - Otherwise keep `v8` focused on document OCR first.

### H. Dakshina

- Type: text-only
- Role: lexicon building, language-model training, synthetic text sampling
- Source: https://github.com/google-research-datasets/dakshina
- Why it matters:
  - It is not an OCR image dataset.
  - It includes Hindi native-script text and lexicons that are useful for vocabulary growth, synthetic prompt generation, and NLP post-processing.
- `v8` decision:
  - Do not treat it as image training data.
  - Use it for corpus and lexicon support only.

## 4. Recommended V8 Training Mix

`v8` should not be trained from one monolithic pool from the start. A staged curriculum is safer.

### Stage 1: Base Recognition Pretraining

Use:

- `IIIT-Synthetic-R-Hindi` train
- existing repo synthetic data
- existing morphed / realistic synthetic data

Reason:

- This stage teaches script coverage, font variation, character composition, and broad lexical support at scale.

### Stage 2: Real Printed Adaptation

Add:

- `Mozhi-Hindi` train + val

Reason:

- This stage aligns the printed branch to real scanned words and page-derived crops.

### Stage 3: Handwriting Adaptation

Add:

- `IIIT-INDIC-HW-WORDS-Hindi` train + val

Reason:

- This stage introduces natural handwriting without immediately jumping to the harder camera-captured regime.

### Stage 4: Real Camera-Captured Adaptation

Add:

- `IIIT-Indic-HW-UC` train + val
- private phone-photo captures

Reason:

- This is the domain that matters most for robust deployment.
- It should be the last adaptation stage so the final model ends close to target conditions.

### Stage 5: Optional Scene-Text Branch

Add only if required:

- `IIIT-Synthetic-IndicSTR-Hindi`
- optional light fine-tune on `IndicSTR12-Hindi`

Reason:

- Scene text is a different domain. It should not dilute a document-focused model unless the product actually needs it.

## 5. Benchmark Design for V8

The benchmark design in [docs/benchmarking.md](./benchmarking.md) stays valid, but `v8` should populate the suites with stronger external data.

### Benchmark A: `synthetic_only`

Recommended sources:

- held-out repo synthetic benchmark
- `IIIT-Synthetic-R-Hindi` official test

Purpose:

- Measures script and lexical competence without real capture noise.

### Benchmark B: `synthetic_morphed_plus_real`

Recommended composition:

- 50% morphed / realistic synthetic holdout
- 25% `Mozhi-Hindi` test
- 25% `IIIT-Indic-HW-UC` test or private real-photo holdout

Purpose:

- Measures domain transfer from synthetic training to mixed realistic conditions.

### Benchmark C: `real_only`

Recommended composition:

- private phone-photo holdout first
- `Mozhi-Hindi` test for printed
- `IIIT-INDIC-HW-WORDS-Hindi` test for handwriting
- `IIIT-Indic-HW-UC` test for camera-captured handwriting

Optional hard slice:

- `IndicSTR12-Hindi` test

Purpose:

- This is the decision benchmark for claiming `v8` is genuinely better.

## 6. Exact Split Policy

These rules are mandatory.

1. Use official dataset `test` splits only for benchmark reporting.
2. Use official `train` and `val` splits for model development unless the source documentation says otherwise.
3. For private data, split by:
   - writer for handwriting
   - page or document for printed scans
   - capture session or source image for cropped phone photos
4. Never split multiple crops from the same page across train and benchmark.
5. Never mix IIIT fallback holdout samples back into training once chosen for benchmarking.

## 7. Canonical Repo Layout

The repo should treat external datasets as normalized inputs, not as ad hoc folders.

Recommended layout:

```text
data/
  external/
    mozhi_hindi/
      raw/
      manifests/
      normalized/
    iiit_hw_words_hindi/
      raw/
      manifests/
      normalized/
    iiit_hw_uc_hindi/
      raw/
      manifests/
      normalized/
    iiit_synth_r_hindi/
      raw/
      manifests/
      normalized/
    indicstr12_hindi/
      raw/
      manifests/
      normalized/
    private_real_hindi/
      raw/
      manifests/
      normalized/
  benchmarks/
  combined/
```

Reason:

- `raw/` preserves the downloaded source.
- `normalized/` stores repo-ready label files.
- `manifests/` stores provenance, checksums, split counts, and preparation notes.

## 8. Canonical Label Format

All normalized label files should use one format:

```text
/absolute/path/to/image.png|ground_truth_text
```

Why this format:

- It already matches the newer training and benchmark code in this repo.
- It is simple to combine, shuffle, subset, and inspect with shell tools.

Each normalized dataset should produce:

- `train_labels.txt`
- `val_labels.txt`
- `test_labels.txt`
- `manifest.json`

## 9. Required Metadata Per Dataset

Every prepared dataset should include a `manifest.json` with at least:

- dataset name
- source URL
- modality
- language
- license or usage notes if available
- original archive names
- raw file checksums if practical
- normalization date
- split counts
- text normalization rules used
- image normalization rules used
- known exclusions or corrupted files

Reason:

- Six months later, the main failure mode is not model code. It is forgetting what exactly went into training.

## 10. Normalization Rules

These should be consistent across all added sources.

### Text normalization

1. Normalize Unicode to `NFC`.
2. Strip leading and trailing whitespace.
3. Preserve valid Devanagari punctuation and digits.
4. Do not silently remove characters unless the removal rule is documented in the dataset manifest.
5. If a dataset contains labels outside the model vocabulary, record the frequency before deciding whether to keep, map, or drop them.

### Image normalization

1. Preserve original raw images in `raw/`.
2. Do not overwrite downloaded files.
3. Store normalized trainable image paths separately if resizing, binarization, denoising, or cropping is applied.
4. If no image rewrite is needed, normalized labels may point directly to raw extracted files.

## 11. Deduplication Policy

This is important enough to be explicit.

Run deduplication at three levels:

1. **Path duplicate**
   Same image path appears more than once.
2. **Exact content duplicate**
   Same image bytes under different filenames.
3. **Near-duplicate page leak**
   Multiple crops from the same page appear across train and benchmark.

Reason:

- Without this, benchmark quality is not trustworthy.

Minimum action:

- hash all normalized image files
- keep page-level provenance where available
- keep writer/page/session IDs in manifests if the source provides them

## 12. Data Quality Checks

Each dataset preparation step should produce a short QA report:

1. sample count by split
2. unique text count by split
3. character frequency table
4. min / median / max width and height
5. missing file count
6. unreadable or corrupt image count
7. duplicate count removed
8. out-of-vocabulary character count relative to current `vocab.json`

Reason:

- This catches bad imports before a long training run burns GPU time.

## 13. Initial V8 Build Order

The work should happen in this order.

### Step 1

Prepare `IIIT-Synthetic-R-Hindi`.

Reason:

- It is the cleanest high-scale starting point and easiest to wire into the new benchmark flow.

### Step 2

Prepare `Mozhi-Hindi`.

Reason:

- This gives a real printed benchmark and immediately improves the usefulness of `real_only`.

### Step 3

Prepare `IIIT-INDIC-HW-WORDS-Hindi`.

Reason:

- This adds a real handwritten branch before the harder camera-captured data.

### Step 4

Prepare `IIIT-Indic-HW-UC`.

Reason:

- This is the highest-value real-photo dataset, but it may need the most careful import and provenance handling.

### Step 5

Prepare private phone-photo data.

Reason:

- This should eventually become the benchmark that matters most.

### Step 6

Build frozen benchmark suites under `data/benchmarks/`.

Reason:

- Training should not begin until the benchmark holdouts are locked.

## 14. What Should Not Go Into V8 Training Yet

Avoid these mistakes:

1. throwing all real and synthetic sources into one shuffled file immediately
2. using official `test` splits for convenience during training
3. mixing scene-text data into a document model without a product reason
4. over-weighting synthetic data after enough real data is available
5. evaluating on the same source distribution that drove the latest fine-tune

## 15. Concrete Success Criteria

The `v8` data pipeline is ready when:

1. all Tier 1 datasets have normalized manifests and split files
2. at least one frozen `real_only` benchmark exists from external data
3. private phone-photo holdout exists, even if small
4. benchmark builder manifests clearly show which files are synthetic, real, and fallback
5. training configs can point to staged label files without manual editing

## 16. Immediate Implementation Checklist

These are the next repo tasks implied by this plan.

1. Add dataset import scripts for:
   - `Mozhi-Hindi`
   - `IIIT-INDIC-HW-WORDS-Hindi`
   - `IIIT-Indic-HW-UC`
   - `IIIT-Synthetic-R-Hindi`
2. Add normalized manifest generation for every import script.
3. Add a dedup / QA summary script for normalized label files.
4. Extend the benchmark builder inputs to accept official dataset test splits directly.
5. Create a `v8` training config that points to staged label pools instead of one monolithic combined file.

## 17. References

- `IIIT-Indic-HW-UC`: https://cvit.iiit.ac.in/usodi/ucciohd.php
- `Mozhi-Hindi`: https://ilocr.iiit.ac.in/dataset/7/
- `IIIT-INDIC-HW-WORDS-Hindi`: https://ilocr.iiit.ac.in/dataset/18/
- `IIIT-Synthetic-R-Hindi`: https://ilocr.iiit.ac.in/dataset/49/
- `IndicSTR12-Hindi`: https://ilocr.iiit.ac.in/dataset/30/
- `IIIT-Synthetic-IndicSTR-Hindi`: https://ilocr.iiit.ac.in/dataset/60/
- `Dakshina`: https://github.com/google-research-datasets/dakshina

## 18. Summary

If only three external additions are made for `v8`, they should be:

1. `IIIT-Indic-HW-UC`
2. `Mozhi-Hindi`
3. `IIIT-Synthetic-R-Hindi`

Reason:

- Together they cover the most important missing triangle:
  - real camera-captured handwriting
  - real printed documents
  - high-scale synthetic pretraining

That is the most defensible `v8` data foundation for both training and benchmarking.
