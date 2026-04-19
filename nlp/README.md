# NLP — Jnandeep's module

Language-model components for post-OCR correction: the v7 spell-beam reranker and the infrastructure to grow it into a KenLM-backed beam decoder for v9 and beyond.

## Modules

- `corpus_builder.py` — scrapes / ingests Hindi text corpora.
- `dictionaries/` — curated wordlists.
- `vocab.py` — char-level vocab utilities aligned with `data/combined/vocab.json`.
- `ngram_lm.py` / `train_lm.py` — n-gram LM training.
- `spell_checker.py` — edit-distance / dictionary-based correction used by v7.
- `phonetic_rules.py`, `yuktakshar.py` — Devanagari-specific transforms (phonetic equivalence, conjunct handling).
- `postprocessor.py` — composes the above into a single rerank pass over OCR output.
- `evaluate_nlp.py` — CER/WER scorer for end-to-end OCR+NLP evaluation.

## v9 outlook

`model/ctc_decoder.py::decode_prefix_beam` is ready to take a language-model prior at the beam step. The planned integration is a KenLM ARPA trained on Hindi Wikipedia (now scraped as part of the auto-labeled pipeline) combined with this module's dictionary checks. Expected WER impact on the verified benchmark: a further 4–8 pp beyond the current 31.62%. Wiring deferred; see v9 design doc.
