from __future__ import annotations

import json
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class WordSample:
    image_path: str
    text: str
    split: str
    source_id: str


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def parse_tab_gt(
    gt_path: str | Path,
    images_dir: str | Path,
    split: str,
    source_id: str,
) -> tuple[list[WordSample], dict]:
    labels_path = Path(gt_path)
    images_root = Path(images_dir)

    samples: list[WordSample] = []
    skipped_missing = 0
    skipped_empty = 0
    malformed = 0

    with open(labels_path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            if "\t" in line:
                filename, text = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    malformed += 1
                    continue
                filename, text = parts

            filename = filename.strip()
            text = normalize_text(text)
            if not filename or not text:
                skipped_empty += 1
                continue

            image_path = _resolve_image_path(filename, images_root=images_root, labels_path=labels_path)
            if image_path is None:
                skipped_missing += 1
                continue

            samples.append(
                WordSample(
                    image_path=str(image_path.resolve()),
                    text=text,
                    split=split,
                    source_id=source_id,
                )
            )

    report = {
        "samples_loaded": len(samples),
        "skipped_missing_files": skipped_missing,
        "skipped_empty_labels": skipped_empty,
        "skipped_malformed_lines": malformed,
    }
    return samples, report


def _resolve_image_path(filename: str, images_root: Path, labels_path: Path) -> Path | None:
    raw = Path(filename)
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(images_root / raw)
        candidates.append(images_root.parent / raw)
        candidates.append(labels_path.parent / raw)

        if raw.parts and raw.parts[0].lower() == "images":
            stripped = Path(*raw.parts[1:]) if len(raw.parts) > 1 else Path(raw.name)
            candidates.append(images_root / stripped)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_pipe_labels(
    labels_file: str | Path,
    split: str,
    source_id: str,
) -> tuple[list[WordSample], dict]:
    labels_path = Path(labels_file)

    samples: list[WordSample] = []
    skipped_missing = 0
    skipped_empty = 0
    malformed = 0

    with open(labels_path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if "|" not in line:
                malformed += 1
                continue

            image_path_raw, text = line.split("|", 1)
            image_path = Path(image_path_raw.strip()).resolve()
            text = normalize_text(text)

            if not image_path_raw.strip() or not text:
                skipped_empty += 1
                continue
            if not image_path.exists():
                skipped_missing += 1
                continue

            samples.append(
                WordSample(
                    image_path=str(image_path),
                    text=text,
                    split=split,
                    source_id=source_id,
                )
            )

    report = {
        "samples_loaded": len(samples),
        "skipped_missing_files": skipped_missing,
        "skipped_empty_labels": skipped_empty,
        "skipped_malformed_lines": malformed,
    }
    return samples, report


def summarize_samples(samples: list[WordSample], inspect_limit: int = 1000) -> dict:
    text_counter = Counter(sample.text for sample in samples)
    char_counter = Counter()
    widths: list[int] = []
    heights: list[int] = []
    corrupt_images = 0

    for sample in samples:
        char_counter.update(sample.text)

    inspected = 0
    for sample in samples:
        if inspected >= inspect_limit:
            break
        try:
            with Image.open(sample.image_path) as image:
                widths.append(image.width)
                heights.append(image.height)
                inspected += 1
        except Exception:
            corrupt_images += 1

    summary = {
        "sample_count": len(samples),
        "unique_text_count": len(text_counter),
        "character_frequency": dict(sorted(char_counter.items(), key=lambda item: item[0])),
        "inspected_images": inspected,
        "corrupt_images_within_inspection_window": corrupt_images,
        "width_stats": _summarize_numeric(widths),
        "height_stats": _summarize_numeric(heights),
    }
    return summary


def _summarize_numeric(values: list[int]) -> dict:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(values),
        "median": int(statistics.median(values)),
        "max": max(values),
    }


def write_normalized_dataset(
    dataset_root: str | Path,
    dataset_name: str,
    dataset_slug: str,
    language: str,
    modality: str,
    source_url: str,
    split_samples: dict[str, list[WordSample]],
    source_inputs: dict,
    import_reports: dict[str, dict],
    inspect_limit: int = 1000,
    notes: list[str] | None = None,
) -> Path:
    root = Path(dataset_root)
    normalized_dir = root / "normalized"
    manifests_dir = root / "manifests"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    split_summaries = {}
    for split_name, samples in split_samples.items():
        labels_path = normalized_dir / f"{split_name}_labels.txt"
        with open(labels_path, "w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(f"{sample.image_path}|{sample.text}\n")

        split_summaries[split_name] = {
            "labels_file": str(labels_path.resolve()),
            **summarize_samples(samples, inspect_limit=inspect_limit),
            "import_report": import_reports.get(split_name, {}),
        }

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "dataset_name": dataset_name,
        "dataset_slug": dataset_slug,
        "language": language,
        "modality": modality,
        "source_url": source_url,
        "label_format": "absolute_path|ground_truth_text",
        "normalization_rules": {
            "text_unicode_normalization": "NFC",
            "text_trim": True,
            "images_preserved_in_place": True,
            "image_inspection_limit_per_split": inspect_limit,
        },
        "source_inputs": source_inputs,
        "split_summaries": split_summaries,
        "notes": notes or [],
    }

    manifest_path = manifests_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return manifest_path


def merge_label_files(
    label_files: list[str | Path],
    output_file: str | Path,
    source_name: str,
) -> tuple[int, dict]:
    seen: set[tuple[str, str]] = set()
    kept = 0
    duplicates = 0
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_counts: dict[str, int] = {}
    with open(output_path, "w", encoding="utf-8") as out_handle:
        for raw_path in label_files:
            labels_path = Path(raw_path)
            source_counts[str(labels_path.resolve())] = 0

            with open(labels_path, encoding="utf-8") as in_handle:
                for raw_line in in_handle:
                    line = raw_line.strip()
                    if "|" not in line:
                        continue
                    image_path, text = line.split("|", 1)
                    key = (str(Path(image_path).resolve()), normalize_text(text))
                    if key in seen:
                        duplicates += 1
                        continue
                    seen.add(key)
                    source_counts[str(labels_path.resolve())] += 1
                    kept += 1
                    out_handle.write(f"{key[0]}|{key[1]}\n")

    summary = {
        "stage_source": source_name,
        "merged_count": kept,
        "duplicates_removed": duplicates,
        "source_counts": source_counts,
        "output_file": str(output_path.resolve()),
    }
    return kept, summary
