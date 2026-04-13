from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class LabelRecord:
    image_path: str
    text: str
    bucket: str
    label_file: str


def _load_records(
    label_files: list[str] | list[Path],
    bucket: str,
    seen: set[tuple[str, str]] | None = None,
) -> tuple[list[LabelRecord], list[str]]:
    records: list[LabelRecord] = []
    missing: list[str] = []
    local_seen = seen if seen is not None else set()

    for raw_path in label_files:
        label_path = Path(raw_path)
        if not label_path.exists():
            missing.append(str(label_path))
            continue

        with open(label_path, encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if "|" not in line:
                    continue
                image_path, text = line.split("|", 1)
                image_path = image_path.strip()
                text = text.strip()
                if not image_path or not text:
                    continue
                key = (image_path, text)
                if key in local_seen:
                    continue
                local_seen.add(key)
                records.append(
                    LabelRecord(
                        image_path=image_path,
                        text=text,
                        bucket=bucket,
                        label_file=str(label_path.resolve()),
                    )
                )

    return records, missing


def _sample_records(
    records: list[LabelRecord],
    target_size: int | None,
    rng: random.Random,
) -> list[LabelRecord]:
    if not records:
        return []

    shuffled = list(records)
    rng.shuffle(shuffled)
    if target_size is None:
        return shuffled
    return shuffled[: min(target_size, len(shuffled))]


def _write_labels(path: Path, records: list[LabelRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f"{record.image_path}|{record.text}\n")


def _bucket_counts(records: list[LabelRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.bucket] = counts.get(record.bucket, 0) + 1
    return counts


def _origin_counts(records: list[LabelRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.label_file] = counts.get(record.label_file, 0) + 1
    return counts


def _build_suite_metadata(name: str, labels_file: Path, records: list[LabelRecord]) -> dict:
    return {
        "name": name,
        "labels_file": str(labels_file.resolve()),
        "sample_count": len(records),
        "composition": _bucket_counts(records),
        "source_files": _origin_counts(records),
    }


def _compose_real_side(
    real_records: list[LabelRecord],
    iiit_records: list[LabelRecord],
    target_size: int | None,
    rng: random.Random,
) -> list[LabelRecord]:
    chosen_real = _sample_records(real_records, target_size, rng)
    if target_size is None:
        chosen_iiit = _sample_records(iiit_records, None, rng)
        return chosen_real + chosen_iiit

    deficit = max(0, target_size - len(chosen_real))
    if deficit == 0:
        return chosen_real
    chosen_iiit = _sample_records(iiit_records, deficit, rng)
    return chosen_real + chosen_iiit


def build_benchmark_suites(
    output_dir: str | Path,
    synthetic_labels: list[str] | list[Path] | None = None,
    morphed_labels: list[str] | list[Path] | None = None,
    real_labels: list[str] | list[Path] | None = None,
    iiit_train_labels: str | Path | None = None,
    seed: int = 42,
    synthetic_size: int | None = None,
    mixed_size: int | None = None,
    real_size: int | None = None,
    iiit_holdout_ratio: float = 0.15,
    iiit_holdout_count: int | None = None,
) -> dict:
    rng = random.Random(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    synthetic_labels = list(synthetic_labels or [])
    morphed_labels = list(morphed_labels or [])
    real_labels = list(real_labels or [])

    shared_seen: set[tuple[str, str]] = set()
    synthetic_pool, missing_synthetic = _load_records(synthetic_labels, "synthetic", shared_seen)
    morphed_pool, missing_morphed = _load_records(morphed_labels, "morphed", shared_seen)
    real_pool, missing_real = _load_records(real_labels, "real", shared_seen)

    warnings: list[str] = []
    for missing in (missing_synthetic + missing_morphed + missing_real):
        warnings.append(f"Missing input labels file: {missing}")

    max_real_needed = 0
    if real_size:
        max_real_needed = max(max_real_needed, real_size)
    if mixed_size:
        max_real_needed = max(max_real_needed, math.ceil(mixed_size / 2))
    elif morphed_pool:
        max_real_needed = max(max_real_needed, len(morphed_pool))

    iiit_metadata = None
    iiit_holdout_pool: list[LabelRecord] = []
    effective_real_pool = list(real_pool)

    if iiit_train_labels:
        iiit_records, missing_iiit = _load_records([iiit_train_labels], "iiit_holdout")
        for missing in missing_iiit:
            warnings.append(f"Missing IIIT labels file: {missing}")

        if iiit_records:
            requested_holdout = iiit_holdout_count
            if requested_holdout is None:
                deficit = max(0, max_real_needed - len(real_pool))
                ratio_count = max(1, math.ceil(len(iiit_records) * iiit_holdout_ratio))
                requested_holdout = max(ratio_count, deficit)

            requested_holdout = min(requested_holdout, len(iiit_records))
            shuffled_iiit = list(iiit_records)
            rng.shuffle(shuffled_iiit)
            iiit_holdout = shuffled_iiit[:requested_holdout]
            iiit_retrain = shuffled_iiit[requested_holdout:]
            iiit_holdout_pool = iiit_holdout

            iiit_dir = output_path / "iiit_holdout"
            holdout_labels_path = iiit_dir / "holdout_labels.txt"
            retrain_labels_path = iiit_dir / "retrain_labels.txt"
            _write_labels(holdout_labels_path, iiit_holdout)
            _write_labels(retrain_labels_path, iiit_retrain)

            if len(real_pool) < max_real_needed:
                effective_real_pool.extend(iiit_holdout)
                warnings.append(
                    "Real-photo pool was too small, so the IIIT holdout was appended to the real benchmark pool."
                )

            iiit_metadata = {
                "source_labels": str(Path(iiit_train_labels).resolve()),
                "holdout_labels": str(holdout_labels_path.resolve()),
                "retrain_labels": str(retrain_labels_path.resolve()),
                "holdout_count": len(iiit_holdout),
                "retrain_count": len(iiit_retrain),
                "holdout_ratio": iiit_holdout_ratio,
            }

    benchmarks: list[dict] = []

    synthetic_only = _sample_records(synthetic_pool, synthetic_size, rng)
    if synthetic_only:
        suite_dir = output_path / "synthetic_only"
        labels_path = suite_dir / "labels.txt"
        _write_labels(labels_path, synthetic_only)
        benchmarks.append(_build_suite_metadata("synthetic_only", labels_path, synthetic_only))
    else:
        warnings.append("Skipped synthetic_only benchmark because no synthetic labels were supplied.")

    mixed_per_bucket: int | None = None
    if morphed_pool and effective_real_pool:
        if mixed_size is not None:
            mixed_per_bucket = max(1, mixed_size // 2)
        else:
            mixed_per_bucket = min(len(morphed_pool), len(effective_real_pool))

    mixed_records: list[LabelRecord] = []
    if mixed_per_bucket:
        mixed_records.extend(_sample_records(morphed_pool, mixed_per_bucket, rng))
        mixed_records.extend(
            _compose_real_side(real_pool, iiit_holdout_pool, mixed_per_bucket, rng)
        )
        rng.shuffle(mixed_records)
        suite_dir = output_path / "synthetic_morphed_plus_real"
        labels_path = suite_dir / "labels.txt"
        _write_labels(labels_path, mixed_records)
        benchmarks.append(
            _build_suite_metadata("synthetic_morphed_plus_real", labels_path, mixed_records)
        )
    else:
        warnings.append(
            "Skipped synthetic_morphed_plus_real benchmark because both morphed and real pools are required."
        )

    real_only = _compose_real_side(real_pool, iiit_holdout_pool, real_size, rng)
    if real_only:
        suite_dir = output_path / "real_only"
        labels_path = suite_dir / "labels.txt"
        _write_labels(labels_path, real_only)
        benchmarks.append(_build_suite_metadata("real_only", labels_path, real_only))
    else:
        warnings.append("Skipped real_only benchmark because no real-photo labels were supplied.")

    if not benchmarks:
        raise ValueError("No benchmark suites were created. Provide at least one valid labels file.")

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "seed": seed,
        "inputs": {
            "synthetic_labels": [str(Path(p).resolve()) for p in synthetic_labels],
            "morphed_labels": [str(Path(p).resolve()) for p in morphed_labels],
            "real_labels": [str(Path(p).resolve()) for p in real_labels],
            "iiit_train_labels": str(Path(iiit_train_labels).resolve()) if iiit_train_labels else None,
        },
        "source_counts": {
            "synthetic": len(synthetic_pool),
            "morphed": len(morphed_pool),
            "real": len(real_pool),
            "effective_real": len(effective_real_pool),
        },
        "iiit_holdout": iiit_metadata,
        "benchmarks": benchmarks,
        "warnings": warnings,
    }

    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return manifest
