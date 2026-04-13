from pathlib import Path

from PIL import Image

from data.v8_dataset_prep import merge_label_files, normalize_text, parse_tab_gt, write_normalized_dataset


def _make_image(path: Path, size=(32, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, 255).save(path)


def test_parse_tab_gt_normalizes_text_and_skips_missing(tmp_path):
    images_dir = tmp_path / "images"
    gt_file = tmp_path / "train_gt.txt"

    _make_image(images_dir / "one.png")
    _make_image(images_dir / "two.png")

    gt_file.write_text(
        "one.png\t नमस्ते \n"
        "two.png\tकिताब\n"
        "missing.png\tगायब\n",
        encoding="utf-8",
    )

    samples, report = parse_tab_gt(gt_file, images_dir, split="train", source_id="mozhi_hindi")

    assert [sample.text for sample in samples] == ["नमस्ते", "किताब"]
    assert report["samples_loaded"] == 2
    assert report["skipped_missing_files"] == 1


def test_write_normalized_dataset_creates_labels_and_manifest(tmp_path):
    image_path = tmp_path / "raw" / "sample.png"
    _make_image(image_path, size=(40, 20))

    manifest_path = write_normalized_dataset(
        dataset_root=tmp_path / "external" / "mozhi_hindi",
        dataset_name="Mozhi-Hindi",
        dataset_slug="mozhi_hindi",
        language="hi",
        modality="printed",
        source_url="https://example.com/mozhi",
        split_samples={
            "train": [
                type("Sample", (), {
                    "image_path": str(image_path.resolve()),
                    "text": "नमस्ते",
                    "split": "train",
                    "source_id": "mozhi_hindi",
                })()
            ]
        },
        source_inputs={"train": {"gt_file": "/tmp/train_gt.txt", "images_dir": "/tmp/images"}},
        import_reports={"train": {"samples_loaded": 1}},
        inspect_limit=10,
    )

    labels_file = manifest_path.parent.parent / "normalized" / "train_labels.txt"
    assert labels_file.exists()
    assert "sample.png|नमस्ते" in labels_file.read_text(encoding="utf-8")
    assert manifest_path.exists()


def test_merge_label_files_deduplicates_exact_pairs(tmp_path):
    img1 = tmp_path / "a.png"
    img2 = tmp_path / "b.png"
    _make_image(img1)
    _make_image(img2)

    labels_a = tmp_path / "a_labels.txt"
    labels_b = tmp_path / "b_labels.txt"
    labels_a.write_text(f"{img1.resolve()}|नमस्ते\n{img2.resolve()}|किताब\n", encoding="utf-8")
    labels_b.write_text(f"{img1.resolve()}|नमस्ते\n", encoding="utf-8")

    count, summary = merge_label_files(
        [labels_a, labels_b],
        output_file=tmp_path / "merged.txt",
        source_name="stage1",
    )

    assert count == 2
    assert summary["duplicates_removed"] == 1
    assert normalize_text(" नमस्ते ") == "नमस्ते"
