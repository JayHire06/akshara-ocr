from pathlib import Path

from data.benchmark_suites import build_benchmark_suites


def _write_labels(path: Path, prefix: str, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for idx in range(count):
            image_path = path.parent / f"{prefix}_{idx}.png"
            image_path.write_bytes(b"fake")
            handle.write(f"{image_path}|{prefix}_{idx}\n")


def test_build_benchmark_suites_uses_iiit_holdout_when_real_pool_is_small(tmp_path):
    synthetic_labels = tmp_path / "synthetic" / "labels.txt"
    morphed_labels = tmp_path / "morphed" / "labels.txt"
    real_labels = tmp_path / "real" / "labels.txt"
    iiit_labels = tmp_path / "iiit" / "labels.txt"

    _write_labels(synthetic_labels, "synthetic", 6)
    _write_labels(morphed_labels, "morphed", 8)
    _write_labels(real_labels, "real", 1)
    _write_labels(iiit_labels, "iiit", 10)

    manifest = build_benchmark_suites(
        output_dir=tmp_path / "benchmarks",
        synthetic_labels=[synthetic_labels],
        morphed_labels=[morphed_labels],
        real_labels=[real_labels],
        iiit_train_labels=iiit_labels,
        mixed_size=6,
        real_size=4,
        iiit_holdout_count=5,
        seed=7,
    )

    benchmark_map = {item["name"]: item for item in manifest["benchmarks"]}

    assert benchmark_map["synthetic_only"]["sample_count"] == 6
    assert benchmark_map["synthetic_morphed_plus_real"]["composition"]["morphed"] == 3
    assert benchmark_map["synthetic_morphed_plus_real"]["composition"]["iiit_holdout"] == 2
    assert benchmark_map["synthetic_morphed_plus_real"]["composition"]["real"] == 1
    assert benchmark_map["real_only"]["sample_count"] == 4

    iiit_meta = manifest["iiit_holdout"]
    assert iiit_meta["holdout_count"] == 5
    assert iiit_meta["retrain_count"] == 5
