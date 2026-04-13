from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model.crnn import CRNN, CRNNv6
from model.crnn_v9 import CRNNv9
from model.ctc_decoder import CTCDecoder

import unicodedata


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFC", text).strip()


def levenshtein_distance(s1, s2) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        current = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                current.append(distances[i1])
            else:
                current.append(1 + min(distances[i1], distances[i1 + 1], current[-1]))
        distances = current
    return distances[-1]


def calculate_cer(prediction: str, target: str) -> float:
    prediction = normalize_text(prediction)
    target = normalize_text(target)
    if not target:
        return 1.0 if prediction else 0.0
    return min(1.0, levenshtein_distance(prediction, target) / len(target))


def calculate_wer(prediction: str, target: str) -> float:
    prediction_words = normalize_text(prediction).split()
    target_words = normalize_text(target).split()
    if not target_words:
        return 1.0 if prediction_words else 0.0
    return min(1.0, levenshtein_distance(prediction_words, target_words) / len(target_words))


def _load_vocab_list(root_dir: Path, vocab_size: int) -> list[str]:
    possible_vocabs = [
        root_dir / "data" / "combined" / "vocab.json",
        root_dir / "vocab.json",
    ]
    vocab_list = ["<PAD>"] + [chr(i) for i in range(33, 33 + vocab_size - 1)]

    for vocab_path in possible_vocabs:
        if not vocab_path.exists():
            continue
        with open(vocab_path, encoding="utf-8") as handle:
            vocab_dict = json.load(handle)
        vocab_list = ["<UNK>"] * vocab_size
        for char, idx in vocab_dict.items():
            if idx >= vocab_size:
                continue
            vocab_list[idx] = "<PAD>" if char == "<blank>" else char
        break

    return vocab_list


class CheckpointLoadError(RuntimeError):
    """Raised when a checkpoint's state dict is incompatible with current code."""


def _load_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]

    fc_bias_keys = [
        key
        for key in state_dict.keys()
        if ".fc.bias" in key
        or ".fc.2.bias" in key
        or "linear.bias" in key
        or key == "fc.bias"
        or key == "head.bias"
    ]
    if not fc_bias_keys:
        raise CheckpointLoadError(
            f"{checkpoint_path}: cannot infer vocab size — no fc/linear/head output bias found."
        )

    vocab_size = state_dict[fc_bias_keys[-1]].shape[0]
    has_stn = any("stn." in key for key in state_dict.keys())
    has_depthwise = any("depthwise" in key for key in state_dict.keys())
    has_block_cnn = any(key.startswith("cnn.block") for key in state_dict.keys())
    has_transformer = any("transformer.layers" in key for key in state_dict.keys())
    has_sequential_cnn = (
        any(key.startswith("cnn.0.0") for key in state_dict.keys()) and not has_depthwise
    )

    if has_sequential_cnn:
        raise CheckpointLoadError(
            f"{checkpoint_path}: legacy Sequential CRNN format detected "
            f"(keys like cnn.0.0.* + fc.*). Current model.crnn.CRNN uses named "
            f"blocks (cnn.block1.* + linear.*). This checkpoint is architecturally "
            f"orphaned and must be retrained under the new architecture."
        )

    config = {
        "in_channels": 1,
        "lstm_input_size": 512,
        "lstm_hidden_size": 256,
        "lstm_num_layers": 2,
        "lstm_dropout": 0.0,
        "blank_id": 0,
        "use_stn": has_stn,
        "lstm_hidden": 256,
        "lstm_layers": 2,
    }

    if has_transformer:
        model = CRNNv9(vocab_size=vocab_size, config=config).to(device)
        arch = "CRNNv9"
    elif has_stn or has_depthwise:
        model = CRNNv6(vocab_size=vocab_size, config=config).to(device)
        arch = "CRNNv6"
    else:
        if not has_block_cnn:
            raise CheckpointLoadError(
                f"{checkpoint_path}: unrecognised CRNN variant — neither "
                f"cnn.block* nor depthwise keys present."
            )
        model = CRNN(vocab_size=vocab_size, config=config).to(device)
        arch = "CRNN"

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    expected_params = len(model.state_dict())
    loaded_ratio = (expected_params - len(missing)) / expected_params
    if loaded_ratio < 0.90:
        raise CheckpointLoadError(
            f"{checkpoint_path}: only {loaded_ratio * 100:.0f}% of {arch} "
            f"parameters loaded (missing={len(missing)}, unexpected={len(unexpected)}). "
            f"First missing: {missing[:3]!r}. First unexpected: {unexpected[:3]!r}. "
            f"Refusing to report metrics on a random-init model."
        )
    if missing or unexpected:
        # partial but above threshold: surface a warning, still load
        print(
            f"  [warn] {arch} load partial: missing={len(missing)} "
            f"unexpected={len(unexpected)} (above 90% threshold, continuing)"
        )

    model.eval()
    return model, vocab_size


def evaluate_checkpoint_on_labels(
    checkpoint_path: str | Path,
    labels_file: str | Path,
    device: torch.device,
    root_dir: str | Path | None = None,
    preview_limit: int = 0,
) -> dict:
    labels_path = Path(labels_file)
    project_root = Path(root_dir) if root_dir else Path(__file__).resolve().parent.parent

    model, vocab_size = _load_model(checkpoint_path, device)
    vocab_list = _load_vocab_list(project_root, vocab_size)
    decoder = CTCDecoder(vocab=vocab_list, blank_id=0)

    total_cer = 0.0
    total_wer = 0.0
    total_samples = 0
    empty_predictions = 0
    previews: list[dict[str, str]] = []

    with open(labels_path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if "|" not in line:
                continue
            image_path, expected_text = line.split("|", 1)
            image_file = Path(image_path)
            if not image_file.exists():
                continue

            pil_img = Image.open(image_file).convert("L")
            width, height = pil_img.size
            resized = pil_img.resize((max(1, int(width * 32 / max(height, 1))), 32))
            array = np.array(resized).astype(np.float32) / 255.0
            tensor = torch.tensor(array).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                decoded = decoder.decode_best_path(output)

            prediction = decoded[0].replace("<UNK>", "").strip() if decoded else ""
            if not prediction:
                empty_predictions += 1

            total_cer += calculate_cer(prediction, expected_text)
            total_wer += calculate_wer(prediction, expected_text)
            total_samples += 1

            if len(previews) < preview_limit:
                previews.append(
                    {
                        "ground_truth": normalize_text(expected_text),
                        "prediction": normalize_text(prediction),
                    }
                )

    if total_samples == 0:
        return {
            "cer": None,
            "wer": None,
            "empty_rate": None,
            "samples_evaluated": 0,
            "preview": previews,
        }

    return {
        "cer": (total_cer / total_samples) * 100.0,
        "wer": (total_wer / total_samples) * 100.0,
        "empty_rate": (empty_predictions / total_samples) * 100.0,
        "samples_evaluated": total_samples,
        "preview": previews,
    }


def discover_default_checkpoints(root_dir: str | Path | None = None) -> list[tuple[str, str | None]]:
    project_root = Path(root_dir) if root_dir else Path(__file__).resolve().parent.parent
    model_root = project_root / "model"

    patterns = [
        ("v1 (Base Gen.)", model_root / "checkpoints_v1" / "*.pth"),
        ("v2 (Early Aug.)", model_root / "checkpoints_v2" / "*.pth"),
        ("v3 (200K Extended)", model_root / "checkpoints_200k" / "*.pth"),
        ("v4 (Realistic)", model_root / "checkpoints_realistic_v2" / "*.pth"),
        ("v5 (Current Prod)", model_root / "checkpoints" / "run_v5_*" / "best.pth"),
        ("v6 (Edge STN)", model_root / "checkpoints_v6_edge" / "run_v6_*" / "best_v6.pth"),
        ("v7 (NLP-Guided)", model_root / "checkpoints_v7_nlp" / "run_v7_*" / "best_v7.pth"),
        ("v8 (Staged Curriculum)", model_root / "checkpoints_v8" / "run_v8_*" / "best_v8.pth"),
        ("v9 (Transformer)",       model_root / "checkpoints_v9" / "run_v9_*" / "best_v9.pth"),
    ]

    discovered: list[tuple[str, str | None]] = []
    for name, pattern in patterns:
        matches = glob.glob(str(pattern))
        if not matches:
            discovered.append((name, None))
            continue
        discovered.append((name, max(matches, key=lambda path: Path(path).stat().st_mtime)))
    return discovered
