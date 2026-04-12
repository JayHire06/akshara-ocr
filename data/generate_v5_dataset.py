"""
Akshara-OCR v5 Dataset Generator
──────────────────────────────────
Generates synthetic Hindi/Devanagari OCR training images from the fonts
directory and writes pipe-separated label files usable by train_v5.py.

Output:
    data/combined/train_labels.txt  (~80k lines)
    data/combined/val_labels.txt    (~10k lines)

Format per line:  /abs/path/to/image.png|label_text
"""

import os
import sys
import random
import json
import time
import hashlib
import multiprocessing as mp
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# ─── Config ──────────────────────────────────────────────────────────────────

ROOT_DIR       = Path(__file__).resolve().parent.parent
FONTS_DIR      = ROOT_DIR / "fonts"
OUTPUT_DIR     = ROOT_DIR / "data" / "combined"
VOCAB_FILE     = OUTPUT_DIR / "vocab.json"

NUM_SAMPLES    = 100_000
TRAIN_RATIO    = 0.85
NUM_WORKERS    = max(1, mp.cpu_count() - 2)   # leave 2 cores for OS

# Rich vocabulary of Hindi words + phrases
HINDI_WORDS = [
    "नमस्ते", "भारत", "हिंदी", "देवनागरी", "अक्षर", "विज्ञान", "प्रौद्योगिकी",
    "ज्ञान", "कम्प्यूटर", "आर्टिफिशियल", "बुद्धिमत्ता", "मशीन", "डेटा",
    "शिक्षा", "विद्यालय", "विश्वविद्यालय", "छात्र", "शिक्षक", "पुस्तक", "अध्ययन",
    "अनुसंधान", "तकनीक", "भविष्य", "नवाचार", "विकास", "सामाजिक", "आर्थिक",
    "स्वास्थ्य", "पर्यावरण", "संस्कृति", "परंपरा", "उत्सव", "समाज", "परिवार",
    "सरकार", "नीति", "कानून", "न्याय", "स्वतंत्रता", "लोकतंत्र", "संविधान",
    "यात्रा", "पर्यटन", "इतिहास", "भूगोल", "कला", "संगीत", "साहित्य", "कविता",
    "व्यापार", "उद्योग", "कृषि", "उत्पादन", "बाजार", "मूल्य", "अर्थव्यवस्था",
    "रेलवे", "विमान", "सड़क", "पुल", "नदी", "पहाड़", "मैदान", "जंगल",
    "प्रधानमंत्री", "राष्ट्रपति", "मंत्री", "संसद", "राज्य", "जिला", "गाँव",
    "स्वच्छ", "प्रदूषण", "ऊर्जा", "सौर", "पवन", "जल", "भूमि", "आकाश",
    "दिल्ली", "मुंबई", "कोलकाता", "चेन्नई", "बेंगलुरु", "हैदराबाद", "पुणे",
    "राष्ट्रीय", "अंतर्राष्ट्रीय", "वैश्विक", "स्थानीय", "क्षेत्रीय", "केंद्रीय",
    "महत्वपूर्ण", "आवश्यक", "उपयोगी", "प्रभावी", "सफल", "असफल", "कठिन", "सरल",
    "रामायण", "महाभारत", "गीता", "उपनिषद", "वेद", "पुराण", "इतिहास",
    "सत्यमेव", "जयते", "वंदे", "मातरम्", "जन", "गण", "मन",
]

HINDI_PHRASES = [
    "भारत एक कृषी प्रधान देश है",
    "सत्यमेव जयते",
    "ज्ञान ही शक्ति है",
    "स्वच्छ भारत अभियान",
    "राष्ट्रीय शिक्षा नीति",
    "आपका खाता सफलतापूर्वक",
    "दिल्ली विश्वविद्यालय",
    "प्रदूषण नियंत्रण बोर्ड",
    "महत्वपूर्ण सूचना",
    "कठोर परिश्रम सफलता की कुंजी है",
    "समय धन से अधिक मूल्यवान है",
    "विद्या ददाति विनयम",
    "आर्टिफिशियल इंटेलिजेंस का भविष्य",
    "डेटा विज्ञान के अनुप्रयोग",
    "प्रिय महोदय",
    "सादर प्रणाम",
    "जय हिन्द",
    "भारत माता की जय",
    "वंदे मातरम्",
    "साइबर सुरक्षा",
    "इंटरनेट सेवाएं",
    "मोबाइल भुगतान",
    "डिजिटल इंडिया",
    "मेक इन इंडिया",
    "स्टार्टअप इंडिया",
    "आत्मनिर्भर भारत",
    "नई दिल्ली रेलवे स्टेशन",
    "मुंबई हवाई अड्डा",
    "राष्ट्रीय राजमार्ग संख्या",
]


# ─── Text sampling ────────────────────────────────────────────────────────────

def get_random_text(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.40:
        return rng.choice(HINDI_WORDS)
    elif r < 0.70:
        return " ".join(rng.choices(HINDI_WORDS, k=rng.randint(2, 3)))
    else:
        return rng.choice(HINDI_PHRASES)


# ─── Image rendering + augmentation ──────────────────────────────────────────

def render_text_image(text: str, font_path: str, rng: random.Random,
                      augment: bool = True) -> Image.Image:
    font_size = rng.randint(16, 36)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Measure text
    dummy = Image.new("L", (1, 1))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), text, font=font)
    text_w = max(bbox[2] - bbox[0], 1)
    text_h = max(bbox[3] - bbox[1], 1)
    pad_x, pad_y = rng.randint(4, 12), rng.randint(3, 8)
    w, h = text_w + pad_x * 2, text_h + pad_y * 2

    # Background
    bg_colors = [255, 250, 245, 240, 235]
    bg = rng.choice(bg_colors)
    img = Image.new("L", (w, h), bg)

    # Paper texture
    if augment and rng.random() < 0.7:
        arr = np.array(img, dtype=np.float32)
        noise = rng.gauss(0, rng.uniform(3, 18))
        arr += np.random.normal(0, abs(noise) + 1, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Draw text
    d = ImageDraw.Draw(img)
    ink = rng.randint(0, 40)
    d.text((pad_x, pad_y), text, fill=ink, font=font)

    if not augment:
        return img

    # Augmentations
    if rng.random() < 0.5:
        factor = rng.uniform(0.85, 1.15)
        img = img.resize((max(1, int(img.width * factor)), img.height), Image.BILINEAR)

    if rng.random() < 0.4:
        radius = rng.uniform(0.3, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    if rng.random() < 0.3:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(rng.uniform(0.6, 1.4))

    if rng.random() < 0.25:
        # Resolution degradation
        factor = rng.uniform(0.4, 0.7)
        small = img.resize((max(1, int(img.width * factor)), max(1, int(img.height * factor))), Image.BILINEAR)
        img = small.resize(img.size, Image.BICUBIC)

    return img


# ─── Worker function (runs in subprocess) ────────────────────────────────────

def _worker(args):
    worker_id, indices, fonts, split, out_dir = args
    rng = random.Random(worker_id * 7919 + 42)
    np.random.seed(worker_id * 31 + 1)

    split_dir = out_dir / split / "images"
    split_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    augment = (split == "train")

    for i in indices:
        text = get_random_text(rng)
        font_path = rng.choice(fonts)

        try:
            img = render_text_image(text, font_path, rng, augment=augment)
        except Exception:
            continue

        # Unique filename
        uid = hashlib.md5(f"{worker_id}_{i}_{text}".encode()).hexdigest()[:12]
        fname = split_dir / f"{uid}.png"
        try:
            img.save(str(fname))
        except Exception:
            continue

        lines.append(f"{fname}|{text}")

    return lines


# ─── Main ────────────────────────────────────────────────────────────────────

def build_vocab(label_files):
    chars = set()
    for lf in label_files:
        with open(lf, encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    label = line.strip().split("|", 1)[1]
                    chars.update(label)
    vocab = {"<blank>": 0}
    for i, c in enumerate(sorted(chars), start=1):
        vocab[c] = i
    return vocab


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check fonts
    fonts = [str(FONTS_DIR / f) for f in os.listdir(FONTS_DIR)
             if f.lower().endswith((".ttf", ".otf"))]
    if not fonts:
        print(f"ERROR: No fonts found in {FONTS_DIR}")
        sys.exit(1)
    print(f"Found {len(fonts)} fonts in {FONTS_DIR}")

    # Check if data already exists
    train_file = OUTPUT_DIR / "train_labels.txt"
    val_file = OUTPUT_DIR / "val_labels.txt"
    if train_file.exists() and val_file.exists():
        n_train = sum(1 for _ in open(train_file, encoding="utf-8"))
        n_val   = sum(1 for _ in open(val_file, encoding="utf-8"))
        print(f"Dataset already exists: {n_train} train, {n_val} val samples.")
        print("Delete data/combined/ to regenerate. Continuing with existing data.")
        return

    n_train = int(NUM_SAMPLES * TRAIN_RATIO)
    n_val   = NUM_SAMPLES - n_train
    print(f"Generating {n_train} train + {n_val} val images using {NUM_WORKERS} workers...")
    t0 = time.time()

    # Split fonts: use different fonts for val (generalization test)
    random.shuffle(fonts)
    n_val_fonts = max(1, len(fonts) // 5)
    val_fonts   = fonts[:n_val_fonts]
    train_fonts = fonts[n_val_fonts:] if len(fonts) > n_val_fonts else fonts

    def chunk_indices(total, n_workers):
        size = total // n_workers
        chunks = []
        for k in range(n_workers):
            start = k * size
            end = start + size if k < n_workers - 1 else total
            chunks.append(list(range(start, end)))
        return chunks

    pool = mp.Pool(processes=NUM_WORKERS)

    # Train jobs
    train_chunks = chunk_indices(n_train, NUM_WORKERS)
    train_args = [
        (i, chunk, train_fonts, "train", OUTPUT_DIR)
        for i, chunk in enumerate(train_chunks)
    ]
    # Val jobs
    val_chunks = chunk_indices(n_val, NUM_WORKERS)
    val_args = [
        (NUM_WORKERS + i, chunk, val_fonts, "val", OUTPUT_DIR)
        for i, chunk in enumerate(val_chunks)
    ]

    all_args = train_args + val_args
    results = pool.map(_worker, all_args)
    pool.close()
    pool.join()

    # Write label files
    train_lines, val_lines = [], []
    for i, lines in enumerate(results):
        if i < NUM_WORKERS:
            train_lines.extend(lines)
        else:
            val_lines.extend(lines)

    with open(train_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines))
    with open(val_file, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines))

    elapsed = time.time() - t0
    print(f"\n✓ Generated {len(train_lines)} train + {len(val_lines)} val samples in {elapsed/60:.1f} min")
    print(f"  → {train_file}")
    print(f"  → {val_file}")

    # Build and save vocab
    vocab = build_vocab([train_file, val_file])
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  → {VOCAB_FILE}  (vocab_size={len(vocab)})")


if __name__ == "__main__":
    main()
