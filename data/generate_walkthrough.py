import os
import random

artifact_path = r"c:\Users\jayhi\.gemini\antigravity\brain\10d3b60a-6a2e-41d3-98b2-1d2dad8fa35d\walkthrough.md"
old_dir = r"C:\Users\jayhi\Downloads\akshara-ocr\data\train\hi"
new_dir = r"C:\Users\jayhi\Downloads\akshara-ocr\data\realistic_synthetic\train\hi"

try:
    old_files = [f for f in os.listdir(old_dir) if f.endswith('.png')]
    new_files = [f for f in os.listdir(new_dir) if f.endswith('.png')]
    
    random.seed(42)
    selected_old = random.sample(old_files, 20)
    selected_new = random.sample(new_files, 20)

    md_content = """# Realistic Data Generation Verification

We successfully redesigned the generation pipeline. The new data mimics actual document captures far better than the previous baseline.

Key improvements implemented:
- **Perspective Warping** to simulate 3D camera angles.
- **Lighting & Shadows** via multi-stop gradients across the document surface.
- **Realistic Blurring** (motion blur simulating shake + gaussian).
- **Ink Bleed & Scanner Noise** (morphological filters + artifact lines).
- **True Paper Texture & Variable Backgrounds**.
- **Disjoint Validation Split** with distinctly severe / weak augmentations to ensure true evaluation.

Here is a side-by-side comparison.

| Clean (Old) | Realistic (New) |
| :---: | :---: |
"""

    for old, new in zip(selected_old, selected_new):
        old_path = os.path.join(old_dir, old).replace('\\', '/')
        new_path = os.path.join(new_dir, new).replace('\\', '/')
        md_content += f"| ![Old Baseline](file:///{old_path}) | ![Realistic Augmentation](file:///{new_path}) |\n"

    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print("Walkthrough artifact successfully generated.")

except Exception as e:
    print("Error:", e)
