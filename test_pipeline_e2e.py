import os
import sys
import numpy as np
from PIL import Image

root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root_dir)

from preprocess.pipeline import preprocess
from model.inference import recognize

try:
    from nlp.postprocessor import correct
except ImportError:
    # Fallback to no post-processor if not found
    print("Warning: nlp.postprocessor.correct not found. Skipping NLP post-processing.")
    def correct(text, lang): return text

def calculate_exact_match(gt, pred):
    return gt.strip() == pred.strip()

def has_unk(pred):
    return '<UNK>' in pred

def evaluate_real_world():
    test_dir = os.path.join(root_dir, 'data', 'real_test', 'hindi')
    gt_file = os.path.join(test_dir, 'ground_truth.txt')
    
    if not os.path.exists(gt_file):
        print(f"Ground truth file not found at {gt_file}")
        return
        
    results = []
    perfect_matches = 0
    unk_count = 0
    total = 0
    
    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            filename, gt_text = parts[0], parts[1]
            img_path = os.path.join(test_dir, filename)
            
            if not os.path.exists(img_path):
                print(f"Missing image {img_path}")
                continue
                
            try:
                # 1. Load Image
                image = Image.open(img_path).convert('RGB')
                
                # 2. Preprocess
                line_images = preprocess(image, language='hi')
                
                # 3. Model Inference
                lines_np = []
                for l_img in line_images:
                    arr = np.array(l_img).astype(np.float32)
                    if arr.max() > 1.0:
                        arr = arr / 255.0
                    lines_np.append(arr)
                
                recognized_lines = recognize(lines_np)
                
                # 4. Post-Process (NLP)
                raw_text = "\n".join(recognized_lines)
                final_text = correct(raw_text, 'hi')
                
                # Evaluation Metrics
                is_exact = calculate_exact_match(gt_text, final_text)
                has_u = has_unk(final_text)
                
                if is_exact:
                    perfect_matches += 1
                if has_u:
                    unk_count += 1
                total += 1
                
                results.append({
                    "filename": filename,
                    "gt": gt_text,
                    "pred": final_text,
                    "exact": is_exact,
                    "unk": has_u
                })
                
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
                
    print("\n" + "="*50)
    print("E2E PIPELINE REAL-WORLD TEST RESULTS")
    print("="*50)
    
    for r in results:
        status = "✅" if r['exact'] else "❌"
        unk_marker = " (Has <UNK>)" if r['unk'] else ""
        print(f"\nImage: {r['filename']}")
        print(f"Ground Truth : {r['gt']}")
        print(f"Prediction   : {r['pred']}")
        print(f"Status       : {status}{unk_marker}")
        
    print("\n" + "="*50)
    print(f"Total Images   : {total}")
    print(f"Perfect Matches: {perfect_matches} ({(perfect_matches/total)*100:.2f}%)")
    print(f"Images w/ UNK  : {unk_count} ({(unk_count/total)*100:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    evaluate_real_world()
