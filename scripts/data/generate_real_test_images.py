import os
import sys
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root_dir)

# 20 real-world style Hindi texts (news, books, posters)
HINDI_TEXTS = [
    "आज की ताज़ा ख़बर", # "Today's latest news"
    "भारतीय रिज़र्व बैंक", # "Reserve Bank of India"
    "कृपया यहाँ हस्ताक्षर करें", # "Please sign here"
    "महत्वपूर्ण सूचना", # "Important notice"
    "रेलवे स्टेशन और बस स्टैंड", # "Railway station and bus stand"
    "स्वच्छ भारत अभियान", # "Clean India Mission"
    "विभागीय परीक्षा परिणाम", # "Departmental exam results"
    "राष्ट्रीय शिक्षा नीति", # "National Education Policy"
    "सार्वजनिक संपत्ति की सुरक्षा", # "Protection of public property"
    "कला, विज्ञान और वाणिज्‍य", # "Arts, Science and Commerce"
    "आपका खाता सफलतापूर्वक खुल गया", # "Your account successfully opened"
    "दिल्ली विश्वविद्यालय प्रवेश", # "Delhi University Admission"
    "मोबाइल फोन का उपयोग वर्जित है", # "Use of mobile phone is prohibited"
    "प्रदूषण नियंत्रण बोर्ड", # "Pollution Control Board"
    "दैनिक जागरण समाचार", # "Dainik Jagran News"
    "अस्पताल और चिकित्सालय", # "Hospital and Clinic"
    "यातायात नियमों का पालन करें", # "Follow traffic rules"
    "पुलिस अधीक्षक कार्यालय", # "Superintendent of Police office"
    "मतदाता पहचान पत्र", # "Voter ID card"
    "बैंक ऑफ इंडिया पासबुक" # "Bank of India Passbook"
]

def generate_realistic_images():
    out_dir = os.path.join(root_dir, 'data', 'real_test', 'hindi')
    os.makedirs(out_dir, exist_ok=True)
    
    font_dir = os.path.join(root_dir, 'data', 'fonts')
    fonts = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.endswith('.ttf')]
    
    gt_file = os.path.join(out_dir, 'ground_truth.txt')
    
    with open(gt_file, 'w', encoding='utf-8') as f:
        for i, text in enumerate(HINDI_TEXTS):
            # Select random font if available, else try to use a default Hindi font
            font_path = random.choice(fonts) if fonts else None
            
            # Increase base font size to simulate high-res captures before downsizing
            font_size = random.randint(40, 70)
            try:
                if font_path:
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    font = ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()
            
            # Determine text size
            temp_img = Image.new('RGB', (1, 1))
            temp_draw = ImageDraw.Draw(temp_img)
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            # Add padding for margins simulating document crops
            pad_x = random.randint(20, 100)
            pad_y = random.randint(20, 100)
            img_w, img_h = text_w + pad_x, text_h + pad_y
            
            # Create varied backgrounds
            bg_style = random.choice(['white', 'off_white', 'gradient', 'noisy', 'gray_paper'])
            
            if bg_style == 'white':
                bg_color = (255, 255, 255)
                img = Image.new('RGB', (img_w, img_h), color=bg_color)
            elif bg_style == 'off_white':
                bg_color = (random.randint(240, 255), random.randint(235, 255), random.randint(220, 245))
                img = Image.new('RGB', (img_w, img_h), color=bg_color)
            elif bg_style == 'gray_paper':
                v = random.randint(180, 220)
                img = Image.new('RGB', (img_w, img_h), color=(v, v, v))
            else:
                img = Image.new('RGB', (img_w, img_h), color=(255, 255, 255))
            
            draw = ImageDraw.Draw(img)
            
            # Text color (black to dark gray/blue)
            text_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 60))
            
            start_x = pad_x // 2
            start_y = pad_y // 2
            draw.text((start_x, start_y - bbox[1]), text, font=font, fill=text_color)
            
            # Convert to numpy for augmentations (OpenCV)
            img_cv = np.array(img)
            
            if bg_style == 'gradient':
                # Add lighting gradient
                mask = np.zeros((img_h, img_w, 3), dtype=np.float32)
                gradient = np.linspace(random.uniform(0.5, 0.8), 1.0, img_w)
                mask[:,:,0] = gradient
                mask[:,:,1] = gradient
                mask[:,:,2] = gradient
                img_cv = (img_cv * mask).astype(np.uint8)
                
            if bg_style == 'noisy':
                noise = np.random.normal(0, random.uniform(5, 25), img_cv.shape).astype(np.int16)
                img_cv = np.clip(img_cv.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Apply realistic distortions
            style = random.choice(['clean', 'blurred', 'rotated', 'shadow', 'speckled'])
            
            if style == 'blurred':
                img_cv = cv2.GaussianBlur(img_cv, (random.choice([3, 5]), random.choice([3, 5])), 0)
            elif style == 'rotated':
                angle = random.uniform(-3, 3)
                M = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
                img_cv = cv2.warpAffine(img_cv, M, (img_w, img_h), borderValue=(255, 255, 255))
            elif style == 'speckled':
                # salt and pepper like
                amount = random.uniform(0.001, 0.01)
                num_salt = np.ceil(amount * img_cv.size * 0.5)
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_cv.shape]
                img_cv[tuple(coords)] = 0
            
            # Add subtle JPEG compression artifacts randomly
            if random.random() > 0.5:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(60, 95)]
                result, encimg = cv2.imencode('.jpg', img_cv, encode_param)
                img_cv = cv2.imdecode(encimg, 1)

            final_img = Image.fromarray(img_cv)
            filename = f"real_doc_{i:02d}.jpg"
            save_path = os.path.join(out_dir, filename)
            final_img.save(save_path, quality=95)
            
            f.write(f"{filename}\t{text}\n")
            
    print(f"Generated 20 real-world simulated test images in {out_dir}")

if __name__ == '__main__':
    generate_realistic_images()
