# data/labeling_tool/main.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import csv

app = FastAPI()

# Make sure 'templates' and 'images_to_label' directory exists
os.makedirs("templates", exist_ok=True)
IMAGES_DIR = "images_to_label"
os.makedirs(IMAGES_DIR, exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

LABELS_FILE = "manual_labels.csv"

def get_unlabeled_images():
    labeled = set()
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None) # skip header
            for row in reader:
                if row:
                    labeled.add(row[0])
                    
    all_images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return [img for img in all_images if img not in labeled]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    unlabeled = get_unlabeled_images()
    current_image = unlabeled[0] if unlabeled else None
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "image": current_image, 
        "remaining": len(unlabeled)
    })

@app.post("/submit", response_class=HTMLResponse)
async def submit(request: Request, filename: str = Form(...), text: str = Form(...), language: str = Form(...)):
    if not os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "text", "language", "font", "is_synthetic"])
            
    with open(LABELS_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([filename, text, language, "manual", False])
        
    return await index(request)

if __name__ == '__main__':
    # Example usage
    import uvicorn
    print("Run `uvicorn main:app --reload` to start the labeling tool.")
    print("Add images to the `images_to_label` directory to get started.")
