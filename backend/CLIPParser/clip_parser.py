import os
import tempfile
import time
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import torch
import pandas as pd
import uvicorn
from transformers import CLIPProcessor, CLIPModel
import easyocr

# Initialize FastAPI
app = FastAPI()

# Device setup
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

print(f"Using device: {device}")

# Initialize CLIP model and processor
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("✅ CLIP model loaded successfully")
except Exception as e:
    print(f"⚠️ Failed to load CLIP model: {e}")
    clip_model = None
    clip_processor = None

# Initialize OCR
try:
    ocr_reader = easyocr.Reader(['en'], gpu=(device != 'cpu'))
    print("✅ EasyOCR initialized successfully")
except Exception as e:
    print(f"⚠️ Failed to initialize EasyOCR: {e}")
    ocr_reader = None

def detect_ui_elements(image_array, confidence_threshold=0.3):
    """
    Simple UI element detection using contours and basic heuristics
    Much faster than YOLO-based detection
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and process contours to find UI elements
    ui_elements = []
    image_height, image_width = image_array.shape[:2]
    
    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by size (avoid very small or very large elements)
        if w < 20 or h < 20 or w > image_width * 0.8 or h > image_height * 0.8:
            continue
            
        # Calculate aspect ratio
        aspect_ratio = w / h
        
        # Filter by aspect ratio (avoid very thin elements)
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            continue
            
        # Calculate area
        area = cv2.contourArea(contour)
        if area < 400:  # Minimum area threshold
            continue
            
        ui_elements.append({
            'bbox': [x, y, x + w, y + h],
            'confidence': min(area / 10000, 1.0),  # Simple confidence based on area
            'type': 'ui_element'
        })
    
    # Remove overlapping elements (simple NMS)
    ui_elements = simple_nms(ui_elements, 0.5)
    
    return ui_elements

def simple_nms(elements, iou_threshold):
    """Simple Non-Maximum Suppression"""
    if not elements:
        return []
    
    # Sort by confidence
    elements = sorted(elements, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    for i, elem in enumerate(elements):
        should_keep = True
        for kept_elem in keep:
            if calculate_iou(elem['bbox'], kept_elem['bbox']) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(elem)
    
    return keep

def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def extract_text_with_ocr(image_array):
    """Extract text using OCR"""
    if ocr_reader is None:
        return []
    
    try:
        # Use EasyOCR to detect text
        results = ocr_reader.readtext(image_array)
        
        text_elements = []
        for (bbox, text, confidence) in results:
            if confidence > 0.5 and len(text.strip()) > 0:  # Filter low confidence and empty text
                # Convert bbox format
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                text_elements.append({
                    'bbox': [x_min, y_min, x_max, y_max],
                    'text': text.strip(),
                    'confidence': confidence,
                    'type': 'text'
                })
        
        return text_elements
    except Exception as e:
        print(f"OCR error: {e}")
        return []

def classify_elements_with_clip(image_array, elements):
    """Use CLIP to classify UI elements"""
    if clip_model is None or clip_processor is None:
        return elements
    
    try:
        # Define common UI element types
        ui_types = [
            "button", "text field", "icon", "image", "menu item", 
            "checkbox", "radio button", "dropdown", "slider", "tab"
        ]
        
        pil_image = Image.fromarray(image_array)
        
        for element in elements:
            if element['type'] == 'text':
                continue  # Skip text elements
                
            # Extract element region
            x1, y1, x2, y2 = element['bbox']
            element_region = pil_image.crop((x1, y1, x2, y2))
            
            # Prepare inputs for CLIP
            inputs = clip_processor(
                text=[f"a {ui_type}" for ui_type in ui_types],
                images=element_region,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                
            # Get the most likely UI type
            best_idx = probs.argmax().item()
            best_confidence = probs[0, best_idx].item()
            
            if best_confidence > 0.3:  # Confidence threshold
                element['ui_type'] = ui_types[best_idx]
                element['clip_confidence'] = best_confidence
            else:
                element['ui_type'] = 'unknown'
                element['clip_confidence'] = best_confidence
                
    except Exception as e:
        print(f"CLIP classification error: {e}")
    
    return elements

def draw_labeled_image(image_array, elements):
    """Draw bounding boxes and labels on the image"""
    pil_image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    colors = {
        'text': 'red',
        'button': 'blue',
        'icon': 'green',
        'ui_element': 'orange',
        'unknown': 'purple'
    }
    
    for i, element in enumerate(elements):
        x1, y1, x2, y2 = element['bbox']
        
        # Determine color based on type
        element_type = element.get('ui_type', element['type'])
        color = colors.get(element_type, 'gray')
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Prepare label
        if element['type'] == 'text':
            label = f"{i}: {element['text'][:20]}"
        else:
            ui_type = element.get('ui_type', 'element')
            label = f"{i}: {ui_type}"
        
        # Draw label background
        text_bbox = draw.textbbox((x1, y1 - 15), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        
        # Draw label text
        draw.text((x1, y1 - 15), label, fill='white', font=font)
    
    return pil_image

@app.post("/process_image/")
async def process_image(
    file: UploadFile = File(...),
    box_threshold: float = Form(0.3),
    iou_threshold: float = Form(0.5),
    imgsz_component: int = Form(640)  # Keep for compatibility, but not used
):
    try:
        start_time = time.time()
        
        # Save uploaded file to temporary path
        contents = await file.read()
        temp_dir = tempfile.mkdtemp()
        temp_image_path = os.path.join(temp_dir, file.filename)

        with open(temp_image_path, 'wb') as f:
            f.write(contents)
        
        # Load image
        image = Image.open(temp_image_path).convert('RGB')
        image_array = np.array(image)
        
        # Extract text with OCR
        text_elements = extract_text_with_ocr(image_array)
        
        # Detect UI elements
        ui_elements = detect_ui_elements(image_array, confidence_threshold=box_threshold)
        
        # Combine all elements
        all_elements = text_elements + ui_elements
        
        # Classify elements with CLIP (optional, can be disabled for speed)
        if len(all_elements) > 0:
            all_elements = classify_elements_with_clip(image_array, all_elements)
        
        # Draw labeled image
        labeled_image = draw_labeled_image(image_array, all_elements)
        
        # Convert labeled image to base64
        buffer = io.BytesIO()
        labeled_image.save(buffer, format='PNG')
        labeled_image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Format parsed content for compatibility with OmniParser
        parsed_content_list = []
        for i, element in enumerate(all_elements):
            x1, y1, x2, y2 = element['bbox']
            
            if element['type'] == 'text':
                content = f"{i}: {element['text']}"
            else:
                ui_type = element.get('ui_type', 'ui_element')
                content = f"{i}: {ui_type}"
            
            parsed_content_list.append({
                'ID': i,
                'bbox': [x1, y1, x2, y2],
                'text': element.get('text', ''),
                'type': element['type'],
                'ui_type': element.get('ui_type', ''),
                'confidence': element['confidence'],
                'content': content
            })
        
        elapsed_time = time.time() - start_time
        
        # Clean up temporary files
        os.remove(temp_image_path)
        os.rmdir(temp_dir)
        
        return {
            "status": "success",
            "parsed_content": parsed_content_list,
            "labeled_image": labeled_image_base64,
            "e_time": elapsed_time
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "CLIP Parser"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
