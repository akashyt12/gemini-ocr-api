"""
PaddleOCR API - ADVANCED OCR Service
Image Crop + Resolution Boost + Multi-Part OCR
"""

import os
import io
import base64
import subprocess
import tempfile
from typing import Optional, List
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ============================================
# APP SETUP
# ============================================
app = FastAPI(
    title="Advanced PaddleOCR API",
    description="Advanced OCR with Image Crop + Resolution Boost",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# OCR ENGINE
# ============================================
ocr_engine = None

def get_ocr():
    global ocr_engine
    if ocr_engine is None:
        from paddleocr import PaddleOCR
        ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            show_log=False,
            det_limit_side_len=1280,
            rec_batch_num=32,
            max_batch_size=32,
            det_db_thresh=0.2,
            det_db_box_thresh=0.4,
            det_db_unclip_ratio=2.0,
            use_dilation=True,
            det_db_score_mode='slow',
        )
    return ocr_engine

# ============================================
# IMAGE PROCESSING UTILS
# ============================================

def increase_resolution(image: Image.Image, factor: int = 3) -> Image.Image:
    """Resolution 3x badhao"""
    width, height = image.size
    new_width = width * factor
    new_height = height * factor
    return image.resize((new_width, new_height), Image.LANCZOS)

def enhance_image(image: Image.Image) -> Image.Image:
    """Image enhance karo - contrast, sharpness, denoise"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Increase sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    # Increase brightness slightly
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)
    
    # Denoise
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    return image

def crop_image_grid(image: Image.Image, rows: int = 2, cols: int = 2) -> list:
    """Image ko grid mein crop karo (2x2 = 4 parts)"""
    width, height = image.size
    part_width = width // cols
    part_height = height // rows
    
    parts = []
    for r in range(rows):
        for c in range(cols):
            left = c * part_width
            top = r * part_height
            right = left + part_width
            bottom = top + part_height
            
            cropped = image.crop((left, top, right, bottom))
            parts.append({
                'image': cropped,
                'position': f'row{r+1}_col{c+1}',
                'bbox': [left, top, right, bottom]
            })
    
    return parts

def crop_image_sliding(image: Image.Image, window_size: float = 0.5, overlap: float = 0.25) -> list:
    """Sliding window se image crop karo"""
    width, height = image.size
    win_w = int(width * window_size)
    win_h = int(height * window_size)
    step_x = int(win_w * (1 - overlap))
    step_y = int(win_h * (1 - overlap))
    
    parts = []
    y = 0
    row = 0
    while y + win_h <= height:
        x = 0
        col = 0
        while x + win_w <= width:
            cropped = image.crop((x, y, x + win_w, y + win_h))
            parts.append({
                'image': cropped,
                'position': f'window_{row}_{col}',
                'bbox': [x, y, x + win_w, y + win_h]
            })
            x += step_x
            col += 1
        y += step_y
        row += 1
    
    # Add remaining edge parts
    if y < height and x < width:
        cropped = image.crop((width - win_w, height - win_h, width, height))
        parts.append({
            'image': cropped,
            'position': 'window_corner',
            'bbox': [width - win_w, height - win_h, width, height]
        })
    
    return parts

def ffmpeg_enhance(image_path: str, output_path: str) -> str:
    """FFmpeg se image enhance karo"""
    try:
        cmd = [
            'ffmpeg', '-y', '-i', image_path,
            '-vf', 'unsharp=5:5:1.0:5:5:0.0',
            '-q:v', '1',
            output_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=30)
        return output_path
    except Exception:
        return image_path

def upscale_with_ffmpeg(image_path: str, output_path: str, scale: int = 2) -> str:
    """FFmpeg se resolution badhao"""
    try:
        cmd = [
            'ffmpeg', '-y', '-i', image_path,
            '-vf', f'scale=iw*{scale}:ih*{scale}:flags=lanczos',
            '-q:v', '1',
            output_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=30)
        return output_path
    except Exception:
        return image_path

# ============================================
# ADVANCED OCR PIPELINE
# ============================================

def advanced_ocr(image: Image.Image, mode: str = "grid") -> dict:
    """
    Advanced OCR Pipeline:
    1. Resolution increase
    2. Image enhance
    3. Crop into parts
    4. OCR each part
    5. Merge results
    """
    ocr = get_ocr()
    all_words = []
    all_text = []
    
    # Step 1: Resolution Increase
    image_3x = increase_resolution(image, factor=3)
    
    # Step 2: Enhance
    image_enhanced = enhance_image(image_3x)
    
    # Step 3: Crop into parts
    if mode == "grid":
        parts = crop_image_grid(image_enhanced, rows=2, cols=2)
    elif mode == "sliding":
        parts = crop_image_sliding(image_enhanced, window_size=0.5, overlap=0.25)
    else:
        parts = [{'image': image_enhanced, 'position': 'full', 'bbox': [0, 0, image_enhanced.width, image_enhanced.height]}]
    
    # Step 4: OCR each part
    for part in parts:
        part_img = np.array(part['image'])
        try:
            result = ocr.ocr(part_img, cls=True)
            if result and len(result) > 0:
                for line in result:
                    if line:
                        for item in line:
                            bbox = item[0]
                            text = item[1][0]
                            confidence = item[1][1]
                            
                            # Adjust bbox to original coordinates
                            offset_x = part['bbox'][0]
                            offset_y = part['bbox'][1]
                            scale_factor = 3  # We scaled 3x
                            
                            adjusted_bbox = [
                                [int(p[0]/scale_factor + offset_x), int(p[1]/scale_factor + offset_y)] 
                                for p in bbox
                            ]
                            
                            # Avoid duplicates
                            word_entry = {
                                'text': text,
                                'confidence': round(confidence, 4),
                                'bbox': adjusted_bbox,
                                'part': part['position']
                            }
                            
                            # Check if similar word already exists
                            is_duplicate = False
                            for existing in all_words:
                                if (existing['text'].lower() == text.lower() and 
                                    abs(existing['bbox'][0][0] - adjusted_bbox[0][0]) < 50):
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                all_words.append(word_entry)
                                all_text.append(text)
        except Exception as e:
            continue
    
    # Step 5: Sort by position (top to bottom, left to right)
    all_words.sort(key=lambda w: (w['bbox'][0][1], w['bbox'][0][0]))
    
    avg_confidence = sum(w['confidence'] for w in all_words) / len(all_words) if all_words else 0
    
    return {
        'text': ' '.join(all_text),
        'words': all_words,
        'confidence': round(avg_confidence, 4),
        'word_count': len(all_words),
        'parts_processed': len(parts),
        'mode': mode
    }

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "name": "Advanced PaddleOCR API v2.0",
        "status": "running",
        "features": [
            "Image Crop (Grid/Sliding Window)",
            "Resolution Boost (3x)",
            "Image Enhancement",
            "Multi-Part OCR",
            "Deduplication"
        ],
        "endpoints": {
            "ocr_grid": "/ocr?mode=grid - Grid crop (2x2=4 parts)",
            "ocr_sliding": "/ocr?mode=sliding - Sliding window",
            "ocr_full": "/ocr?mode=full - Full image",
            "ocr_base64": "/ocr/base64 - Base64 image",
            "ocr_url": "/ocr/url - Image URL",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0", "engine": "PaddleOCR Advanced"}

@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    language: str = Query(default="en"),
    mode: str = Query(default="grid", description="grid, sliding, or full"),
    enhance: bool = Query(default=True)
):
    """Advanced OCR - Image crop + Resolution boost"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        result = advanced_ocr(image, mode=mode)
        result['filename'] = file.filename
        result['language'] = language
        result['original_size'] = [image.width, image.height]
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr/base64")
async def ocr_base64_endpoint(
    image: str = Query(...),
    language: str = Query(default="en"),
    mode: str = Query(default="grid"),
    enhance: bool = Query(default=True)
):
    """Advanced OCR - Base64 image"""
    try:
        if ',' in image:
            image = image.split(',')[1]
        
        image_data = base64.b64decode(image)
        image = Image.open(io.BytesIO(image_data))
        
        result = advanced_ocr(image, mode=mode)
        result['language'] = language
        result['original_size'] = [image.width, image.height]
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr/url")
async def ocr_url_endpoint(
    url: str = Query(...),
    language: str = Query(default="en"),
    mode: str = Query(default="grid"),
    enhance: bool = Query(default=True)
):
    """Advanced OCR - Image URL"""
    try:
        import requests
        response = requests.get(url, timeout=15)
        image = Image.open(io.BytesIO(response.content))
        
        result = advanced_ocr(image, mode=mode)
        result['language'] = language
        result['url'] = url
        result['original_size'] = [image.width, image.height]
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr/batch")
async def ocr_batch_endpoint(
    files: List[UploadFile] = File(...),
    language: str = Query(default="en"),
    mode: str = Query(default="grid")
):
    """Advanced OCR - Multiple images"""
    try:
        results = []
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            result = advanced_ocr(image, mode=mode)
            result['filename'] = file.filename
            results.append(result)
        
        return JSONResponse(content={"results": results, "total": len(results)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# RUN SERVER
# ============================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
