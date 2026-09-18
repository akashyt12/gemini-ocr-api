"""
PaddleOCR API - Advanced OCR Service
Host on Railway - FREE!
"""

import os
import io
import base64
import tempfile
from typing import Optional, List
from PIL import Image
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ============================================
# APP SETUP
# ============================================
app = FastAPI(
    title="PaddleOCR API",
    description="Advanced OCR API with PaddleOCR - Host on Railway",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# OCR ENGINE (Lazy Loading)
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
            # Speed Optimizations
            det_limit_side_len=960,
            rec_batch_num=16,
            max_batch_size=16,
            # Accuracy Optimizations
            det_db_thresh=0.3,
            det_db_box_thresh=0.6,
            det_db_unclip_ratio=1.5,
            use_dilation=False,
            det_db_score_mode='fast',
        )
    return ocr_engine

# ============================================
# HELPER FUNCTIONS
# ============================================
def decode_base64_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to numpy array"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)

def process_result(result):
    """Process OCR result into clean JSON"""
    if result is None or len(result) == 0:
        return {"text": "", "words": [], "confidence": 0}
    
    all_text = []
    words = []
    confidences = []
    
    for line in result:
        if line is not None:
            for item in line:
                bbox = item[0]
                text = item[1][0]
                confidence = item[1][1]
                
                words.append({
                    "text": text,
                    "confidence": round(confidence, 4),
                    "bbox": [[int(p[0]), int(p[1])] for p in bbox]
                })
                all_text.append(text)
                confidences.append(confidence)
    
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        "text": " ".join(all_text),
        "words": words,
        "confidence": round(avg_confidence, 4),
        "word_count": len(words)
    }

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "name": "PaddleOCR API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "ocr": "/ocr - POST image for OCR",
            "ocr_base64": "/ocr/base64 - POST base64 image",
            "health": "/health - Health check"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "engine": "PaddleOCR"}

@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    language: str = Query(default="en", description="Language code"),
    enhance: bool = Query(default=False, description="Enable image enhancement")
):
    """
    OCR endpoint - Upload image file
    
    Parameters:
    - file: Image file (PNG, JPG, JPEG)
    - language: Language code (en, hi, ch, etc.)
    - enhance: Enable image enhancement for better accuracy
    """
    try:
        # Read file
        contents = await file.read()
        
        # Open image
        image = Image.open(io.BytesIO(contents))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Optional: Image enhancement
        if enhance:
            import cv2
            # Denoise
            if len(img_array.shape) == 3:
                img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
            # Sharpen
            kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
            img_array = cv2.filter2D(img_array, -1, kernel)
        
        # Get OCR engine
        ocr = get_ocr()
        
        # Run OCR
        result = ocr.ocr(img_array, cls=True)
        
        # Process result
        processed = process_result(result)
        processed["filename"] = file.filename
        processed["language"] = language
        
        return JSONResponse(content=processed)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr/base64")
async def ocr_base64_endpoint(
    image: str = Query(..., description="Base64 encoded image"),
    language: str = Query(default="en", description="Language code"),
    enhance: bool = Query(default=False, description="Enable image enhancement")
):
    """
    OCR endpoint - Base64 image
    
    Parameters:
    - image: Base64 encoded image string
    - language: Language code (en, hi, ch, etc.)
    - enhance: Enable image enhancement
    """
    try:
        # Decode base64
        img_array = decode_base64_image(image)
        
        # Optional: Image enhancement
        if enhance:
            import cv2
            if len(img_array.shape) == 3:
                img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
            kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
            img_array = cv2.filter2D(img_array, -1, kernel)
        
        # Get OCR engine
        ocr = get_ocr()
        
        # Run OCR
        result = ocr.ocr(img_array, cls=True)
        
        # Process result
        processed = process_result(result)
        processed["language"] = language
        
        return JSONResponse(content=processed)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr/batch")
async def ocr_batch_endpoint(
    files: List[UploadFile] = File(...),
    language: str = Query(default="en", description="Language code")
):
    """
    Batch OCR - Upload multiple images
    
    Parameters:
    - files: Multiple image files
    - language: Language code
    """
    try:
        results = []
        
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            img_array = np.array(image)
            
            ocr = get_ocr()
            result = ocr.ocr(img_array, cls=True)
            processed = process_result(result)
            processed["filename"] = file.filename
            
            results.append(processed)
        
        return JSONResponse(content={
            "results": results,
            "total": len(results)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr/url")
async def ocr_url_endpoint(
    url: str = Query(..., description="Image URL"),
    language: str = Query(default="en", description="Language code")
):
    """
    OCR from URL - Extract text from image URL
    
    Parameters:
    - url: Image URL
    - language: Language code
    """
    try:
        import requests
        
        # Download image
        response = requests.get(url, timeout=10)
        image = Image.open(io.BytesIO(response.content))
        img_array = np.array(image)
        
        # Run OCR
        ocr = get_ocr()
        result = ocr.ocr(img_array, cls=True)
        
        # Process result
        processed = process_result(result)
        processed["url"] = url
        processed["language"] = language
        
        return JSONResponse(content=processed)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# RUN SERVER
# ============================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
