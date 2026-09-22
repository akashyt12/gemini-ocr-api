"""
OCR API v4.0 - GEMINI VISION ONLY
Ultra Lightweight
"""

import os
import io
import base64
from PIL import Image
import requests

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="OCR API v4.0 - Gemini Vision")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_KEY = os.environ.get("GEMINI_KEY", "")

def gemini_ocr(image: Image.Image) -> dict:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=85)
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}'
    payload = {
        'contents': [{'parts': [
            {'inline_data': {'mime_type': 'image/jpeg', 'data': img_b64}},
            {'text': 'Read ALL text in this image exactly as shown. Return every word, number, symbol. Format as clean text.'}
        ]}]
    }
    r = requests.post(url, json=payload, timeout=45)
    if r.status_code == 200:
        data = r.json()
        if data.get('candidates'):
            text = data['candidates'][0]['content']['parts'][0]['text']
            return {'text': text, 'confidence': 0.99, 'engine': 'Gemini 2.0 Flash'}
    # Fallback to 3.6 flash
    url2 = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={GEMINI_KEY}'
    r2 = requests.post(url2, json=payload, timeout=45)
    if r2.status_code == 200:
        data2 = r2.json()
        if data2.get('candidates'):
            text2 = data2['candidates'][0]['content']['parts'][0]['text']
            return {'text': text2, 'confidence': 0.99, 'engine': 'Gemini 3.6 Flash'}
    return {'text': '', 'confidence': 0, 'engine': 'none', 'error': f'Gemini errors: {r.status_code}, {r2.status_code}'}

@app.get("/")
async def root():
    return {"name": "OCR API v4.0", "engine": "Gemini Vision", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "key_set": bool(GEMINI_KEY)}

@app.post("/ocr/url")
async def ocr_url(url: str = Query(...), language: str = Query(default="en")):
    try:
        resp = requests.get(url, timeout=15)
        image = Image.open(io.BytesIO(resp.content))
        result = gemini_ocr(image)
        result['url'] = url
        result['language'] = language
        result['size'] = [image.width, image.height]
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr")
async def ocr_upload(file: UploadFile = File(...), language: str = Query(default="en")):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = gemini_ocr(image)
        result['filename'] = file.filename
        result['language'] = language
        result['size'] = [image.width, image.height]
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr/base64")
async def ocr_b64(image: str = Query(...), language: str = Query(default="en")):
    try:
        if ',' in image:
            image = image.split(',')[1]
        image_data = base64.b64decode(image)
        image = Image.open(io.BytesIO(image_data))
        result = gemini_ocr(image)
        result['language'] = language
        result['size'] = [image.width, image.height]
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
