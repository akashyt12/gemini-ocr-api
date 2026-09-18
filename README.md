# PaddleOCR API - Railway Deployment 🚀

Advanced OCR API with PaddleOCR - Host on Railway for FREE!

## Features

- ✅ **Multi-language OCR** (80+ languages)
- ✅ **Image Enhancement** (Auto-denoise, sharpen)
- ✅ **Batch Processing** (Multiple images)
- ✅ **Base64 Support** (No file upload needed)
- ✅ **URL Support** (Extract from image URL)
- ✅ **High Accuracy** (PP-OCRv5)
- ✅ **Fast Response** (Optimized inference)

## API Endpoints

### 1. OCR from File
```bash
POST /ocr
Content-Type: multipart/form-data

curl -X POST "https://your-app.up.railway.app/ocr" \
  -F "file=@image.png" \
  -F "language=en" \
  -F "enhance=true"
```

### 2. OCR from Base64
```bash
POST /ocr/base64
Content-Type: application/json

curl -X POST "https://your-app.up.railway.app/ocr/base64" \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_string_here", "language": "en"}'
```

### 3. OCR from URL
```bash
POST /ocr/url
Content-Type: application/json

curl -X POST "https://your-app.up.railway.app/ocr/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.png", "language": "en"}'
```

### 4. Batch OCR
```bash
POST /ocr/batch
Content-Type: multipart/form-data

curl -X POST "https://your-app.up.railway.app/ocr/batch" \
  -F "files=@image1.png" \
  -F "files=@image2.png" \
  -F "language=en"
```

### 5. Health Check
```bash
GET /health

curl "https://your-app.up.railway.app/health"
```

## Response Format

```json
{
  "text": "Extracted text from image",
  "words": [
    {
      "text": "Word",
      "confidence": 0.9876,
      "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }
  ],
  "confidence": 0.9876,
  "word_count": 15,
  "filename": "image.png",
  "language": "en"
}
```

## Deployment to Railway

### Step 1: Install Railway CLI
```bash
npm install -g @railway/cli
```

### Step 2: Login
```bash
railway login
```

### Step 3: Create Project
```bash
railway init
```

### Step 4: Deploy
```bash
railway up
```

### Step 5: Get URL
```bash
railway domain
```

## Supported Languages

| Code | Language |
|------|----------|
| en | English |
| hi | Hindi |
| ch | Chinese |
| ja | Japanese |
| ko | Korean |
| fr | French |
| de | German |
| es | Spanish |
| and 80+ more... | |

## Usage Examples

### Python
```python
import requests

# OCR from file
with open('image.png', 'rb') as f:
    response = requests.post(
        'https://your-app.up.railway.app/ocr',
        files={'file': f},
        params={'language': 'en', 'enhance': True}
    )
print(response.json())

# OCR from URL
response = requests.post(
    'https://your-app.up.railway.app/ocr/url',
    json={'url': 'https://example.com/image.png'}
)
print(response.json())
```

### JavaScript
```javascript
// OCR from file
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('https://your-app.up.railway.app/ocr', {
    method: 'POST',
    body: formData
});
const result = await response.json();
console.log(result);
```

## Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| PORT | 8000 | Server port |
| WORKERS | 1 | Number of workers |

## License

MIT
