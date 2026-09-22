import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, r'C:\Users\Welcome\Downloads\paddleocr-api')
from main import gemini_vision_ocr
from PIL import Image
import requests, time

# Image 1 - Retry
for attempt in range(3):
    print(f"=== IMAGE 1 - Attempt {attempt+1} ===")
    url1 = 'https://i.ibb.co/xnd4rPn/Screenshot-2026-09-20-072144.png'
    resp1 = requests.get(url1, timeout=15)
    img1 = Image.open(io.BytesIO(resp1.content))
    result1 = gemini_vision_ocr(img1)
    if result1:
        print("Engine:", result1['engine'])
        print("Confidence:", result1['confidence'])
        print("Words:", result1['word_count'])
        print(result1['text'])
        break
    else:
        print("FAILED - Retrying in 3s...")
        time.sleep(3)

print()
print("=== IMAGE 2 - Gemini Vision OCR ===")
url2 = 'https://i.ibb.co/JWKM2pL2/Screenshot-2026-09-20-072907.png'
resp2 = requests.get(url2, timeout=15)
img2 = Image.open(io.BytesIO(resp2.content))
result2 = gemini_vision_ocr(img2)
if result2:
    print("Engine:", result2['engine'])
    print("Confidence:", result2['confidence'])
    print("Words:", result2['word_count'])
    print(result2['text'])
else:
    print("FAILED")
