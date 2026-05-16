# -*- coding: utf-8 -*-
import sys, io, fitz

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True, write_through=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True, write_through=True)

import os, base64, threading, subprocess, tempfile
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from PIL import Image as PILImage

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

app = Flask(__name__)
CORS(app)

# =============================================================================
# PYTHON ENVIRONMENT GUARD
# =============================================================================
if sys.version_info[:2] not in ((3, 10), (3, 11), (3, 12)):
    print(f"  WARNING: Running Python {'.'.join(map(str, sys.version_info[:2]))}")
    print(f"  Recommended: Python 3.10 or 3.11")


# =============================================================================
# CNN CHARACTER MODEL (Draw tab)
# =============================================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "character_model_tf210.h5")
EMNIST_BALANCED_MAPPING = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'a','b','d','e','f','g','h','n','q','r','t'
]
cnn_model = None

def load_cnn_model():
    global cnn_model
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        import tensorflow as tf
        cnn_model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[CNN] Loaded. Input: {cnn_model.input_shape}")
        return True
    except Exception as e:
        print(f"[CNN] Error: {e}")
        return False


# =============================================================================
# ENGINE 1 - GEMINI VISION API (highest accuracy - free tier)
# Get a free API key at: https://aistudio.google.com/apikey
# 1500 free requests/day, no credit card required
# =============================================================================
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '').strip()
GROQ_API_KEY   = os.environ.get('GROQ_API_KEY', '').strip()
gemini_status  = "ready" if GEMINI_API_KEY else "no_key"
# =============================================================================
# ENGINE 2 - OLLAMA LOCAL LLM  (offline, no API key, best local accuracy)
# Recommended model: qwen2.5vl:7b  (~4.7 GB, best for document OCR)
# Setup:  1. Install Ollama from https://ollama.com/download
#         2. Run in terminal:  ollama pull qwen2.5vl:7b
# =============================================================================
OLLAMA_URL   = os.environ.get('OLLAMA_URL',   'http://127.0.0.1:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen2.5vl:7b')


def _crop_to_text_region(pil_img, margin=20):
    """
    Crop the image to only include rows that contain handwritten ink.
    Removes the large blank ruled-line area at the bottom of notebook pages.
    This prevents LLaVA from seeing empty space and hallucinating text to fill it.
    """
    img_np = np.array(pil_img.convert('RGB'))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Detect dark ink pixels (handwriting is darker than paper/ruled lines)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove horizontal ruled lines (70% width kernel)
    w = img_np.shape[1]
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, int(w*0.70)), 1))
    ruled = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk, iterations=1)
    ek = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    text_only = cv2.subtract(binary, cv2.dilate(ruled, ek, iterations=1))

    # Find the last row that contains ink
    row_sums = np.sum(text_only, axis=1)
    ink_rows = np.where(row_sums > gray.shape[1] * 0.5)[0]

    if len(ink_rows) == 0:
        return pil_img  # no ink found, return original

    first_row = max(0, ink_rows[0] - margin)
    last_row  = min(img_np.shape[0], ink_rows[-1] + margin)

    cropped = img_np[first_row:last_row, :]
    print(f"[Ollama] Cropped image to text region: rows {first_row}-{last_row} (was {img_np.shape[0]}px tall)")
    return PILImage.fromarray(cropped)

def ollama_available():
    """Returns True if Ollama is running and the vision model is pulled."""
    try:
        import urllib.request, json
        with urllib.request.urlopen(OLLAMA_URL + '/api/tags', timeout=3) as r:
            data   = json.loads(r.read())
            models = [m['name'] for m in data.get('models', [])]
            base   = OLLAMA_MODEL.split(':')[0]
            return any(base in m for m in models)
    except Exception:
        return False

def ollama_ocr(pil_img):
    """
    Send image to Ollama qwen2.5vl:7b for strict handwriting transcription.
    Uses the /api/chat endpoint with vision support.
    """
    import urllib.request, json, urllib.error

    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    strict_prompt = (
        "Transcribe ONLY the handwritten text visible in this image. "
        "Copy each word exactly as written, line by line from top to bottom. "
        "IMPORTANT: Do NOT add any text that is not in the image. "
        "Do NOT use your knowledge of stories or books. "
        "Do NOT continue or complete any sentences. "
        "When you reach the last written word, STOP immediately."
    )

    # Try /api/chat first (works better with qwen2.5vl)
    chat_payload = json.dumps({
        "model":    OLLAMA_MODEL,
        "messages": [{
            "role":    "user",
            "content": strict_prompt,
            "images":  [img_b64]
        }],
        "stream":   False,
        "options":  {
            "temperature":    0.0,    # zero = fully deterministic, no creativity
            "num_predict":    1024,
            "repeat_penalty": 1.3     # strong penalty against repeating lines
        }
    }).encode('utf-8')

    try:
        req = urllib.request.Request(
            OLLAMA_URL + '/api/chat',
            data=chat_payload,
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            result  = json.loads(resp.read())
            text    = result.get('message', {}).get('content', '').strip()
            if text:
                return text
    except Exception as e:
        print(f"[Ollama] /api/chat failed ({e}), trying /api/generate...")

    # Fallback to /api/generate
    gen_payload = json.dumps({
        "model":   OLLAMA_MODEL,
        "prompt":  strict_prompt,
        "images":  [img_b64],
        "stream":  False,
        "options": {"temperature": 0.0, "num_predict": 1024, "repeat_penalty": 1.3}
    }).encode('utf-8')

    req = urllib.request.Request(
        OLLAMA_URL + '/api/generate',
        data=gen_payload,
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())['response'].strip()



def gemini_ocr(pil_img):
    """
    Use Google Gemini 2.0 Flash to read handwritten text.
    Tries with retry on rate-limit (429).
    """
    import urllib.request, urllib.error, json, time

    # Save as JPEG to reduce payload size (PNG was too large)
    buf = io.BytesIO()
    pil_img.convert('RGB').save(buf, format='JPEG', quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Use Gemini 2.5 Flash - it has a 1500 request/day free limit
    models = ['gemini-2.5-flash']

    for model in models:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={GEMINI_API_KEY}")
        payload = json.dumps({
            "contents": [{
                "parts": [
                    {"text": (
                        "Read ALL the handwritten text in this image exactly "
                        "as written, line by line. Output ONLY the transcribed "
                        "text. No explanations or commentary."
                    )},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                ]
            }],
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048}
        }).encode('utf-8')

        req = urllib.request.Request(
            url, data=payload, headers={'Content-Type': 'application/json'})

        for attempt in range(3):   # retry up to 3x on 429
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode('utf-8'))
                    try:
                        text = result['candidates'][0]['content']['parts'][0]['text'].strip()
                        print(f"[Gemini] OK with {model}")
                        return text
                    except (KeyError, IndexError):
                        # E.g. finishReason == 'SAFETY' and no parts returned
                        print(f"[Gemini] Blocked or empty response from {model}")
                        return "[Extracted content was blocked by API safety filters or no text was found]"
            except urllib.error.HTTPError as e:
                body = e.read().decode('utf-8', errors='replace')
                print(f"[Gemini] {model} attempt {attempt+1}: HTTP {e.code} - {body[:120]}")
                if e.code == 429:
                    wait = 10 * (attempt + 1)
                    print(f"[Gemini] Rate limited - waiting {wait}s...")
                    time.sleep(wait)
                else:
                    break   # not a rate limit - skip to next model
            except Exception as e:
                print(f"[Gemini] {model} error: {e}")
                break

    raise RuntimeError("All Gemini models failed - check API key and quota")

def gemini_ocr_multi(pil_images):
    """
    Use Google Gemini 1.5 Flash to read handwritten text from multiple pages in ONE request.
    This bypasses the 15 RPM rate limit and processes documents significantly faster.
    """
    import urllib.request, urllib.error, json, time

    parts = [{"text": "Read ALL the handwritten text in these document pages exactly as written, line by line. Output ONLY the transcribed text. Separate the transcription of each page with exactly this marker on its own line: ---PAGE_BREAK---"}]
    
    for img in pil_images:
        buf = io.BytesIO()
        img.convert('RGB').save(buf, format='JPEG', quality=90)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})

    model = 'gemini-2.5-flash'
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    payload = json.dumps({
        "contents": [{"parts": parts}],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192}
    }).encode('utf-8')

    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                try:
                    text = result['candidates'][0]['content']['parts'][0]['text'].strip()
                    print(f"[Gemini] OK multi-page processing")
                    # Split by page marker
                    pages = [p.strip() for p in text.split('---PAGE_BREAK---')]
                    # Handle case where Gemini returns fewer splits than images
                    while len(pages) < len(pil_images):
                        pages.append("[Error: Page missing in transcription]")
                    return pages[:len(pil_images)]
                except (KeyError, IndexError):
                    print(f"[Gemini] Blocked or empty response")
                    return ["[Extracted content was blocked by API safety filters or no text was found]"] * len(pil_images)
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', errors='replace')
            print(f"[Gemini] multi attempt {attempt+1}: HTTP {e.code} - {body[:120]}")
            if e.code == 429:
                wait = 10 * (attempt + 1)
                time.sleep(wait)
            else:
                break
        except Exception as e:
            print(f"[Gemini] multi error: {e}")
            break

    raise RuntimeError("Gemini multi-page processing failed")

def groq_ocr(pil_img):
    """
    Use Groq Llama-3.2-90b-vision-preview to read handwritten text.
    """
    import urllib.request, urllib.error, json, time

    buf = io.BytesIO()
    pil_img.convert('RGB').save(buf, format='JPEG', quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = json.dumps({
        "model": "llama-3.2-90b-vision-preview",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Read ALL the handwritten text in this image exactly as written, line by line. Output ONLY the transcribed text. No explanations or commentary."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }],
        "temperature": 0.1,
        "max_tokens": 2048
    }).encode('utf-8')

    req = urllib.request.Request(url, data=payload, headers={
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'User-Agent': 'Mozilla/5.0'
    })

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                text = result['choices'][0]['message']['content'].strip()
                print("[Groq] OK")
                return text
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', errors='replace')
            print(f"[Groq] attempt {attempt+1}: HTTP {e.code} - {body[:120]}")
            if e.code == 429:
                wait = 10 * (attempt + 1)
                print(f"[Groq] Rate limited - waiting {wait}s...")
                time.sleep(wait)
            else:
                break
        except Exception as e:
            print(f"[Groq] error: {e}")
            break

    raise RuntimeError("Groq API failed - check API key and quota")

# =============================================================================
# ENGINE 2 - EasyOCR (offline, no API key, handles own text detection)
# =============================================================================
easyocr_reader = None
easyocr_status = "idle"
easyocr_error  = ""

def _load_easyocr():
    global easyocr_reader, easyocr_status, easyocr_error
    try:
        easyocr_status = "loading"
        import easyocr
        easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        easyocr_status = "ready"
        print("[EasyOCR] Ready.")
    except Exception as e:
        easyocr_status = "error"
        easyocr_error  = str(e)
        print(f"[EasyOCR] Failed: {e}")

threading.Thread(target=_load_easyocr, daemon=True).start()


def easyocr_document(pil_img):
    """Run EasyOCR on a full document image, return (text, boxes)."""
    img_np  = preprocess_for_ocr(pil_img)
    results = easyocr_reader.readtext(np.array(img_np), detail=1, paragraph=False)
    if not results:
        return "", []

    # Sort top-to-bottom, group into lines by y-proximity
    results.sort(key=lambda r: r[0][0][1])
    lines, cur, last_y = [], [], None
    for bbox, text, conf in results:
        y = bbox[0][1]
        if last_y is None or abs(y - last_y) < 50:
            cur.append((bbox, text, conf))
        else:
            if cur: lines.append(cur)
            cur = [(bbox, text, conf)]
        last_y = y
    if cur: lines.append(cur)

    text_lines = []
    for line in lines:
        line.sort(key=lambda r: r[0][0][0])
        text_lines.append(' '.join(r[1] for r in line))

    full_text = '\n'.join(text_lines)
    boxes = []
    for r in results:
        # Cast numpy types to standard python int/float for JSON serialization
        conf = float(r[2])
        bbox = [[int(pt[0]), int(pt[1])] for pt in r[0]]
        boxes.append({'text': r[1], 'confidence': int(conf * 100), 'bbox': bbox})

    return full_text, boxes


# =============================================================================
# ENGINE 3 - TrOCR large (offline, line-by-line)
# =============================================================================
trocr_processor = None
trocr_model     = None
trocr_status    = "idle"
trocr_error_msg = ""
TROCR_MODEL_ID  = "microsoft/trocr-large-handwritten"

def _load_trocr():
    global trocr_processor, trocr_model, trocr_status, trocr_error_msg
    try:
        trocr_status = "loading"
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
        trocr_model     = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_ID)
        trocr_model.eval()
        trocr_status = "ready"
        print("[TrOCR] Ready.")
    except Exception as e:
        trocr_status    = "error"
        trocr_error_msg = str(e)
        print(f"[TrOCR] Failed: {e}")

threading.Thread(target=_load_trocr, daemon=True).start()


def _crop_left_margin(img_np):
    h, w = img_np.shape[:2]
    sw   = max(1, int(w * 0.12))
    reg  = img_np[:, :sw]
    r, g, b = reg[:,:,0].astype(int), reg[:,:,1].astype(int), reg[:,:,2].astype(int)
    mask = (r > 140) & (r - g > 60) & (r - b > 60)
    col  = mask.sum(axis=0)
    if col.max() > h * 0.25:
        return img_np[:, int(np.argmax(col)) + 8:]
    return img_np

def _ink_ratio(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    return np.sum(g < 160) / g.size

def _remove_ruled_lines(binary, img_w):
    ml   = max(40, int(img_w * 0.70))
    hk   = cv2.getStructuringElement(cv2.MORPH_RECT, (ml, 1))
    rl   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk, iterations=1)
    ek   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    return cv2.subtract(binary, cv2.dilate(rl, ek, iterations=1))

def segment_lines(pil_img, min_h=20, pad=10):
    img = np.array(pil_img.convert('RGB'))
    img = _crop_left_margin(img)
    g   = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    if max(h, w) < 1600:
        s = 1600 / max(h, w)
        g   = cv2.resize(g,   (int(w*s), int(h*s)), interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g     = clahe.apply(g)
    _, bn = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bt    = _remove_ruled_lines(bn, img.shape[1])
    vk    = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
    bt    = cv2.dilate(bt, vk, iterations=1)
    proj  = np.sum(bt, axis=1).astype(np.float32)
    k     = max(3, g.shape[0] // 60)
    if k % 2 == 0: k += 1
    ps    = cv2.GaussianBlur(proj.reshape(-1,1),(1,k),0).flatten()
    thr   = max(1.0, ps.max() * 0.05)
    in_b, bands, st = False, [], 0
    for i, v in enumerate(ps):
        if not in_b and v > thr: in_b, st = True, i
        elif in_b and v <= thr:
            in_b = False
            if i - st >= min_h: bands.append((st, i))
    if in_b and len(ps)-st >= min_h: bands.append((st, len(ps)))
    lines, ih = [], img.shape[0]
    for y0, y1 in bands:
        c = img[max(0,y0-pad):min(ih,y1+pad), :]
        if c.shape[0] > 5 and _ink_ratio(c) >= 0.010:
            lines.append(PILImage.fromarray(c))
    print(f"[TrOCR seg] {len(bands)} bands -> {len(lines)} lines")
    return lines if lines else [pil_img]

def trocr_predict_line(line_pil):
    import torch
    rgb = np.array(line_pil.convert('RGB'))
    rgb = cv2.fastNlMeansDenoisingColored(rgb, None, 5, 5, 7, 21)
    h, w = rgb.shape[:2]
    if h < 48:
        rgb = cv2.resize(rgb, (int(w*(48/h)), 48), cv2.INTER_CUBIC)
    pv = trocr_processor(images=PILImage.fromarray(rgb), return_tensors='pt').pixel_values
    with __import__('torch').no_grad():
        ids = trocr_model.generate(pv, max_new_tokens=128, num_beams=6,
                                   no_repeat_ngram_size=3)
    return trocr_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

def trocr_document(pil_img):
    enhanced = preprocess_for_ocr(pil_img)
    lines    = segment_lines(enhanced)
    out      = []
    for i, ln in enumerate(lines):
        try:
            t = trocr_predict_line(ln).lstrip('#|!').strip()
            if t: out.append(t)
            print(f"[TrOCR] L{i+1}: {t[:70]}")
        except Exception as e:
            print(f"[TrOCR] L{i+1} err: {e}")
    return '\n'.join(out), len(lines)


# =============================================================================
# ENGINE 4 - Tesseract (final fallback)
# =============================================================================
TESSERACT_CMD = os.environ.get('TESSERACT_CMD',
                               r'C:\Program Files\Tesseract-OCR\tesseract.exe')

def tesseract_available():
    if not os.path.exists(TESSERACT_CMD): return False
    try:
        r = subprocess.run([TESSERACT_CMD,'--version'], capture_output=True,
                           text=True, timeout=5)
        return r.returncode == 0
    except: return False

def tesseract_ocr(pil_image, psm=6):
    with tempfile.TemporaryDirectory() as d:
        ip   = os.path.join(d,'in.png')
        ob   = os.path.join(d,'out')
        tp   = os.path.join(d,'out.tsv')
        pil_image.save(ip)
        txt  = subprocess.run([TESSERACT_CMD,ip,'stdout','-l','eng','--oem','3','--psm',str(psm)],
                              capture_output=True, text=True, timeout=60).stdout.strip()
        subprocess.run([TESSERACT_CMD,ip,ob,'-l','eng','--oem','3','--psm',str(psm),'tsv'],
                       capture_output=True, timeout=60)
        blocks = []
        if os.path.exists(tp):
            for line in open(tp, encoding='utf-8').readlines()[1:]:
                p = line.strip().split('\t')
                if len(p) < 12: continue
                conf, word = p[10], p[11].strip()
                if not word or conf == '-1': continue
                try:
                    c=int(conf); x,y,w,h=int(p[6]),int(p[7]),int(p[8]),int(p[9])
                except: continue
                if c>0: blocks.append({'text':word,'confidence':c,
                                       'bbox':[[x,y],[x+w,y],[x+w,y+h],[x,y+h]]})
    return txt, blocks


# =============================================================================
# SHARED IMAGE PREPROCESSING
# =============================================================================
def preprocess_for_ocr(pil_img):
    img  = np.array(pil_img.convert('RGB'))
    h, w = img.shape[:2]
    if max(h,w) < 1600:
        s   = 1600/max(h,w)
        img = cv2.resize(img, (int(w*s),int(h*s)), interpolation=cv2.INTER_CUBIC)
    lab       = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b   = cv2.split(lab)
    clahe     = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    img       = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)
    img       = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    return PILImage.fromarray(img)


# =============================================================================
# FLASK ROUTES
# =============================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/model-status')
def model_status():
    ollama_ready = ollama_available()
    if GEMINI_API_KEY:
        active = "ready"; engine = "Gemini Vision (Google AI)"; model = "gemini-2.5-flash"
    elif ollama_ready:
        active = "ready"; engine = f"Ollama Local LLM ({OLLAMA_MODEL})"; model = OLLAMA_MODEL
    elif easyocr_status == "ready":
        active = "ready"; engine = "EasyOCR (offline)"; model = "EasyOCR 1.7"
    elif trocr_status == "ready":
        active = "ready"; engine = "TrOCR Large (offline)"; model = TROCR_MODEL_ID
    elif easyocr_status == "loading" or trocr_status == "loading":
        active = "loading"; engine = "Loading..."; model = ""
    else:
        active = "error"; engine = "None"; model = ""
    return jsonify({
        'trocr_status':        active,
        'trocr_error':         easyocr_error or trocr_error_msg,
        'trocr_model':         model,
        'engine_name':         engine,
        'gemini_key_set':      bool(GEMINI_API_KEY),
        'easyocr_status':      easyocr_status,
        'trocr_secondary':     trocr_status,
        'tesseract_available': tesseract_available()
    })


@app.route('/predict', methods=['POST'])
def predict():
    global cnn_model
    if cnn_model is None and not load_cnn_model():
        return jsonify({'error': 'CNN model unavailable.'}), 503
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image.'}), 400
    processed = _preprocess_canvas(data['image'])
    if processed is None or np.max(processed) < 0.05:
        return jsonify({'error': 'Canvas empty.'}), 400
    try:
        preds = cnn_model.predict(processed, verbose=0)
        idx   = int(np.argmax(preds[0]))
        conf  = float(preds[0][idx])
        top5  = np.argsort(preds[0])[::-1][:5]
        return jsonify({'character': EMNIST_BALANCED_MAPPING[idx],
                        'confidence': round(conf*100, 2),
                        'alternatives': [{'character': EMNIST_BALANCED_MAPPING[i],
                                          'confidence': float(preds[0][i])} for i in top5[1:]]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _preprocess_canvas(b64):
    if ',' in b64: b64 = b64.split(',')[1]
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None: return None
    if img.ndim == 3 and img.shape[2] == 4:
        a   = img[:,:,3]; bgr = img[:,:,:3]
        img = (bgr*(a[:,:,np.newaxis]/255.) + np.ones_like(bgr)*255*(1-a[:,:,np.newaxis]/255.)).astype(np.uint8)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, bn = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY_INV)
    cs, _ = cv2.findContours(bn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return None
    x,y,w,h = cv2.boundingRect(np.concatenate(cs, 0))
    cr = bn[y:y+h, x:x+w]
    
    # Maintain aspect ratio by resizing such that max_dim = 20
    max_dim = max(w, h)
    scale = 20.0 / max_dim
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    # Prevent dimension from being 0 in extreme cases
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    resized = cv2.resize(cr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Calculate center of mass
    M = cv2.moments(resized)
    if M["m00"] != 0:
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
    else:
        cX, cY = new_w / 2.0, new_h / 2.0
        
    # Translate image so that center of mass is at (14.0, 14.0) in a 28x28 image
    dx = 14.0 - cX
    dy = 14.0 - cY
    M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
    padded = cv2.warpAffine(resized, M_trans, (28, 28), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return (padded.astype(np.float32)/255.).reshape(1,28,28,1)


def process_single_image(pil_img, engine_pref='gemini'):
    order = ['gemini', 'groq'] if engine_pref == 'gemini' else ['groq', 'gemini']
    
    for engine in order:
        if engine == 'groq' and GROQ_API_KEY:
            try:
                print("[Groq] Running OCR...")
                text = groq_ocr(pil_img)
                return {'text': text, 'engine': 'Groq (Llama-3.2 Vision)', 'engine_key': 'groq'}
            except Exception as e:
                print(f"[Groq] Error: {e} -- falling through")
                
        elif engine == 'gemini' and GEMINI_API_KEY:
            try:
                print("[Gemini] Running OCR...")
                text = gemini_ocr(pil_img)
                return {'text': text, 'engine': 'Gemini Vision (Google AI)', 'engine_key': 'gemini'}
            except Exception as e:
                print(f"[Gemini] Error: {e} -- falling through")

    # ENGINE 3: Ollama Local LLM (best offline accuracy)
    if ollama_available():
        try:
            print(f"[Ollama] Running OCR with {OLLAMA_MODEL}...")
            cropped = _crop_to_text_region(pil_img)
            text    = ollama_ocr(cropped)
            return {'text': text, 'engine': f'Ollama ({OLLAMA_MODEL})', 'engine_key': 'ollama'}
        except Exception as e:
            print(f"[Ollama] Error: {e} -- falling through")

    # ENGINE 3: EasyOCR
    if easyocr_status == "ready":
        try:
            print("[EasyOCR] Running OCR...")
            text, boxes = easyocr_document(pil_img)
            return {'text': text, 'blocks': boxes, 'engine': 'EasyOCR (offline)', 'engine_key': 'easyocr'}
        except Exception as e:
            print(f"[EasyOCR] Error: {e} -- falling through")

    # ENGINE 4: TrOCR
    if trocr_status == "ready":
        try:
            text, _ = trocr_document(pil_img)
            return {'text': text, 'engine': 'TrOCR (microsoft/trocr-large-handwritten)', 'engine_key': 'trocr'}
        except Exception as e:
            print(f"[TrOCR] Error: {e} -- falling through")

    if easyocr_status == "loading" or trocr_status == "loading":
        return {'error': 'Model still loading. Please wait.', 'model_loading': True}

    # ENGINE 5: Tesseract
    if tesseract_available():
        try:
            img_np = np.array(pil_img)
            h, w   = img_np.shape[:2]
            if max(h,w) < 1200:
                s      = 1200/max(h,w)
                img_np = cv2.resize(img_np,(int(w*s),int(h*s)),cv2.INTER_CUBIC)
            g     = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(3.0,(8,8))
            g     = clahe.apply(g)
            g     = cv2.fastNlMeansDenoising(g, h=10)
            _, ot = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            at    = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,10)
            bt, bb = '', []
            for thr, psm in [(ot,6),(at,6),(ot,11)]:
                t, b = tesseract_ocr(PILImage.fromarray(thr), psm=psm)
                if len(t) > len(bt): bt, bb = t, b
            return {'text': bt, 'blocks': bb, 'engine': 'Tesseract (fallback)', 'engine_key': 'tesseract'}
        except Exception as e:
            print(f"[Tesseract] Error: {e} -- falling through")

    return {'error': 'No OCR engine available.'}


@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    file = request.files['image']
    ext  = os.path.splitext(file.filename.lower())[1]
    
    if ext not in {'.png','.jpg','.jpeg','.webp','.bmp','.tiff','.tif', '.pdf'}:
        return jsonify({'error': f'Unsupported type: {ext}'}), 400
        
    pil_images = []
    if ext == '.pdf':
        try:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Zoom for better quality
                img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pil_images.append(img)
            doc.close()
        except Exception as e:
            return jsonify({'error': f'Cannot parse PDF: {e}'}), 400
    else:
        try:
            pil_images.append(PILImage.open(io.BytesIO(file.read())).convert('RGB'))
        except Exception as e:
            return jsonify({'error': f'Cannot open image: {e}'}), 400

    if not pil_images:
        return jsonify({'error': 'No pages found in document.'}), 400

    engine_pref = request.form.get('engine_pref', 'gemini')

    # FAST BATCH PROCESSING FOR GEMINI (Avoids rate-limit timeouts for multi-page documents)
    if engine_pref == 'gemini' and GEMINI_API_KEY and len(pil_images) > 0:
        try:
            print(f"Processing ALL {len(pil_images)} pages in one request via Gemini...")
            pages_text = gemini_ocr_multi(pil_images)
            pages_result = []
            total_lines = 0
            for i, text in enumerate(pages_text):
                lc = text.count('\n') + 1 if text.strip() else 0
                total_lines += lc
                pages_result.append({'page_num': i+1, 'text': text, 'blocks': []})
            return jsonify({
                'pages': pages_result,
                'line_count': int(total_lines),
                'engine': 'Gemini Vision (Google AI)',
                'engine_key': 'gemini'
            })
        except Exception as e:
            print(f"[Gemini Multi] Failed: {e}. Falling back to single-page loop...")

    pages_result = []
    primary_engine = "Unknown"
    primary_engine_key = "unknown"
    total_lines = 0

    for i, pil_img in enumerate(pil_images):
        print(f"Processing page {i+1}/{len(pil_images)}")
        res = process_single_image(pil_img, engine_pref)
        if 'error' in res and len(pil_images) == 1:
            if res.get('model_loading'):
                return jsonify(res), 503
            return jsonify(res), 500
        elif 'error' in res:
            # Fallback for multi-page if one page fails
            text = f"[Error processing page {i+1}]"
        else:
            text = res.get('text', '')
            
        lc = text.count('\n') + 1 if text.strip() else 0
        total_lines += lc
        
        primary_engine = res.get('engine', primary_engine)
        primary_engine_key = res.get('engine_key', primary_engine_key)
        
        pages_result.append({
            'page_num': i + 1,
            'text': text,
            'blocks': res.get('blocks', [])
        })

    return jsonify({
        'pages': pages_result,
        'line_count': int(total_lines),
        'engine': primary_engine,
        'engine_key': primary_engine_key
    })


@app.route('/download/docx', methods=['POST'])
def download_docx():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text.'}), 400
    try:
        from docx import Document as DocxDocument
        doc = DocxDocument()
        doc.add_heading('Extracted Handwritten Text', 0)
        for line in data['text'].split('\n'):
            doc.add_paragraph(line)
        buf = io.BytesIO(); doc.save(buf); buf.seek(0)
        return send_file(buf, as_attachment=True,
                         download_name='extracted_text.docx',
                         mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
    except ImportError:
        return jsonify({'error': 'Run: pip install python-docx'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    if GEMINI_API_KEY:
        print("[Gemini] API key found -- using Gemini Vision as primary engine")
    else:
        print("[Gemini] No API key -- using EasyOCR/TrOCR (offline)")
        print("         To get ~95% accuracy: set GEMINI_API_KEY env variable")
        print("         Free key at: https://aistudio.google.com/apikey")
    port = int(os.environ.get('PORT', 8008))
    app.run(host='0.0.0.0', port=port, debug=False)
