"""
Pneumonia Detection — Flask Backend (v5 — ML X-ray Validator)
==============================================================
Run:  python app.py
Then open: http://localhost:5000

Pipeline (per request):
  Step 1 → ml_is_valid_xray(img)   — MobileNetV2 binary classifier
                                      (xray=0 vs not_xray=1)
  Step 2 → classify(img)           — only if Step 1 passes
  Step 3 → gradcam.generate()      — only if Bacterial or Viral Pneumonia

Required files in same folder as app.py:
  densenet121_pneumonia.pth   ← pneumonia classifier
  model_metadata.json         ← class names + normalization stats
  xray_validator.pth          ← ML X-ray validator (train on Kaggle first)
"""

import os, json, io, base64, time, uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageStat
import cv2

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_PATH     = Path("densenet121_pneumonia.pth")
METADATA_PATH  = Path("model_metadata.json")
VALIDATOR_PATH = Path("xray_validator.pth")
HISTORY_FILE   = Path("session_history.json")
IMG_SIZE       = 224
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Confidence threshold for validator:
# If ML says P(not_xray) > this, reject.
# 0.80 means "reject only if 80%+ confident it's NOT an X-ray"
# Raise to 0.90 to be more permissive, lower to 0.70 to be stricter.
VALIDATOR_THRESHOLD = 0.80

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ─────────────────────────────────────────────────────────────
#  LOAD PNEUMONIA CLASSIFIER (DenseNet-121)
# ─────────────────────────────────────────────────────────────
"""
Pneumonia Detection — Flask Backend (v5 — ML X-ray Validator)
==============================================================
Run:  python app.py
Then open: http://localhost:5000

Pipeline (per request):
  Step 1 → ml_is_valid_xray(img)   — MobileNetV2 binary classifier
                                      (xray=0 vs not_xray=1)
  Step 2 → classify(img)           — only if Step 1 passes
  Step 3 → gradcam.generate()      — only if Bacterial or Viral Pneumonia

Required files in same folder as app.py:
  densenet121_pneumonia.pth   ← pneumonia classifier
  model_metadata.json         ← class names + normalization stats
  xray_validator.pth          ← ML X-ray validator (train on Kaggle first)
"""

import os, json, io, base64, time, uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageStat
import cv2

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_PATH     = Path("densenet121_pneumonia.pth")
METADATA_PATH  = Path("model_metadata.json")
VALIDATOR_PATH = Path("xray_validator.pth")
HISTORY_FILE   = Path("session_history.json")
IMG_SIZE       = 224
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Confidence threshold for validator:
# If ML says P(not_xray) > this, reject.
# 0.80 means "reject only if 80%+ confident it's NOT an X-ray"
# Raise to 0.90 to be more permissive, lower to 0.70 to be stricter.
VALIDATOR_THRESHOLD = 0.80

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ─────────────────────────────────────────────────────────────
#  LOAD PNEUMONIA CLASSIFIER (DenseNet-121)
# ─────────────────────────────────────────────────────────────
def build_pneumonia_model():
    m = models.densenet121(weights=None)
    m.classifier = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 3)
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # Fix DataParallel issue
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k.replace("module.", "")] = v
        else:
            new_state_dict[k] = v   # ✅ FIXED indentation

    m.load_state_dict(new_state_dict)  # ✅ load cleaned weights
    m.eval()
    return m.to(DEVICE)

with open(METADATA_PATH) as f:
    META = json.load(f)

CLASS_NAMES = META["class_names"]

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(META["imagenet_mean"], META["imagenet_std"]),
])

print(f"Loading pneumonia model on {DEVICE}...")
model = build_pneumonia_model()
print("Pneumonia model ready.")

# ─────────────────────────────────────────────────────────────
#  LOAD ML X-RAY VALIDATOR (MobileNetV2)
# ─────────────────────────────────────────────────────────────
#
#  Architecture must match exactly what was trained in the Kaggle notebook:
#    MobileNetV2 backbone + custom head: 1280 → 256 → 2
#    Class 0 = xray, Class 1 = not_xray
#
VALIDATOR_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),  # match training preprocessing
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def build_validator_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 2)
    )
    m.load_state_dict(torch.load(VALIDATOR_PATH, map_location=DEVICE))
    m.eval()
    return m.to(DEVICE)

# Try to load ML validator — fall back to rule-based if not found
VALIDATOR_MODEL = None
if VALIDATOR_PATH.exists():
    try:
        VALIDATOR_MODEL = build_validator_model()
        print(f"ML X-ray validator loaded from {VALIDATOR_PATH}")
    except Exception as e:
        print(f"WARNING: Could not load ML validator ({e}). Using rule-based fallback.")
else:
    print(f"WARNING: {VALIDATOR_PATH} not found. Using rule-based fallback validator.")
    print("         Train it on Kaggle using xray_validator_training.ipynb")

# ─────────────────────────────────────────────────────────────
#  STEP 1A — ML VALIDATOR (primary, when model is available)
# ─────────────────────────────────────────────────────────────
def ml_is_valid_xray(pil_img: Image.Image) -> tuple:
    """
    Uses the trained MobileNetV2 binary classifier to decide
    whether the uploaded image is a chest X-ray.

    Returns:
        (True,  "valid",   confidence_score)  → proceed with classification
        (False, reason,    confidence_score)  → reject

    confidence_score = model's confidence that it IS an X-ray (0.0 – 1.0)
    """
    tensor = VALIDATOR_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = VALIDATOR_MODEL(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    p_xray     = float(probs[0])   # probability it IS an X-ray
    p_not_xray = float(probs[1])   # probability it is NOT an X-ray

    if p_not_xray >= VALIDATOR_THRESHOLD:
        return False, (
            f"This does not appear to be a chest X-ray "
            f"(model confidence: {p_not_xray*100:.1f}% not an X-ray). "
            "Please upload a valid grayscale chest X-ray image."
        ), p_xray

    return True, "valid", p_xray


# ─────────────────────────────────────────────────────────────
#  STEP 1B — RULE-BASED FALLBACK (used only if ML model missing)
# ─────────────────────────────────────────────────────────────
def rule_is_valid_xray(pil_img: Image.Image) -> tuple:
    """
    Minimal safe fallback. Only rejects images that are CLEARLY not X-rays.
    Used only when xray_validator.pth has not been trained yet.
    """
    img_rgb = pil_img.convert("RGB")
    w, h    = img_rgb.size
    img_np  = np.array(img_rgb, dtype=np.uint8)

    if w < 100 or h < 100:
        return False, f"Image too small ({w}×{h}px). Upload a full-resolution chest X-ray.", 0.0

    ratio = w / h
    if ratio < 0.3 or ratio > 3.5:
        return False, "Unusual aspect ratio. Chest X-rays are roughly square or portrait.", 0.0

    hsv      = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    mean_sat = float(hsv[:, :, 1].mean())
    if mean_sat > 60:
        return False, (
            f"Image appears to be a colour photo (saturation: {mean_sat:.0f}/255). "
            "Upload a grayscale chest X-ray."
        ), 0.0

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    mean_b = float(gray.mean())
    std_b  = float(gray.std())
    if mean_b < 5 and std_b < 5:
        return False, "Image is completely black. Upload a valid chest X-ray.", 0.0
    if mean_b > 250 and std_b < 5:
        return False, "Image is completely white/blank. Upload a valid chest X-ray.", 0.0

    return True, "valid", 0.5   # unknown confidence for rule-based


# ─────────────────────────────────────────────────────────────
#  UNIFIED VALIDATOR — routes to ML or rule-based automatically
# ─────────────────────────────────────────────────────────────
def is_valid_xray(pil_img: Image.Image) -> tuple:
    """
    Calls ML validator if available, falls back to rule-based otherwise.
    Returns (is_valid: bool, reason: str, xray_confidence: float)
    """
    if VALIDATOR_MODEL is not None:
        return ml_is_valid_xray(pil_img)
    return rule_is_valid_xray(pil_img)


# ─────────────────────────────────────────────────────────────
#  PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────
def preprocess_xray(pil_img: Image.Image) -> Image.Image:
    """
    Grayscale → RGB conversion to match DenseNet-121 training format.
    Removes colour casts and ensures all 3 channels are identical.
    """
    return pil_img.convert("L").convert("RGB")


# ─────────────────────────────────────────────────────────────
#  STEP 2 — PNEUMONIA CLASSIFIER
# ─────────────────────────────────────────────────────────────
def classify(tensor):
    """
    Runs DenseNet-121 forward pass (no gradient).
    Returns: (pred_idx, probs_list)
    """
    with torch.no_grad():
        output = model(tensor.to(DEVICE))
        probs  = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        pred   = int(probs.argmax())
    return pred, probs.tolist()


# ─────────────────────────────────────────────────────────────
#  STEP 3 — GRAD-CAM (only for Bacterial / Viral Pneumonia)
# ─────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM for DenseNet-121.

    Target layer: features.norm5 (BatchNorm after denseblock4).
    Using norm5 instead of denseblock4 directly gives cleaner,
    less scattered activations because batch-norm has already
    normalised the feature map variance.

    Only called when pred_idx is 1 (Bacterial) or 2 (Viral).
    Never called for Normal predictions or invalid images.
    """
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target = model.features.norm5    # ← norm5 for cleaner maps

        def fwd(module, inp, out):
            self.activations = out.detach()

        def bwd(module, gin, gout):
            self.gradients = gout[0].detach()

        target.register_forward_hook(fwd)
        target.register_backward_hook(bwd)

    def generate(self, tensor, pred_idx):
        self.model.eval()
        tensor = tensor.to(DEVICE).requires_grad_(True)
        output = self.model(tensor)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, pred_idx] = 1.0
        output.backward(gradient=one_hot)

        # Global average pool gradients → channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = torch.relu(cam).cpu().numpy()

        # Normalise
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


gradcam = GradCAM(model)


# ─────────────────────────────────────────────────────────────
#  IMAGE HELPERS
# ─────────────────────────────────────────────────────────────
def build_heatmap_overlay(pil_img, cam):
    """
    Builds a clean Grad-CAM heatmap overlay on the original X-ray.

    Fixes for scatter/artifact issues:
      1. Bicubic interpolation when upscaling 7x7 → 224x224 (smoother than linear)
      2. Gaussian blur after resize (removes block artifacts)
      3. Clip to 95th percentile before colormap (removes noisy outlier activations)
      4. Higher original image weight (0.6) so anatomy stays visible under heatmap
    """
    # Resize original to model input size
    img_resized = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))

    # Step 1: Upscale CAM using BICUBIC (much smoother than linear for small maps)
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_CUBIC)

    # Step 2: Gaussian blur to smooth out any remaining block artifacts
    cam_resized = cv2.GaussianBlur(cam_resized, ksize=(11, 11), sigmaX=4)

    # Step 3: Clip to 95th percentile to suppress noisy weak activations
    # This makes the heatmap focus on the strongest activation region only
    p95 = float(np.percentile(cam_resized, 95))
    cam_resized = np.clip(cam_resized, 0, p95)

    # Step 4: Re-normalise to 0-1 after clipping
    cam_resized = cam_resized / (cam_resized.max() + 1e-8)

    # Step 5: Apply JET colormap (blue→green→red, red = highest activation)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Step 6: Blend — keep more of the original (60%) so anatomy is clear
    overlay = np.uint8(0.40 * heatmap + 0.60 * img_resized)
    return Image.fromarray(overlay)


def to_b64(pil_or_array):
    """Converts PIL Image or numpy array to base64 JPEG data URI."""
    if isinstance(pil_or_array, np.ndarray):
        pil_or_array = Image.fromarray(pil_or_array)
    buf = io.BytesIO()
    pil_or_array.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────
#  HISTORY HELPERS
# ─────────────────────────────────────────────────────────────
def load_history():
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory("static/css", filename)

@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory("static/js", filename)


# ── Main prediction endpoint ──────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file         = request.files["file"]
    patient_name = request.form.get("patient_name", "Anonymous")
    patient_id   = request.form.get("patient_id") or str(uuid.uuid4())[:8].upper()

    # ── Open image ────────────────────────────────────────────
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Cannot open file. Please upload a valid image."}), 400

    original_b64 = to_b64(img.resize((IMG_SIZE, IMG_SIZE)))
    t0 = time.time()

    # ══════════════════════════════════════════════════════════
    #  STEP 1 — Validate: is this a chest X-ray? (ML or fallback)
    # ══════════════════════════════════════════════════════════
    is_xray, reason, xray_conf = is_valid_xray(img)

    if not is_xray:
        elapsed_ms = int((time.time() - t0) * 1000)
        record = {
            "id":             str(uuid.uuid4()),
            "timestamp":      datetime.now().isoformat(),
            "patient_name":   patient_name,
            "patient_id":     patient_id,
            "prediction":     "Invalid Image",
            "confidence":     0.0,
            "probabilities":  {n: 0.0 for n in CLASS_NAMES},
            "is_pneumonia":   False,
            "is_invalid":     True,
            "invalid_reason": reason,
            "xray_confidence": round(xray_conf, 4),
            "elapsed_ms":     elapsed_ms,
            "original_img":   original_b64,
            "heatmap_img":    original_b64,
        }
        return jsonify(record)

    # ══════════════════════════════════════════════════════════
    #  STEP 2 — Classify: Normal / Bacterial / Viral
    # ══════════════════════════════════════════════════════════
    # Preprocess: grayscale → RGB → resize → normalise
    clean_img        = preprocess_xray(img)
    tensor           = TRANSFORM(clean_img).unsqueeze(0)
    pred_idx, probs  = classify(tensor)
    pred_label  = CLASS_NAMES[pred_idx]
    confidence  = probs[pred_idx]
    is_pneumonia = pred_idx != 0   # 0 = Normal

    # ══════════════════════════════════════════════════════════
    #  STEP 3 — Grad-CAM: ONLY for Bacterial or Viral Pneumonia
    # ══════════════════════════════════════════════════════════
    if is_pneumonia:
        cam         = gradcam.generate(tensor, pred_idx)
        overlay     = build_heatmap_overlay(clean_img, cam)
        heatmap_b64 = to_b64(overlay)
    else:
        # Normal prediction — return original image, no heatmap
        heatmap_b64 = original_b64

    elapsed_ms = int((time.time() - t0) * 1000)

    # ── Build and save record ─────────────────────────────────
    record = {
        "id":              str(uuid.uuid4()),
        "timestamp":       datetime.now().isoformat(),
        "patient_name":    patient_name,
        "patient_id":      patient_id,
        "prediction":      pred_label,
        "confidence":      round(confidence, 4),
        "probabilities":   {CLASS_NAMES[i]: round(probs[i], 4) for i in range(3)},
        "is_pneumonia":    is_pneumonia,
        "is_invalid":      False,
        "invalid_reason":  None,
        "xray_confidence": round(xray_conf, 4),   # ML validator score
        "gradcam_generated": is_pneumonia,
        "elapsed_ms":      elapsed_ms,
        "original_img":    original_b64,
        "heatmap_img":     heatmap_b64,
    }

    history = load_history()
    history.insert(0, record)
    history = history[:100]
    save_history(history)

    return jsonify(record)


# ── Other endpoints ───────────────────────────────────────────
@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify(load_history())


@app.route("/api/stats", methods=["GET"])
def get_stats():
    history = load_history()
    total   = len(history)
    if total == 0:
        return jsonify({"total": 0, "pneumonia": 0, "normal": 0,
                        "bacterial": 0, "viral": 0, "avg_confidence": 0})
    pneumonia = sum(1 for r in history if r.get("is_pneumonia"))
    bacterial = sum(1 for r in history if r.get("prediction") == "Bacterial Pneumonia")
    viral     = sum(1 for r in history if r.get("prediction") == "Viral Pneumonia")
    avg_conf  = round(sum(r["confidence"] for r in history) / total, 4)
    return jsonify({
        "total":          total,
        "pneumonia":      pneumonia,
        "normal":         total - pneumonia,
        "bacterial":      bacterial,
        "viral":          viral,
        "avg_confidence": avg_conf,
    })


@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    return jsonify(META)


@app.route("/api/history/<record_id>", methods=["DELETE"])
def delete_record(record_id):
    history = [r for r in load_history() if r["id"] != record_id]
    save_history(history)
    return jsonify({"success": True})


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  PneumoScan AI — Diagnostic System v5")
    print(f"  Device    : {DEVICE}")
    print(f"  Classifier: {MODEL_PATH}")
    validator_status = f"{VALIDATOR_PATH} (ML)" if VALIDATOR_MODEL else "rule-based fallback"
    print(f"  Validator : {validator_status}")
    print("  Pipeline  :")
    print("    Step 1 → X-ray validation (ML MobileNetV2 or rule-based)")
    print("    Step 2 → Classification   (Normal / Bacterial / Viral)")
    print("    Step 3 → Grad-CAM         (Bacterial / Viral ONLY)")
    print("  Open      : http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
# ─────────────────────────────────────────────────────────────
#  LOAD ML X-RAY VALIDATOR (MobileNetV2)
# ─────────────────────────────────────────────────────────────
#
#  Architecture must match exactly what was trained in the Kaggle notebook:
#    MobileNetV2 backbone + custom head: 1280 → 256 → 2
#    Class 0 = xray, Class 1 = not_xray
#
VALIDATOR_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),  # match training preprocessing
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def build_validator_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 2)
    )
    m.load_state_dict(torch.load(VALIDATOR_PATH, map_location=DEVICE))
    m.eval()
    return m.to(DEVICE)

# Try to load ML validator — fall back to rule-based if not found
VALIDATOR_MODEL = None
if VALIDATOR_PATH.exists():
    try:
        VALIDATOR_MODEL = build_validator_model()
        print(f"ML X-ray validator loaded from {VALIDATOR_PATH}")
    except Exception as e:
        print(f"WARNING: Could not load ML validator ({e}). Using rule-based fallback.")
else:
    print(f"WARNING: {VALIDATOR_PATH} not found. Using rule-based fallback validator.")
    print("         Train it on Kaggle using xray_validator_training.ipynb")

# ─────────────────────────────────────────────────────────────
#  STEP 1A — ML VALIDATOR (primary, when model is available)
# ─────────────────────────────────────────────────────────────
def ml_is_valid_xray(pil_img: Image.Image) -> tuple:
    """
    Uses the trained MobileNetV2 binary classifier to decide
    whether the uploaded image is a chest X-ray.

    Returns:
        (True,  "valid",   confidence_score)  → proceed with classification
        (False, reason,    confidence_score)  → reject

    confidence_score = model's confidence that it IS an X-ray (0.0 – 1.0)
    """
    tensor = VALIDATOR_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = VALIDATOR_MODEL(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    p_xray     = float(probs[0])   # probability it IS an X-ray
    p_not_xray = float(probs[1])   # probability it is NOT an X-ray

    if p_not_xray >= VALIDATOR_THRESHOLD:
        return False, (
            f"This does not appear to be a chest X-ray "
            f"(model confidence: {p_not_xray*100:.1f}% not an X-ray). "
            "Please upload a valid grayscale chest X-ray image."
        ), p_xray

    return True, "valid", p_xray


# ─────────────────────────────────────────────────────────────
#  STEP 1B — RULE-BASED FALLBACK (used only if ML model missing)
# ─────────────────────────────────────────────────────────────
def rule_is_valid_xray(pil_img: Image.Image) -> tuple:
    """
    Minimal safe fallback. Only rejects images that are CLEARLY not X-rays.
    Used only when xray_validator.pth has not been trained yet.
    """
    img_rgb = pil_img.convert("RGB")
    w, h    = img_rgb.size
    img_np  = np.array(img_rgb, dtype=np.uint8)

    if w < 100 or h < 100:
        return False, f"Image too small ({w}×{h}px). Upload a full-resolution chest X-ray.", 0.0

    ratio = w / h
    if ratio < 0.3 or ratio > 3.5:
        return False, "Unusual aspect ratio. Chest X-rays are roughly square or portrait.", 0.0

    hsv      = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    mean_sat = float(hsv[:, :, 1].mean())
    if mean_sat > 60:
        return False, (
            f"Image appears to be a colour photo (saturation: {mean_sat:.0f}/255). "
            "Upload a grayscale chest X-ray."
        ), 0.0

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    mean_b = float(gray.mean())
    std_b  = float(gray.std())
    if mean_b < 5 and std_b < 5:
        return False, "Image is completely black. Upload a valid chest X-ray.", 0.0
    if mean_b > 250 and std_b < 5:
        return False, "Image is completely white/blank. Upload a valid chest X-ray.", 0.0

    return True, "valid", 0.5   # unknown confidence for rule-based


# ─────────────────────────────────────────────────────────────
#  UNIFIED VALIDATOR — routes to ML or rule-based automatically
# ─────────────────────────────────────────────────────────────
def is_valid_xray(pil_img: Image.Image) -> tuple:
    """
    Calls ML validator if available, falls back to rule-based otherwise.
    Returns (is_valid: bool, reason: str, xray_confidence: float)
    """
    if VALIDATOR_MODEL is not None:
        return ml_is_valid_xray(pil_img)
    return rule_is_valid_xray(pil_img)


# ─────────────────────────────────────────────────────────────
#  PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────
def preprocess_xray(pil_img: Image.Image) -> Image.Image:
    """
    Grayscale → RGB conversion to match DenseNet-121 training format.
    Removes colour casts and ensures all 3 channels are identical.
    """
    return pil_img.convert("L").convert("RGB")


# ─────────────────────────────────────────────────────────────
#  STEP 2 — PNEUMONIA CLASSIFIER
# ─────────────────────────────────────────────────────────────
def classify(tensor):
    """
    Runs DenseNet-121 forward pass (no gradient).
    Returns: (pred_idx, probs_list)
    """
    with torch.no_grad():
        output = model(tensor.to(DEVICE))
        probs  = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        pred   = int(probs.argmax())
    return pred, probs.tolist()


# ─────────────────────────────────────────────────────────────
#  STEP 3 — GRAD-CAM (only for Bacterial / Viral Pneumonia)
# ─────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM for DenseNet-121.

    Target layer: features.norm5 (BatchNorm after denseblock4).
    Using norm5 instead of denseblock4 directly gives cleaner,
    less scattered activations because batch-norm has already
    normalised the feature map variance.

    Only called when pred_idx is 1 (Bacterial) or 2 (Viral).
    Never called for Normal predictions or invalid images.
    """
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target = model.features.norm5    # ← norm5 for cleaner maps

        def fwd(module, inp, out):
            self.activations = out.detach()

        def bwd(module, gin, gout):
            self.gradients = gout[0].detach()

        target.register_forward_hook(fwd)
        target.register_backward_hook(bwd)

    def generate(self, tensor, pred_idx):
        self.model.eval()
        tensor = tensor.to(DEVICE).requires_grad_(True)
        output = self.model(tensor)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, pred_idx] = 1.0
        output.backward(gradient=one_hot)

        # Global average pool gradients → channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = torch.relu(cam).cpu().numpy()

        # Normalise
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


gradcam = GradCAM(model)


# ─────────────────────────────────────────────────────────────
#  IMAGE HELPERS
# ─────────────────────────────────────────────────────────────
def build_heatmap_overlay(pil_img, cam):
    """
    Builds a clean Grad-CAM heatmap overlay on the original X-ray.

    Fixes for scatter/artifact issues:
      1. Bicubic interpolation when upscaling 7x7 → 224x224 (smoother than linear)
      2. Gaussian blur after resize (removes block artifacts)
      3. Clip to 95th percentile before colormap (removes noisy outlier activations)
      4. Higher original image weight (0.6) so anatomy stays visible under heatmap
    """
    # Resize original to model input size
    img_resized = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))

    # Step 1: Upscale CAM using BICUBIC (much smoother than linear for small maps)
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_CUBIC)

    # Step 2: Gaussian blur to smooth out any remaining block artifacts
    cam_resized = cv2.GaussianBlur(cam_resized, ksize=(11, 11), sigmaX=4)

    # Step 3: Clip to 95th percentile to suppress noisy weak activations
    # This makes the heatmap focus on the strongest activation region only
    p95 = float(np.percentile(cam_resized, 95))
    cam_resized = np.clip(cam_resized, 0, p95)

    # Step 4: Re-normalise to 0-1 after clipping
    cam_resized = cam_resized / (cam_resized.max() + 1e-8)

    # Step 5: Apply JET colormap (blue→green→red, red = highest activation)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Step 6: Blend — keep more of the original (60%) so anatomy is clear
    overlay = np.uint8(0.40 * heatmap + 0.60 * img_resized)
    return Image.fromarray(overlay)


def to_b64(pil_or_array):
    """Converts PIL Image or numpy array to base64 JPEG data URI."""
    if isinstance(pil_or_array, np.ndarray):
        pil_or_array = Image.fromarray(pil_or_array)
    buf = io.BytesIO()
    pil_or_array.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────
#  HISTORY HELPERS
# ─────────────────────────────────────────────────────────────
def load_history():
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory("static/css", filename)

@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory("static/js", filename)


# ── Main prediction endpoint ──────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file         = request.files["file"]
    patient_name = request.form.get("patient_name", "Anonymous")
    patient_id   = request.form.get("patient_id") or str(uuid.uuid4())[:8].upper()

    # ── Open image ────────────────────────────────────────────
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Cannot open file. Please upload a valid image."}), 400

    original_b64 = to_b64(img.resize((IMG_SIZE, IMG_SIZE)))
    t0 = time.time()

    # ══════════════════════════════════════════════════════════
    #  STEP 1 — Validate: is this a chest X-ray? (ML or fallback)
    # ══════════════════════════════════════════════════════════
    is_xray, reason, xray_conf = is_valid_xray(img)

    if not is_xray:
        elapsed_ms = int((time.time() - t0) * 1000)
        record = {
            "id":             str(uuid.uuid4()),
            "timestamp":      datetime.now().isoformat(),
            "patient_name":   patient_name,
            "patient_id":     patient_id,
            "prediction":     "Invalid Image",
            "confidence":     0.0,
            "probabilities":  {n: 0.0 for n in CLASS_NAMES},
            "is_pneumonia":   False,
            "is_invalid":     True,
            "invalid_reason": reason,
            "xray_confidence": round(xray_conf, 4),
            "elapsed_ms":     elapsed_ms,
            "original_img":   original_b64,
            "heatmap_img":    original_b64,
        }
        return jsonify(record)

    # ══════════════════════════════════════════════════════════
    #  STEP 2 — Classify: Normal / Bacterial / Viral
    # ══════════════════════════════════════════════════════════
    # Preprocess: grayscale → RGB → resize → normalise
    clean_img        = preprocess_xray(img)
    tensor           = TRANSFORM(clean_img).unsqueeze(0)
    pred_idx, probs  = classify(tensor)
    pred_label  = CLASS_NAMES[pred_idx]
    confidence  = probs[pred_idx]
    is_pneumonia = pred_idx != 0   # 0 = Normal

    # ══════════════════════════════════════════════════════════
    #  STEP 3 — Grad-CAM: ONLY for Bacterial or Viral Pneumonia
    # ══════════════════════════════════════════════════════════
    if is_pneumonia:
        cam         = gradcam.generate(tensor, pred_idx)
        overlay     = build_heatmap_overlay(clean_img, cam)
        heatmap_b64 = to_b64(overlay)
    else:
        # Normal prediction — return original image, no heatmap
        heatmap_b64 = original_b64

    elapsed_ms = int((time.time() - t0) * 1000)

    # ── Build and save record ─────────────────────────────────
    record = {
        "id":              str(uuid.uuid4()),
        "timestamp":       datetime.now().isoformat(),
        "patient_name":    patient_name,
        "patient_id":      patient_id,
        "prediction":      pred_label,
        "confidence":      round(confidence, 4),
        "probabilities":   {CLASS_NAMES[i]: round(probs[i], 4) for i in range(3)},
        "is_pneumonia":    is_pneumonia,
        "is_invalid":      False,
        "invalid_reason":  None,
        "xray_confidence": round(xray_conf, 4),   # ML validator score
        "gradcam_generated": is_pneumonia,
        "elapsed_ms":      elapsed_ms,
        "original_img":    original_b64,
        "heatmap_img":     heatmap_b64,
    }

    history = load_history()
    history.insert(0, record)
    history = history[:100]
    save_history(history)

    return jsonify(record)


# ── Other endpoints ───────────────────────────────────────────
@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify(load_history())


@app.route("/api/stats", methods=["GET"])
def get_stats():
    history = load_history()
    total   = len(history)
    if total == 0:
        return jsonify({"total": 0, "pneumonia": 0, "normal": 0,
                        "bacterial": 0, "viral": 0, "avg_confidence": 0})
    pneumonia = sum(1 for r in history if r.get("is_pneumonia"))
    bacterial = sum(1 for r in history if r.get("prediction") == "Bacterial Pneumonia")
    viral     = sum(1 for r in history if r.get("prediction") == "Viral Pneumonia")
    avg_conf  = round(sum(r["confidence"] for r in history) / total, 4)
    return jsonify({
        "total":          total,
        "pneumonia":      pneumonia,
        "normal":         total - pneumonia,
        "bacterial":      bacterial,
        "viral":          viral,
        "avg_confidence": avg_conf,
    })


@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    return jsonify(META)


@app.route("/api/history/<record_id>", methods=["DELETE"])
def delete_record(record_id):
    history = [r for r in load_history() if r["id"] != record_id]
    save_history(history)
    return jsonify({"success": True})


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  PneumoScan AI — Diagnostic System v5")
    print(f"  Device    : {DEVICE}")
    print(f"  Classifier: {MODEL_PATH}")
    validator_status = f"{VALIDATOR_PATH} (ML)" if VALIDATOR_MODEL else "rule-based fallback"
    print(f"  Validator : {validator_status}")
    print("  Pipeline  :")
    print("    Step 1 → X-ray validation (ML MobileNetV2 or rule-based)")
    print("    Step 2 → Classification   (Normal / Bacterial / Viral)")
    print("    Step 3 → Grad-CAM         (Bacterial / Viral ONLY)")
    print("  Open      : http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)