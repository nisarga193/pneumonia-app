"""
Pneumonia Detection — Flask Backend
====================================
Run:  python app.py
Then open: http://localhost:5000
"""

import os, json, io, base64, time, uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = Path("densenet121_pneumonia.pth")
METADATA_PATH = Path("model_metadata.json")
HISTORY_FILE  = Path("session_history.json")
IMG_SIZE      = 224
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
def build_model():
    m = models.densenet121(weights=None)

    m.classifier = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 3)
    )

    # load weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # remove DataParallel "module." prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    m.load_state_dict(new_state_dict, strict=False)

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

print(f"Loading model on {DEVICE}...")
model = build_model()
print("Model ready.")

# ─────────────────────────────────────────────
#  GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target = model.features.denseblock4

        def fwd(module, inp, out):
            self.activations = out.detach()
        def bwd(module, gin, gout):
            self.gradients = gout[0].detach()

        target.register_forward_hook(fwd)
        target.register_backward_hook(bwd)

    def generate(self, tensor):
        self.model.eval()
        tensor = tensor.to(DEVICE).requires_grad_(True)
        output = self.model(tensor)
        probs  = torch.softmax(output, dim=1).squeeze().detach().cpu().numpy()
        pred   = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, pred] = 1.0
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, pred, probs.tolist()

gradcam = GradCAM(model)

# ─────────────────────────────────────────────
#  HISTORY HELPERS
# ─────────────────────────────────────────────
def load_history():
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory("static/css", filename)

@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory("static/js", filename)

@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file   = request.files["file"]
    patient_name = request.form.get("patient_name", "Anonymous")
    patient_id   = request.form.get("patient_id", str(uuid.uuid4())[:8].upper())

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    # Run model
    tensor = TRANSFORM(img).unsqueeze(0)
    t0 = time.time()
    cam, pred_idx, probs = gradcam.generate(tensor)
    elapsed_ms = int((time.time() - t0) * 1000)

    pred_label = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx]
    is_pneumonia = pred_idx != 0  # 0 = Normal

    # Build heatmap overlay
    img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.45 * heatmap + 0.55 * img_np)

    def to_b64(arr_or_pil):
        if isinstance(arr_or_pil, np.ndarray):
            arr_or_pil = Image.fromarray(arr_or_pil)
        buf = io.BytesIO()
        arr_or_pil.save(buf, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    original_b64 = to_b64(img.resize((IMG_SIZE, IMG_SIZE)))
    overlay_b64  = to_b64(overlay)

    # Save to history
    record = {
        "id":           str(uuid.uuid4()),
        "timestamp":    datetime.now().isoformat(),
        "patient_name": patient_name,
        "patient_id":   patient_id,
        "prediction":   pred_label,
        "confidence":   round(confidence, 4),
        "probabilities": {CLASS_NAMES[i]: round(probs[i], 4) for i in range(3)},
        "is_pneumonia": is_pneumonia,
        "elapsed_ms":   elapsed_ms,
        "original_img": original_b64,
        "heatmap_img":  overlay_b64,
    }
    history = load_history()
    history.insert(0, record)
    history = history[:100]  # keep last 100
    save_history(history)

    return jsonify(record)


@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify(load_history())


@app.route("/api/stats", methods=["GET"])
def get_stats():
    history = load_history()
    total = len(history)
    if total == 0:
        return jsonify({"total": 0, "pneumonia": 0, "normal": 0,
                        "bacterial": 0, "viral": 0, "avg_confidence": 0})
    pneumonia = sum(1 for r in history if r["is_pneumonia"])
    bacterial = sum(1 for r in history if r["prediction"] == "Bacterial Pneumonia")
    viral     = sum(1 for r in history if r["prediction"] == "Viral Pneumonia")
    avg_conf  = round(sum(r["confidence"] for r in history) / total, 4)
    return jsonify({
        "total": total,
        "pneumonia": pneumonia,
        "normal": total - pneumonia,
        "bacterial": bacterial,
        "viral": viral,
        "avg_confidence": avg_conf,
    })


@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    return jsonify(META)


@app.route("/api/history/<record_id>", methods=["DELETE"])
def delete_record(record_id):
    history = load_history()
    history = [r for r in history if r["id"] != record_id]
    save_history(history)
    return jsonify({"success": True})


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Pneumonia AI Diagnosis System")
    print(f"  Device: {DEVICE}")
    print(f"  Model:  {MODEL_PATH}")
    print("  Open:   http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
