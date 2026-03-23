# PneumoScan AI — Local Setup Guide

## 📁 Project Folder Structure

```
pneumonia_app/
├── app.py                        ← Flask backend (inference + API)
├── requirements.txt              ← Python dependencies
├── densenet121_pneumonia.pth     ← ⬅ Copy from Kaggle output
├── model_metadata.json           ← ⬅ Copy from Kaggle output
├── session_history.json          ← Auto-created on first scan
└── static/
    └── index.html                ← Doctor UI (frontend)
```

---

## 🚀 Step-by-Step Setup

### Step 1 — Copy your Kaggle model files
Place these two files in the same folder as `app.py`:
- `densenet121_pneumonia.pth`
- `model_metadata.json`

### Step 2 — Create a Python virtual environment
```bash
# In the pneumonia_app/ folder:
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> If you have a GPU locally:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### Step 4 — Run the server
```bash
python app.py
```

You should see:
```
==================================================
  Pneumonia AI Diagnosis System
  Device: cpu   (or cuda if GPU found)
  Model:  densenet121_pneumonia.pth
  Open:   http://localhost:5000
==================================================
```

### Step 5 — Open the UI
Open your browser and go to:
```
http://localhost:5000
```

---

## 🖥️ UI Pages

| Page       | What it shows |
|------------|---------------|
| Dashboard  | Stats (total scans, normal/bacterial/viral counts), charts, recent results |
| New Scan   | Upload X-ray, enter patient name/ID, run analysis, see heatmap |
| History    | All past scans with filter, search, and delete |
| About      | Model architecture, training details, performance metrics, disclaimer |

---

## ⚠️ Important Notes

- The model runs on **CPU** by default — inference takes ~1–5 seconds per image.
- If you have a CUDA GPU, it will automatically use it (much faster).
- All scan history is saved locally in `session_history.json`.
- This tool is for **research/assistive use only** — not for clinical diagnosis.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: densenet121_pneumonia.pth` | Copy the .pth file to the same folder as app.py |
| `ModuleNotFoundError: torch` | Run `pip install -r requirements.txt` |
| Port 5000 already in use | Change `port=5000` to `port=5001` in app.py |
| Model loads but crashes | Check that `model_metadata.json` matches the .pth file |
