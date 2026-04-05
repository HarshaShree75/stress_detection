# 🧠 Stress Detection System
### Using Machine Learning & Image Processing

> Upload a face photo → get instant "Stressed / Not Stressed" prediction with confidence score.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [How It Works](#-how-it-works)
3. [Folder Structure](#-folder-structure)
4. [File Explanations](#-file-explanations)
5. [Installation (Step-by-Step)](#-installation-step-by-step)
6. [Dataset Setup](#-dataset-setup)
7. [Training the Model](#-training-the-model)
8. [Running the App](#-running-the-app)
9. [Using the App](#-using-the-app)
10. [Troubleshooting](#-troubleshooting)
11. [Technologies Used](#-technologies-used)

---

## 🔍 Project Overview

This project builds an **AI-powered Stress Detection System** that:

- **Accepts** a facial photograph as input
- **Detects** the face in the photo using OpenCV
- **Analyses** the facial expression using a trained CNN model
- **Predicts** whether the person appears **Stressed** or **Not Stressed**
- **Shows** a confidence percentage with a friendly web interface

### What is FER-2013?

FER-2013 (Facial Expression Recognition 2013) is a public dataset containing **35,887 grayscale face images** (48×48 pixels) labelled with 7 emotions:

| FER Label | Emotion  | Stress Mapping  |
|-----------|----------|-----------------|
| 0         | Angry    | ✅ Stressed      |
| 1         | Disgust  | ✅ Stressed      |
| 2         | Fear     | ✅ Stressed      |
| 3         | Happy    | ❌ Not Stressed  |
| 4         | Sad      | ❌ Not Stressed  |
| 5         | Surprise | ❌ Not Stressed  |
| 6         | Neutral  | ❌ Not Stressed  |

We re-label these 7 emotions into our 2 classes (Stressed / Not Stressed).

---

## ⚙️ How It Works

```
User uploads photo
        ↓
OpenCV detects face  →  no face found → show error
        ↓
Crop & preprocess face
  • Grayscale
  • Resize to 48×48
  • Normalise pixels to [0, 1]
        ↓
CNN model predicts probabilities
  [Not Stressed: 23%,  Stressed: 77%]
        ↓
Display result + confidence + annotated image
```

### CNN Architecture

```
Input (48×48×1)
    │
Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)
    │
Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
    │
Conv2D(128) + BatchNorm + MaxPool + Dropout(0.40)
    │
Flatten → Dense(256) → Dropout(0.5)
    │
Dense(2) + Softmax
    │
Output: [P(Not Stressed), P(Stressed)]
```

---

## 📁 Folder Structure

```
stress_detection/
│
├── app.py                  ← Streamlit web interface (run this!)
├── model.py                ← CNN model definition + training script
├── predict.py              ← Face detection + preprocessing + prediction logic
├── download_dataset.py     ← Helper to download FER-2013 dataset
├── requirements.txt        ← Python package dependencies
├── README.md               ← This file
│
├── data/                   ← (create this) Dataset folder
│   └── fer2013.csv         ← Downloaded dataset file
│
├── stress_model.h5         ← Saved model (created after training)
└── training_history.png    ← Loss/accuracy curves (created after training)
```

---

## 📄 File Explanations

### `app.py`
The **web interface** built with Streamlit. It:
- Shows a file upload widget
- Calls `predict.py` to analyse the image
- Displays the original photo, annotated photo (with bounding box), result label, and confidence bars

### `model.py`
The **model building and training** script. It:
- Loads and preprocesses the FER-2013 CSV dataset
- Re-labels 7 emotions → 2 classes (stressed / not stressed)
- Builds a CNN using TensorFlow/Keras
- Trains with data augmentation and early stopping
- Saves the best model to `stress_model.h5`

### `predict.py`
The **prediction engine**. It:
- Loads `stress_model.h5` from disk (once, cached)
- Detects faces using OpenCV's Haar Cascade classifier
- Preprocesses the detected face region
- Runs the model and returns label + confidence
- Draws a coloured bounding box on the image

### `download_dataset.py`
A **helper script** that guides you through downloading the FER-2013 dataset either via the Kaggle API or manually from a browser.

### `requirements.txt`
Lists all Python packages the project needs. Install them all at once with:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Installation (Step-by-Step)

### Prerequisites
- Python 3.9, 3.10, or 3.11 (recommended: 3.10)
- pip (comes with Python)
- ~500 MB of free disk space

### Step 1 — Check your Python version

Open a **Terminal** (Mac/Linux) or **Command Prompt** (Windows) and type:

```bash
python --version
```

You should see something like `Python 3.10.x`. If Python is not installed, download it from [python.org](https://www.python.org/downloads/).

---

### Step 2 — Download this project

If you have Git:
```bash
git clone <repo-url>
cd stress_detection
```

Or just create a folder called `stress_detection` and place all the files inside it.

---

### Step 3 — Create a virtual environment (highly recommended)

A virtual environment keeps this project's packages separate from other Python projects.

```bash
# Create the environment
python -m venv venv

# Activate it:
# On Mac / Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

Your terminal prompt will now show `(venv)` — this means it's active. ✅

---

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs TensorFlow, OpenCV, Streamlit, and other packages.  
**Expected time: 3–10 minutes** depending on your internet speed.

---

## 📊 Dataset Setup

### Option A — Automated (Kaggle API)

1. Create a free account at [kaggle.com](https://www.kaggle.com)
2. Go to **Settings → API → Create New Token** → `kaggle.json` downloads
3. Place `kaggle.json` at:
   - **Mac/Linux:** `~/.kaggle/kaggle.json`
   - **Windows:** `C:\Users\YourName\.kaggle\kaggle.json`
4. Install kaggle: `pip install kaggle`
5. Run the helper script:

```bash
python download_dataset.py
```

### Option B — Manual Download (Browser)

1. Go to: [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
2. Log in to Kaggle (free)
3. Click **⬇ Download** (top-right)
4. Unzip the downloaded `archive.zip`
5. Find `fer2013.csv` inside (≈ 61 MB)
6. Create a `data/` folder in your project and move the CSV there:

```
stress_detection/
└── data/
    └── fer2013.csv   ← paste it here
```

---

## 🏋️ Training the Model

Once `data/fer2013.csv` is in place, train the CNN:

```bash
python model.py data/fer2013.csv
```

**What happens:**
- Loads ~35,000 face images from the CSV
- Splits into 80% training / 20% validation
- Trains for up to 30 epochs (stops early if no improvement)
- Saves the best model as `stress_model.h5`
- Saves training curves as `training_history.png`

**Expected training time:**
| Hardware | Time |
|----------|------|
| CPU only | 30–60 minutes |
| GPU (CUDA) | 5–10 minutes |

> 💡 **Tip:** While waiting, you can look at `training_history.png` to see the accuracy improving over epochs.

---

## 🚀 Running the App

Make sure you are in the `stress_detection/` folder with your virtual environment active, then run:

```bash
streamlit run app.py
```

You'll see output like:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser. 🎉

---

## 🖥️ Using the App

1. **Upload a photo** — Click "Browse files" and choose a JPG/PNG image of a face
2. **Wait** — The app detects the face and analyses it (usually under 1 second)
3. **View results:**
   - 🖼️ Your uploaded image
   - 🔲 Annotated image with a coloured bounding box around the detected face
   - 📊 Prediction: **Stressed** (red) or **Not Stressed** (green)
   - 📈 Confidence breakdown bars

### Tips for best accuracy:
- ✅ Use a clear, well-lit, front-facing photo
- ✅ Only one face in the image (the largest one is analysed)
- ✅ Neutral background works best
- ❌ Avoid sunglasses, heavy shadows, or extreme angles

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `stress_model.h5 not found` | Train the model first: `python model.py data/fer2013.csv` |
| `No face detected` | Try a different photo — more frontal, better lit |
| TensorFlow installation fails | Try `pip install tensorflow-cpu` instead |
| Streamlit not found | Make sure your venv is activated |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.10** | Core programming language |
| **TensorFlow / Keras** | CNN model building & training |
| **OpenCV** | Face detection (Haar Cascade) & image preprocessing |
| **Streamlit** | Web user interface |
| **NumPy** | Array operations |
| **Pandas** | Loading CSV dataset |
| **Pillow** | Image handling in Streamlit |
| **Matplotlib** | Training curve plots |
| **scikit-learn** | Train/validation split |

---

## 📝 Notes for Beginners

**What is a CNN?**  
A Convolutional Neural Network is a type of AI that learns to recognise patterns in images (like edges, shapes, and textures) by processing them through multiple filter layers.

**What is OpenCV?**  
Open Source Computer Vision Library — a powerful tool for processing images and videos in Python.

**What is Streamlit?**  
A Python library that lets you build interactive web apps with just a few lines of Python code — no HTML or JavaScript needed!

**What does confidence mean?**  
The model outputs a probability for each class. A confidence of 87% for "Stressed" means the model is 87% sure the person looks stressed based on their expression.

---

*Built for educational purposes. Stress detection from a single image is inherently limited — results should not be used as medical diagnosis.*
