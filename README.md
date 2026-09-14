# 🛰️ EPSIS — Explainable Predictive Satellite Intelligence System

**EPSIS** (Explainable Predictive Satellite Intelligence System) is a production-grade Remote Sensing and Geospatial AI application for Sentinel-2 satellite image change detection, spectral land-cover classification, quantitative 0–100 severity analysis, contextual risk assessment, and Explainable AI (HiResCAM & Confidence Mapping).

---

## 🌟 Key Features

* **Satellite Imagery Engine**: Automated Sentinel-2 L2A imagery retrieval via **Google Earth Engine (GEE)** with STAC COG API fallback.
* **Siamese Deep Learning Backbone**: Pretrained **TinyCD** model for high-accuracy temporal change detection.
* **Dynamic Otsu Thresholding**: Statistical bimodal thresholding (`cv2.THRESH_OTSU`) for adaptive decision boundaries.
* **Explainable AI (XAI)**:
  * **HiResCAM**: Gradient-weighted latent activation heatmap explainability.
  * **JET Probability Activation Map**: Full-spectrum dynamic contrast stretched probability distribution.
  * **Model Confidence Map**: Distance-from-boundary certainty visualization ($C = \frac{|P - T|}{\max(T, 1 - T)}$).
* **Predictive Risk & Severity Analysis**: Quantitative 0–100 impact scoring, spectral proxy classification (NDVI, NDWI, NDBI), and recommended mitigation actions.
* **Interactive Streamlit Dashboard**: Web UI on `http://localhost:8501` featuring location presets, date range selection, multi-tab XAI maps, and diagnostic telemetry metrics.

---

## 📋 System Requirements & Prerequisites

* **Operating System**: Windows 10/11, macOS, or Linux
* **Python Version**: Python `3.10`, `3.11`, `3.12`, or `3.14`
* **Git**: Installed and configured
* **Google Cloud Project**: GCP Project ID with Earth Engine API enabled

---

## 🚀 Quick Start Guide

### Step 1: Clone the Repository

Open your Command Prompt (`cmd`) or Terminal and clone the repository:

```bash
git clone https://github.com/saiprasad981/EPSIS.git
cd EPSIS
```

---

### Step 2: Create & Activate a Virtual Environment

It is recommended to use a virtual environment to manage dependencies:

#### Windows (CMD):
```cmd
python -m venv venv
venv\Scripts\activate
```

#### Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Required Packages

Install all core deep learning, geospatial, and web dependencies using `pip`:

```bash
pip install -r requirements.txt
```

---

### Step 4: Google Earth Engine (GEE) Setup & CMD Authentication

EPSIS uses Google Earth Engine to fetch Sentinel-2 satellite imagery. Follow these exact steps to set up GEE authentication using Command Prompt (CMD):

#### 1. Create a Google Cloud Project ID
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project or select an existing project (e.g., `gen-lang-client-0301255187`).
3. Enable the **Google Earth Engine API** in your GCP Console:
   * Go to **APIs & Services** $\to$ **Library**.
   * Search for **Earth Engine API** and click **Enable**.

#### 2. Authenticate Earth Engine via CMD
In your Command Prompt (`cmd`) with the virtual environment activated, run:

```bash
earthengine authenticate
```

* A web browser window will automatically open asking you to sign in with your Google account.
* Grant permissions and authorize the Earth Engine CLI.
* Copy the authorization code from the browser and paste it into Command Prompt, then press `Enter`.

#### 3. Configure Environment Variables (`.env`)
Create a `.env` file in the root project directory (or copy from `.env.example`):

```bash
cp .env.example .env
```

Open `.env` in any text editor and set your Google Cloud Project ID:

```env
GCP_PROJECT_ID=gen-lang-client-0301255187
DEFAULT_AOI_BUFFER_M=1000
CLOUD_FILTER_MAX_PERCENT=25
```

---

### Step 5: Launch the EPSIS Web Application

Run the Streamlit application:

```bash
streamlit run app.py
```

* Streamlit will start the local server and print the access URL:
  👉 **`http://localhost:8501`**
* Open your browser and navigate to `http://localhost:8501`.

---

## 🛠️ Step-by-Step Dashboard Usage

1. **Select Location Preset or Search**:
   * Choose a preset from the sidebar (e.g., **Bengaluru Tech Park (India)**, **Kokapet Growth Corridor**, **Hyderabad Hitech City**, or **LEVIR Building Development**).
   * Or use the interactive Folium map / search box to pick any global coordinates.
2. **Select Temporal Date Ranges**:
   * Set **T1 Start/End** (Reference Period, e.g., `2021-01-01` $\to$ `2021-12-31`).
   * Set **T2 Start/End** (Comparison Period, e.g., `2023-01-01` $\to$ `2023-12-31`).
3. **Execute Intelligence Pipeline**:
   * Click **🚀 Run Complete EPSIS Intelligence Analysis**.
4. **Inspect Results & XAI Maps**:
   * **Change Highlight Overlay**: Red transparent highlight on T2 comparison image showing changed pixels.
   * **Binary Change Mask**: White = changed, Black = unchanged.
   * **JET Probability Map**: Continuous probability activation distribution (Blue = 0 $\to$ Red = High).
   * **HiResCAM Explainability**: Model latent feature attribution heatmap.
   * **Model Confidence Map**: Prediction certainty relative to decision threshold.
5. **View Map Telemetry Diagnostics**:
   * Expand **📊 Map Diagnostic Metrics & Statistics** to view live tensor shapes, unique pixel counts, min, max, mean, std, and percentiles.

---

## 🧪 Verification & Pipeline Testing

To run automated verification tests on GEE authentication, TinyCD checkpoints, and pipeline execution without starting the web UI, run:

```bash
python test_epsis_pipeline.py
```

---

## 📁 Repository Directory Structure

```text
EPSIS/
├── app.py                      # Main Streamlit web application
├── satellite.py                # Unified GEE & STAC satellite acquisition engine
├── gee_utils.py                # Google Earth Engine clipping & thumbnail helpers
├── inference.py                # TinyCD model inference, Otsu thresholding & HiResCAM XAI
├── epsis_analysis.py           # Quantitative severity, classification & risk assessment
├── geocode.py                  # Location geocoding utility
├── config.py                   # Centralized configuration & environment loader
├── requirements.txt            # System dependency requirements
├── .env.example                # Template environment variables
├── .gitignore                  # Git ignore rules
└── Tiny_model_4_CD/            # TinyCD pretrained model checkpoints & architecture
    ├── pretrained_models/
    │   ├── levir_best.pth      # LEVIR-CD pretrained weights
    │   └── whu_best.pth        # WHU-CD pretrained weights
    └── models/                 # TinyCD neural network modules
```

---

## 📄 License

This project is open-source and available under the MIT License.
