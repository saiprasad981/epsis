# 🛰️ EPSIS — Explainable Predictive Satellite Intelligence System

**EPSIS** (Explainable Predictive Satellite Intelligence System) is a production-grade Remote Sensing and Geospatial AI application for Sentinel-2 satellite image change detection, spectral land-cover classification, quantitative 0–100 severity analysis, contextual risk assessment, Reference Change Analysis, and Model Explainability (HiResCAM Attention Color Scale).

---

## 🌟 Key Features

* **Satellite Imagery Engine**: Automated Sentinel-2 L2A imagery retrieval via **Google Earth Engine (GEE)** preserving 10m native spatial resolution.
* **Siamese Deep Learning Backbone**: Pretrained **TinyCD** model for high-accuracy temporal change detection.
* **Dynamic Otsu Thresholding**: Statistical bimodal thresholding (`cv2.THRESH_OTSU`) for adaptive decision boundaries.
* **Reference Change Analysis**: Independent temporal multi-spectral & structural difference comparison computed directly from T1/T2 GEE surface reflectance ($0 \to 255$ continuous grayscale spectrum).
* **Model Explainability (HiResCAM)**: Paper-faithful gradient-weighted latent feature attribution heatmap ($\text{Blue} \to \text{Cyan} \to \text{Green} \to \text{Yellow} \to \text{Red}$).
* **Predictive Risk & Severity Analysis**: Quantitative 0–100 impact scoring, spectral proxy classification (NDVI, NDWI, NDBI), and recommended mitigation actions.
* **Interactive Streamlit Dashboard**: Web UI on `http://localhost:8501` featuring interactive map selection, date picking, side-by-side reference & explainability maps, and developer telemetry diagnostics.

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

1. **Select Target Location**:
   * Use the interactive Folium map or location search box to set target coordinates.
2. **Select Satellite Acquisition Dates**:
   * Set **Before Image Date** (e.g., `2021-06-15`).
   * Set **After Image Date** (e.g., `2023-06-20`).
3. **Execute Intelligence Pipeline**:
   * Click **🚀 Run Live GEE Change Analysis**.
4. **Inspect Results**:
   * **Temporal Satellite Imagery**: Side-by-side Before (T1) and After (T2) GEE RGB Surface Reflectance.
   * **Reference Change Analysis**: Continuous grayscale difference map ($0 = \text{Black / Low Difference}$, $255 = \text{White / Strong Difference}$).
   * **HiResCAM Attention Color Scale**: Feature attribution map ($\text{Blue} \to \text{Cyan} \to \text{Green} \to \text{Yellow} \to \text{Red}$).
   * **Predictive Intelligence Report**: Land-cover class breakdown, 0–100 severity score, risk rating, and action items.
5. **View Telemetry Diagnostics**:
   * Expand **🔧 Developer Diagnostics** to view live tensor shapes, autograd activations/gradients, raw heatmap ranges, and ROI bounding box metadata.

---

## 🧪 Verification & Pipeline Testing

To run automated verification tests on GEE authentication, TinyCD checkpoints, Reference Change Analysis, and HiResCAM autograd execution without starting the web UI, run:

```bash
python test_epsis_pipeline.py
```

To run site matrix validation across multiple worldwide locations:

```bash
python test_real_changes.py
python test_matrix_validation.py
```

---

## 📁 Repository Directory Structure

```text
EPSIS/
├── app.py                      # Main Streamlit web application & UI renderer
├── satellite.py                # Native 10m Sentinel-2 GEE acquisition & quality engine
├── gee_utils.py                # Earth Engine clipping & thumbnail helpers
├── inference.py                # TinyCD model inference, Reference Change & HiResCAM autograd XAI
├── epsis_analysis.py           # Quantitative severity, classification & risk assessment
├── geocode.py                  # Worldwide location geocoding utility
├── config.py                   # Centralized configuration & environment loader
├── test_epsis_pipeline.py      # End-to-end system integration & XAI test suite
├── test_real_changes.py        # Real temporal site change verification suite
├── test_matrix_validation.py   # Multi-site location & date matrix quality test suite
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
