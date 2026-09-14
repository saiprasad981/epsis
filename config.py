"""
config.py — EPSIS Project Configuration & Environment Manager
================================================================
Centralized configuration manager loading environment variables safely
from .env file.
"""

import os
from pathlib import Path

# Load environment variables from .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# --- Google Cloud & Earth Engine Configuration ---
# Uses the user's GCP Project ID from .env, defaulting to 'gen-lang-client-0301255187'
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", os.getenv("GEE_PROJECT", "gen-lang-client-0301255187"))

# --- OpenStreetMap / Geocoding Configuration ---
NOMINATIM_CONTACT_EMAIL = os.getenv("NOMINATIM_CONTACT_EMAIL", "epsis-user@example.com")
USER_AGENT_HEADER = f"EPSIS-Satellite-App/2.0 (contact: {NOMINATIM_CONTACT_EMAIL})"

# --- Satellite Acquisition Configuration ---
DEFAULT_AOI_BUFFER_M = int(os.getenv("DEFAULT_AOI_BUFFER_M", "1000"))
CLOUD_FILTER_MAX_PERCENT = int(os.getenv("CLOUD_FILTER_MAX_PERCENT", "20"))

# --- Model & Checkpoint Paths ---
BASE_DIR = Path(__file__).parent
LEVIR_CHECKPOINT_PATH = os.getenv("LEVIR_CHECKPOINT_PATH", str(BASE_DIR / "Tiny_model_4_CD" / "pretrained_models" / "levir_best.pth"))
WHU_CHECKPOINT_PATH = os.getenv("WHU_CHECKPOINT_PATH", str(BASE_DIR / "Tiny_model_4_CD" / "pretrained_models" / "whu_best.pth"))

# --- Streamlit / Web Server Config ---
SERVER_PORT = int(os.getenv("PORT", "8501"))
