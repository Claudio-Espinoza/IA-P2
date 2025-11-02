import os
from pathlib import Path

# Rutas base
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
LOGS_DIR = BASE_DIR / 'logs'

# Archivos de modelo
MODEL_PATH = MODELS_DIR / 'model.joblib'
SCALER_PATH = MODELS_DIR / 'scaler.joblib'

# Crear directorios si no existen
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Configuración de logs
LOG_FILE = LOGS_DIR / 'app.log'

# Parámetros de verificación
THRESHOLD = 0.75
MAX_SIZE_MB = 5
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
FACENET_MODEL = "Facenet"

# Configuración de Flask
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000

# Configuración de la API
API_VERSION = "1.0.0"
API_NAME = "Me Verifier API"