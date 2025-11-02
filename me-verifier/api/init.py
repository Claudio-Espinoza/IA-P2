import sys
import logging
from pathlib import Path
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import setup_logger
from api.config import MODEL_PATH, SCALER_PATH

logger = setup_logger("me_verifier")


class ModelLoader:
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_loaded = False
        self.scaler_loaded = False
    
    
    def load_model(self):
        try:
            logger.info(f"Cargando modelo desde: {MODEL_PATH}")
            
            if not MODEL_PATH.exists():
                logger.error(f"❌ Archivo de modelo no encontrado: {MODEL_PATH}")
                logger.info("   Ejecuta: python train.py")
                return False
            
            self.model = joblib.load(MODEL_PATH)
            self.model_loaded = True
            logger.info("✅ Modelo cargado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error al cargar el modelo: {e}")
            return False
    
    
    def load_scaler(self):
        try:
            logger.info(f"Cargando escalador desde: {SCALER_PATH}")
            
            if not SCALER_PATH.exists():
                logger.error(f"❌ Archivo de escalador no encontrado: {SCALER_PATH}")
                logger.info("   Ejecuta: python train.py")
                return False
            
            self.scaler = joblib.load(SCALER_PATH)
            self.scaler_loaded = True
            logger.info("✅ Escalador cargado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error al cargar el escalador: {e}")
            return False
    
    
    def load_all(self):
        logger.info("=" * 50)
        logger.info("=== Iniciando carga de recursos ===")
        logger.info("=" * 50)
        
        model_ok = self.load_model()
        scaler_ok = self.load_scaler()
        
        if model_ok and scaler_ok:
            logger.info("=" * 50)
            logger.info("Todos los recursos cargados exitosamente")
            logger.info("=" * 50)
            return True
        else:
            logger.warning("=" * 50)
            logger.warning("Algunos recursos no se cargaron correctamente")
            if not model_ok:
                logger.warning("   ❌ Modelo no disponible")
            if not scaler_ok:
                logger.warning("   ❌ Escalador no disponible")
            logger.warning("=" * 50)
            return False
    
    
    def is_ready(self):
        return self.model_loaded and self.scaler_loaded
   
   
    def get_status(self):
        return {
            'model_loaded': self.model_loaded,
            'scaler_loaded': self.scaler_loaded,
            'ready': self.is_ready()
        }


model_loader = ModelLoader()