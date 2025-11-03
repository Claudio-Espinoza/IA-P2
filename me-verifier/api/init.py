"""
Módulo de inicialización y configuración de la aplicación
"""
import sys
import logging
import subprocess
import time
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
                logger.info("   Ejecuta: python setup.py")
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
                logger.info("   Ejecuta: python setup.py")
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
            logger.info("✅ Todos los recursos cargados exitosamente")
            logger.info("=" * 50)
            return True
        else:
            logger.warning("=" * 50)
            logger.warning("⚠️ Algunos recursos no se cargaron correctamente")
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


class SetupManager:    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / 'data'
        self.errors = []
        self.warnings = []
    
    def _count_images_in_directory(self, directory):
        """Cuenta imágenes válidas en un directorio"""
        if not directory.exists():
            return 0
        
        valid_extensions = ['.jpg', '.jpeg', '.png']
        images = []
        for ext in valid_extensions:
            images.extend(directory.glob(f'*{ext}'))
            images.extend(directory.glob(f'*{ext.upper()}'))
        
        return len(images)
    
    def _is_directory_empty(self, directory):
        """Verifica si el directorio está vacío"""
        return self._count_images_in_directory(directory) == 0
    
    def download_images_if_needed(self):
        """Descarga imágenes 'not_me' si el directorio está vacío"""
        logger.info("=" * 60)
        logger.info("PASO 0: Validando imágenes necesarias")
        logger.info("=" * 60)
        
        # Verificar directorio 'me'
        me_dir = self.data_dir / 'me'
        me_count = self._count_images_in_directory(me_dir)
        
        if me_count == 0:
            self.warnings.append("Directorio 'me' está vacío")
            logger.warning(f"⚠️ Directorio 'me' está vacío (0 imágenes)")
        else:
            logger.info(f"✅ Directorio 'me': {me_count} imágenes encontradas")
        
        # Verificar directorio 'not_me'
        not_me_dir = self.data_dir / 'not_me'
        not_me_count = self._count_images_in_directory(not_me_dir)
        
        if not_me_count == 0:
            logger.warning(f"⚠️ Directorio 'not_me' está vacío (0 imágenes)")
            logger.info("Descargando imágenes de 'not_me' automáticamente...")
            
            if not self._download_not_me_images():
                self.errors.append("Falló la descarga de imágenes 'not_me'")
                logger.error("❌ Falló la descarga de imágenes 'not_me'")
                return False
            
            # Verificar nuevamente
            not_me_count = self._count_images_in_directory(not_me_dir)
            logger.info(f"✅ Directorio 'not_me': {not_me_count} imágenes descargadas")
        else:
            logger.info(f"✅ Directorio 'not_me': {not_me_count} imágenes encontradas")
        
        return True
    
    def _download_not_me_images(self):
        """Ejecuta el script de descarga de imágenes"""
        try:
            search_img_path = self.base_dir / 'search_img.py'
            
            if not search_img_path.exists():
                logger.error(f"❌ Script no encontrado: {search_img_path}")
                return False
            
            logger.info(f"Ejecutando descarga: {search_img_path}")
            
            result = subprocess.run(
                [sys.executable, str(search_img_path)],
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                logger.info("✅ Descarga completada exitosamente")
                logger.debug(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"❌ Error en la descarga: {result.stderr}")
                self.errors.append(f"Error descargando imágenes: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Timeout descargando imágenes (> 600s)")
            self.errors.append("Timeout descargando imágenes")
            return False
        except Exception as e:
            logger.error(f"❌ Error descargando imágenes: {e}")
            self.errors.append(f"Error descargando imágenes: {e}")
            return False
    
    def validate_data_directories(self):
        logger.info("=" * 60)
        logger.info("PASO 1: Validando directorios de datos")
        logger.info("=" * 60)
        
        required_dirs = {
            'me': self.data_dir / 'me',
            'not_me': self.data_dir / 'not_me'
        }
        
        for dir_name, dir_path in required_dirs.items():
            if dir_path.exists():
                files = list(dir_path.glob('*'))
                logger.info(f"✅ Directorio '{dir_name}': {len(files)} archivos")
                
                if len(files) == 0:
                    self.warnings.append(f"Directorio '{dir_name}' está vacío")
                    logger.warning(f"   ⚠️ Directorio '{dir_name}' está vacío")
            else:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"📁 Creado directorio: {dir_path}")
                self.warnings.append(f"Directorio '{dir_name}' creado vacío")
        
        return len(self.warnings) == 0
    
    def _run_script(self, script_name, step_number, description):
        logger.info("=" * 60)
        logger.info(f"PASO {step_number}: {description}")
        logger.info("=" * 60)
        
        try:
            script_path = self.base_dir / script_name
            
            if not script_path.exists():
                self.errors.append(f"Script no encontrado: {script_path}")
                logger.error(f"❌ Script no encontrado: {script_path}")
                return False
            
            logger.info(f"Ejecutando: {script_path}")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {description} completado")
                return True
            else:
                self.errors.append(f"Error en {script_name}: {result.stderr}")
                logger.error(f"❌ Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.errors.append(f"Timeout en {script_name}")
            logger.error(f"❌ Timeout ejecutando {script_name}")
            return False
        except Exception as e:
            self.errors.append(f"Error ejecutando {script_name}: {e}")
            logger.error(f"❌ Error: {e}")
            return False
    
    def run_crop_faces(self):
        return self._run_script(
            'scripts/crop_faces.py',
            2,
            'Recortando rostros'
        )
    
    def run_embeddings(self):
        return self._run_script(
            'scripts/embeddings.py',
            3,
            'Extrayendo embeddings faciales'
        )
    
    def run_train(self):
        return self._run_script(
            'train.py',
            4,
            'Entrenando modelo'
        )
    
    def run_evaluate(self):
        return self._run_script(
            'evaluate.py',
            5,
            'Evaluando modelo'
        )
    
    def print_summary(self):
        logger.info("=" * 60)
        logger.info("RESUMEN")
        logger.info("=" * 60)
        
        if self.errors:
            logger.error(f"❌ Se encontraron {len(self.errors)} error(es):")
            for idx, error in enumerate(self.errors, 1):
                logger.error(f"   {idx}. {error}")
        else:
            logger.info("✅ No hay errores")
        
        if self.warnings:
            logger.warning(f"⚠️ Se encontraron {len(self.warnings)} advertencia(s):")
            for idx, warning in enumerate(self.warnings, 1):
                logger.warning(f"   {idx}. {warning}")
        else:
            logger.info("✅ No hay advertencias")
        
        logger.info("=" * 60)
    
    def run_setup(self, skip_evaluation=False):
        logger.info("=" * 60)
        logger.info("🚀 INICIANDO CONFIGURACIÓN DE ME VERIFIER")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # PASO 0: Descargar imágenes si es necesario
        if not self.download_images_if_needed():
            logger.error("❌ Falló la validación/descarga de imágenes")
            self.print_summary()
            return False
        
        # PASO 1: Validar directorios
        if not self.validate_data_directories():
            logger.warning("⚠️ Hay advertencias en los directorios de datos")
        
        # PASO 2: Recortar rostros
        if not self.run_crop_faces():
            logger.error("❌ Falló el recorte de rostros")
            self.print_summary()
            return False
        
        # PASO 3: Extraer embeddings
        if not self.run_embeddings():
            logger.error("❌ Falló la extracción de embeddings")
            self.print_summary()
            return False
        
        # PASO 4: Entrenar modelo
        if not self.run_train():
            logger.error("❌ Falló el entrenamiento")
            self.print_summary()
            return False
        
        # PASO 5: Evaluar modelo (opcional)
        if not skip_evaluation:
            if not self.run_evaluate():
                logger.warning("⚠️ Falló la evaluación, pero el modelo está entrenado")
        
        elapsed_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("✅ ¡CONFIGURACIÓN COMPLETADA!")
        logger.info(f"⏱️  Tiempo total: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
        logger.info("=" * 60)
        logger.info("🎉 La API está lista para ejecutarse:")
        logger.info("   python -m api.app")
        logger.info("=" * 60)
        
        self.print_summary()
        return True


model_loader = ModelLoader()
setup_manager = SetupManager()