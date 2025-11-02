import os
import sys
from pathlib import Path
import shutil

sys.path.insert(0, os.path.dirname(__file__))

from bing_image_downloader import downloader
from logger import setup_logger

logger = setup_logger("descargar_fotos")

QUERY = "face white man"
LIMIT = 100
VALID_FORMATS = ['.jpg', '.jpeg', '.png']


def validate_downloaded_images(directory):
    try:
        logger.info(f"Validando formatos de imágenes en: {directory}")
        path = Path(directory)
        deleted = 0
        
        for img_file in path.glob('*'):
            if img_file.suffix.lower() not in VALID_FORMATS:
                try:
                    img_file.unlink()
                    logger.debug(f"Eliminado archivo inválido: {img_file.name}")
                    deleted += 1
                except Exception as e:
                    logger.error(f"Error al eliminar {img_file.name}: {e}")
        
        if deleted > 0:
            logger.warning(f"Se eliminaron {deleted} archivo(s) con formato inválido")
        
        valid_images = list(path.glob('*.jpg')) + list(path.glob('*.jpeg')) + list(path.glob('*.png'))
        logger.info(f"Imágenes válidas después de validación: {len(valid_images)}")
        return len(valid_images)
        
    except Exception as e:
        logger.error(f"Error validando imágenes: {e}")
        return 0


def move_images_from_temp(temp_dir, output_dir):
    try:
        temp_path = Path(temp_dir)
        output_path = Path(output_dir)
        
        if not temp_path.exists():
            logger.warning(f"Directorio temporal no encontrado: {temp_dir}")
            return 0
        
        moved = 0
        for img_file in temp_path.rglob('*'):
            if img_file.is_file() and img_file.suffix.lower() in VALID_FORMATS:
                try:
                    shutil.move(str(img_file), str(output_path / img_file.name))
                    moved += 1
                    logger.debug(f"Movida imagen: {img_file.name}")
                except Exception as e:
                    logger.error(f"Error moviendo {img_file.name}: {e}")
        
        try:
            shutil.rmtree(temp_path)
            logger.debug(f"Directorio temporal eliminado: {temp_dir}")
        except Exception as e:
            logger.error(f"Error eliminando directorio temporal: {e}")
        
        logger.info(f"Se movieron {moved} imágenes al directorio final")
        return moved
        
    except Exception as e:
        logger.error(f"Error moviendo imágenes: {e}")
        return 0


def download_images(limit, query, output_dir):
    try:
        logger.info(f"Iniciando descarga: '{query}' (límite={limit})")
        logger.info(f"Directorio temporal: {output_dir}")
        
        downloader.download(
            query,
            limit=limit,
            output_dir=output_dir, 
            adult_filter_off=False,
            force_replace=False,
            timeout=60,
            verbose=False
        )
        logger.info("Descarga completada")
        return True
        
    except Exception as e:
        logger.error(f"Error durante la descarga: {e}")
        return False


def define_output_path():
    base_dir = os.path.join(os.path.dirname(__file__), 'data', 'not_me')
    return base_dir


if __name__ == '__main__':
    temp_dir = None
    
    try:
        output_dir = define_output_path()
        os.makedirs(output_dir, exist_ok=True)
        
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp_download')
        
        logger.info("=== Iniciando descarga de imágenes ===")
        logger.info(f"Consulta: '{QUERY}'")
        logger.info(f"Límite: {LIMIT}")
        logger.info(f"Directorio final: {output_dir}")
        
        if download_images(LIMIT, QUERY, temp_dir):
            moved = move_images_from_temp(temp_dir, output_dir)
            
            if moved > 0:
                valid_count = validate_downloaded_images(output_dir)
                
                if valid_count > 0:
                    logger.info(f"Proceso completado exitosamente. {valid_count} imágenes válidas")
                else:
                    logger.warning("No se descargaron imágenes válidas")
            else:
                logger.warning("No se movieron imágenes")
        else:
            logger.error("La descarga falló")
            
    except Exception as e:
        logger.error(f"Error en el script: {e}")
        raise
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.debug("Directorio temporal limpiado")
            except Exception as e:
                logger.warning(f"No se pudo limpiar el directorio temporal: {e}")