import os
import cv2
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import setup_logger

logger = setup_logger(__name__)


def validate_image_file(img_file):
    """Valida si el archivo es una imagen válida"""
    valid_formats = ['.jpg', '.jpeg', '.png']
    return img_file.suffix.lower() in valid_formats


def load_image(img_file):
    try:
        img = cv2.imread(str(img_file))
        if img is None:
            logger.warning(f"No se pudo leer la imagen: {img_file}")
            return None
        logger.debug(f"Imagen cargada: {img_file}")
        return img
    except Exception as e:
        logger.error(f"Error al cargar la imagen {img_file}: {e}")
        return None


def detect_faces(img, face_cascade):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        logger.debug(f"Se detectaron {len(faces)} rostro(s)")
        return faces
    except Exception as e:
        logger.error(f"Error al detectar rostros: {e}")
        return []


def crop_and_save_face(img, face_coords, img_file, output_path, face_index):
    try:
        x, y, w, h = face_coords
        face = img[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (160, 160))
        
        output_file = output_path / f"{img_file.stem}_face{face_index}.jpg"
        cv2.imwrite(str(output_file), face_resized)
        logger.debug(f"Rostro recortado guardado: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error al guardar rostro {face_index} de {img_file}: {e}")
        return False


def is_directory_empty(directory):
    path = Path(directory)
    if not path.exists():
        logger.warning(f"El directorio no existe: {directory}")
        return True
    
    image_files = list(path.glob('*.jpg')) + list(path.glob('*.jpeg')) + list(path.glob('*.png'))
    return len(image_files) == 0


def download_missing_images(data_dir, label):
    label_dir = Path(data_dir) / label
    
    if not is_directory_empty(label_dir):
        logger.info(f"El directorio '{label}' contiene imágenes. Se omite la descarga.")
        return True
    
    logger.warning(f"El directorio '{label}' está vacío. Intentando descargar imágenes...")
    
    try:
        search_img_path = Path(__file__).parent.parent / 'search_img.py'
        
        if not search_img_path.exists():
            logger.error(f"search_img.py no encontrado en: {search_img_path}")
            return False
        
        logger.info(f"Ejecutando search_img.py para '{label}'...")
        result = subprocess.run(
            [sys.executable, str(search_img_path)],
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            logger.info(f"Imágenes descargadas exitosamente para '{label}'")
            return True
        else:
            logger.error(f"Error al descargar imágenes: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Tiempo de espera agotado en la descarga para '{label}'")
        return False
    except Exception as e:
        logger.error(f"Error descargando imágenes para '{label}': {e}")
        return False


def crop_faces(input_dir, output_dir, label):
    logger.info(f"Iniciando recorte de rostros para la etiqueta: {label}")
    
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            logger.error("Error al cargar el clasificador Haar Cascade")
            return
        
        input_path = Path(input_dir)
        output_path = Path(output_dir) / label
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directorio de entrada: {input_path}")
        logger.info(f"Directorio de salida: {output_path}")
        
        count = 0
        processed = 0
        failed = 0
        
        for img_file in input_path.glob('*'):
            if not validate_image_file(img_file):
                continue
            
            processed += 1
            img = load_image(img_file)
            if img is None:
                failed += 1
                continue
            
            faces = detect_faces(img, face_cascade)
            if len(faces) == 0:
                logger.warning(f"No se detectaron rostros en: {img_file}")
                continue
            
            for face_idx, face_coords in enumerate(faces):
                if crop_and_save_face(img, face_coords, img_file, output_path, face_idx):
                    count += 1
                else:
                    failed += 1
        
        logger.info(f"=== Resumen de Recorte para '{label}' ===")
        logger.info(f"Imágenes procesadas: {processed}")
        logger.info(f"Rostros recortados: {count}")
        logger.info(f"Operaciones fallidas: {failed}")
        logger.info(f"Salida guardada en: {output_path}")
        
    except Exception as e:
        logger.error(f"Error en el recorte de rostros para '{label}': {e}")
        raise


if __name__ == '__main__':
    logger.info("Iniciando script de recorte de rostros")
    
    try:
        base_dir = Path(__file__).parent.parent
        data_dir = base_dir / 'data'
        cropped_dir = data_dir / 'cropped'
        
        crop_faces(data_dir / 'me', cropped_dir, 'me')
        
        download_missing_images(data_dir, 'not_me')
        crop_faces(data_dir / 'not_me', cropped_dir, 'not_me')
        
        logger.info("Recorte de rostros completado exitosamente")
    except Exception as e:
        logger.error(f"Error en el script: {e}")
        raise