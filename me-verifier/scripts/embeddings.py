import numpy as np
from pathlib import Path
import cv2
import sys
from deepface import DeepFace

sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import setup_logger

logger = setup_logger(__name__)


def load_model(model_name="Facenet"):
    try:
        logger.info(f"Cargando modelo de reconocimiento facial: {model_name}...")
        DeepFace.build_model(model_name)
        logger.info(f"Modelo cargado exitosamente: {model_name}")
        return True
    except Exception as e:
        logger.error(f"Error al cargar el modelo {model_name}: {e}")
        return False


def extract_embedding_from_image(img_file, model_name):
    try:
        img = cv2.imread(str(img_file))
        if img is None:
            logger.warning(f"No se pudo leer la imagen: {img_file}")
            return None
        
        result = DeepFace.represent(img, 
                                    model_name=model_name, 
                                    enforce_detection=False)
        
        embedding = result[0]['embedding']
        logger.debug(f"Embedding extraído de: {img_file}")
        return embedding
    except Exception as e:
        logger.error(f"Error al extraer embedding de {img_file}: {e}")
        return None


def process_label_directory(label_dir, label_value, model_name):
    embeddings = []
    labels = []
    
    if not label_dir.exists():
        logger.warning(f"El directorio no existe: {label_dir}")
        return embeddings, labels
    
    label_name = label_dir.name
    logger.info(f"Procesando imágenes de '{label_name}' desde: {label_dir}")
    
    count = 0
    failed = 0
    
    for img_file in label_dir.glob('*.jpg'):
        embedding = extract_embedding_from_image(img_file, model_name)
        
        if embedding is not None:
            embeddings.append(embedding)
            labels.append(label_value)
            count += 1
        else:
            failed += 1
    
    logger.info(f"Procesadas imágenes de '{label_name}': {count} extraídos, {failed} fallidos")
    return embeddings, labels


def save_embeddings(embeddings, labels, output_file):
    try:
        if not embeddings:
            logger.warning("No hay embeddings para guardar")
            return False
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(output_file, 
                 embeddings=np.array(embeddings),
                 labels=np.array(labels))
        
        logger.info(f"Embeddings guardados exitosamente en: {output_file}")
        logger.info(f"Total de embeddings: {len(embeddings)}")
        logger.info(f"Positivos (yo): {sum(np.array(labels) == 1)}")
        logger.info(f"Negativos (no yo): {sum(np.array(labels) == 0)}")
        return True
    except Exception as e:
        logger.error(f"Error al guardar embeddings en {output_file}: {e}")
        return False


def extract_embeddings(cropped_dir, output_file='embeddings.npz'):
    logger.info("Iniciando extracción de embeddings...")
    try:
        model_name = "Facenet"
        
        if not load_model(model_name):
            logger.error("Error al inicializar el modelo. Abortando.")
            return
        
        embeddings = []
        labels = []
        
        cropped_path = Path(cropped_dir)
        logger.info(f"Directorio de entrada: {cropped_path}")
        
        me_dir = cropped_path / 'me'
        me_embeddings, me_labels = process_label_directory(me_dir, 1, model_name)
        embeddings.extend(me_embeddings)
        labels.extend(me_labels)
        
        not_me_dir = cropped_path / 'not_me'
        not_me_embeddings, not_me_labels = process_label_directory(not_me_dir, 0, model_name)
        embeddings.extend(not_me_embeddings)
        labels.extend(not_me_labels)
        
        save_embeddings(embeddings, labels, output_file)
        logger.info("Extracción de embeddings completada exitosamente")
        
    except Exception as e:
        logger.error(f"Error en la extracción de embeddings: {e}")
        raise


if __name__ == '__main__':
    logger.info("Iniciando script de extracción de embeddings")
    
    try:
        base_dir = Path(__file__).parent.parent
        data_dir = base_dir / 'data'
        output_path = data_dir / 'embeddings.npz'
        
        extract_embeddings(data_dir / 'cropped', output_file=output_path)
        logger.info("Script completado exitosamente")
    except Exception as e:
        logger.error(f"Error en el script: {e}")
        raise