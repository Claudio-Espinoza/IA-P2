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
        logger.info(f"Loading face recognition model: {model_name}...")
        DeepFace.build_model(model_name)
        logger.info(f"Model loaded successfully: {model_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return False

def extract_embedding_from_image(img_file, model_name):
    try:
        img = cv2.imread(str(img_file))
        if img is None:
            logger.warning(f"Could not read image: {img_file}")
            return None
        
        result = DeepFace.represent(img, 
                                    model_name=model_name, 
                                    enforce_detection=False)
        
        embedding = result[0]['embedding']
        logger.debug(f"Embedding extracted from: {img_file}")
        return embedding
    except Exception as e:
        logger.error(f"Error extracting embedding from {img_file}: {e}")
        return None

def process_label_directory(label_dir, label_value, model_name):
    embeddings = []
    labels = []
    
    if not label_dir.exists():
        logger.warning(f"Directory does not exist: {label_dir}")
        return embeddings, labels
    
    label_name = label_dir.name
    logger.info(f"Processing '{label_name}' images from: {label_dir}")
    
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
    
    logger.info(f"Processed '{label_name}': {count} images extracted, {failed} failed")
    return embeddings, labels

def save_embeddings(embeddings, labels, output_file):
    try:
        if not embeddings:
            logger.warning("No embeddings to save")
            return False
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(output_file, 
                 embeddings=np.array(embeddings),
                 labels=np.array(labels))
        
        logger.info(f"Embeddings saved successfully to: {output_file}")
        logger.info(f"Total embeddings: {len(embeddings)}")
        logger.info(f"Positive (me): {sum(np.array(labels) == 1)}")
        logger.info(f"Negative (not_me): {sum(np.array(labels) == 0)}")
        return True
    except Exception as e:
        logger.error(f"Failed to save embeddings to {output_file}: {e}")
        return False

def extract_embeddings(cropped_dir, output_file='embeddings.npz'):
    logger.info("Starting embedding extraction...")
    try:
        model_name = "Facenet"
        
        if not load_model(model_name):
            logger.error("Failed to initialize model. Aborting.")
            return
        
        embeddings = []
        labels = []
        
        cropped_path = Path(cropped_dir)
        logger.info(f"Input directory: {cropped_path}")
        
        me_dir = cropped_path / 'me'
        me_embeddings, me_labels = process_label_directory(me_dir, 1, model_name)
        embeddings.extend(me_embeddings)
        labels.extend(me_labels)
        
        not_me_dir = cropped_path / 'not_me'
        not_me_embeddings, not_me_labels = process_label_directory(not_me_dir, 0, model_name)
        embeddings.extend(not_me_embeddings)
        labels.extend(not_me_labels)
        
        save_embeddings(embeddings, labels, output_file)
        logger.info("Embedding extraction completed successfully")
        
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        raise


if __name__ == '__main__':
    logger.info("Starting embedding extraction script")
    
    try:
        base_dir = Path(__file__).parent.parent
        data_dir = base_dir / 'data'
        output_path = data_dir / 'embeddings.npz'
        
        extract_embeddings(data_dir / 'cropped', output_file=output_path)
        logger.info("Script completed successfully")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise