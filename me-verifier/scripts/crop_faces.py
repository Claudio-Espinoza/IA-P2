import os
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import setup_logger

logger = setup_logger(__name__)


def validate_image_file(img_file):
    valid_formats = ['.jpg', '.jpeg', '.png']
    return img_file.suffix.lower() in valid_formats


def load_image(img_file):
    try:
        img = cv2.imread(str(img_file))
        if img is None:
            logger.warning(f"Could not read image: {img_file}")
            return None
        logger.debug(f"Image loaded: {img_file}")
        return img
    except Exception as e:
        logger.error(f"Error loading image {img_file}: {e}")
        return None


def detect_faces(img, face_cascade):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        logger.debug(f"Detected {len(faces)} face(s)")
        return faces
    except Exception as e:
        logger.error(f"Error detecting faces: {e}")
        return []


def crop_and_save_face(img, face_coords, img_file, output_path, face_index):
    try:
        x, y, w, h = face_coords
        face = img[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (160, 160))
        
        output_file = output_path / f"{img_file.stem}_face{face_index}.jpg"
        cv2.imwrite(str(output_file), face_resized)
        logger.debug(f"Saved cropped face: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving face {face_index} from {img_file}: {e}")
        return False


def crop_faces(input_dir, output_dir, label):
    logger.info(f"Starting face cropping for label: {label}")
    
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            logger.error("Failed to load Haar Cascade classifier")
            return
        
        input_path = Path(input_dir)
        output_path = Path(output_dir) / label
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Input directory: {input_path}")
        logger.info(f"Output directory: {output_path}")
        
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
                logger.warning(f"No faces detected in: {img_file}")
                continue
            
            for face_idx, face_coords in enumerate(faces):
                if crop_and_save_face(img, face_coords, img_file, output_path, face_idx):
                    count += 1
                else:
                    failed += 1
        
        logger.info(f"=== Cropping Summary for '{label}' ===")
        logger.info(f"Images processed: {processed}")
        logger.info(f"Faces cropped: {count}")
        logger.info(f"Failed operations: {failed}")
        logger.info(f"Output saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Face cropping failed for label '{label}': {e}")
        raise


if __name__ == '__main__':
    logger.info("Starting face cropping script")
    
    try:
        base_dir = Path(__file__).parent.parent
        data_dir = base_dir / 'data'
        cropped_dir = data_dir / 'cropped'
        
        crop_faces(data_dir / 'me', cropped_dir, 'me')
        crop_faces(data_dir / 'not_me', cropped_dir, 'not_me')
        
        logger.info("Face cropping completed successfully")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise