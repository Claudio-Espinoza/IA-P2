import sys
import time
import traceback
from pathlib import Path

from flask import Flask, request, jsonify
import numpy as np
import cv2
from deepface import DeepFace

sys.path.insert(0, str(Path(__file__).parent.parent))
from logger import setup_logger
from api.init import model_loader
from api.config import (
    THRESHOLD, MAX_SIZE_MB, ALLOWED_EXTENSIONS, 
    FACENET_MODEL, DEBUG, HOST, PORT, API_VERSION, API_NAME
)

logger = setup_logger(__name__)

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    logger.debug("Solicitud GET /")
    
    return jsonify({
        'name': API_NAME,
        'version': API_VERSION,
        'status': 'running',
        'endpoints': {
            'info': 'GET /',
            'health': 'GET /healthz',
            'verify': 'POST /verify'
        }
    }), 200


@app.route('/healthz', methods=['GET'])
def healthz():
    logger.debug("Solicitud GET /healthz")
    
    status = {
        'status': 'healthy' if model_loader.is_ready() else 'degraded',
        'model_loaded': model_loader.model_loaded,
        'scaler_loaded': model_loader.scaler_loaded,
        'ready': model_loader.is_ready()
    }
    
    http_code = 200 if model_loader.is_ready() else 503
    return jsonify(status), http_code


@app.route('/verify', methods=['POST'])
def verify():
    logger.info("Solicitud POST /verify recibida")
    
    if not model_loader.is_ready():
        logger.error("Solicitud /verify rechazada: recursos no cargados")
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Por favor ejecuta: python train.py'
        }), 503
    
    if 'image' not in request.files:
        logger.warning("Solicitud /verify sin archivo 'image'")
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if not file.filename:
        logger.warning("Solicitud /verify con archivo sin nombre")
        return jsonify({'error': 'No filename provided'}), 400
    
    if '.' not in file.filename:
        logger.warning(f"Archivo sin extensión: {file.filename}")
        return jsonify({'error': 'File has no extension'}), 400
    
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Extensión no permitida: {file_ext}")
        return jsonify({
            'error': f'Only {list(ALLOWED_EXTENSIONS)} allowed'
        }), 400
    
    start_time = time.time()
    
    try:
        img_bytes = file.read()
        logger.debug(f"Imagen recibida: {len(img_bytes)} bytes")
        
        # Validar tamaño
        if len(img_bytes) > MAX_SIZE_MB * 1024 * 1024:
            logger.warning(f"Imagen demasiado grande: {len(img_bytes)} bytes")
            return jsonify({
                'error': f'Image too large (max {MAX_SIZE_MB}MB)'
            }), 400
        
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.warning("Error al decodificar imagen")
            return jsonify({'error': 'Invalid image format'}), 400
        
        logger.debug(f"Imagen decodificada: {img.shape}")
        
        try:
            logger.debug(f"Extrayendo embedding con modelo: {FACENET_MODEL}")
            embeddings_objs = DeepFace.represent(
                img_path=img,
                model_name=FACENET_MODEL,
                enforce_detection=False
            )
            
            if not embeddings_objs:
                logger.warning("No se detectó rostro en la imagen")
                return jsonify({
                    'error': 'No face detected in image',
                    'is_me': False
                }), 400
            
            embedding = embeddings_objs[0]['embedding']
            logger.debug(f"Embedding extraído: dimensión {len(embedding)}")
            
        except Exception as e:
            logger.error(f"Error extrayendo embedding: {e}")
            return jsonify({
                'error': f'Face extraction failed: {str(e)}'
            }), 400
        
        embedding_scaled = model_loader.scaler.transform([embedding])
        prediction = model_loader.model.predict(embedding_scaled)[0]
        probabilities = model_loader.model.predict_proba(embedding_scaled)[0]
        confidence = float(probabilities[prediction])
        
        logger.debug(f"Predicción: {prediction}, Confianza: {confidence:.3f}")
        
        is_me = bool(prediction == 1 and confidence >= THRESHOLD)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        result = {
            'is_me': is_me,
            'score': round(confidence, 4),
            'threshold': THRESHOLD,
            'timing_ms': round(elapsed_ms, 1),
            'model_version': API_VERSION
        }
        
        log_status = "✅ IDENTIFICADO" if is_me else "❌ NO IDENTIFICADO"
        logger.info(
            f"{log_status} - Score: {confidence:.3f} - "
            f"Tiempo: {elapsed_ms:.1f}ms"
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error en /verify: {traceback.format_exc()}")
        return jsonify({
            'error': f'Processing failed: {str(e)}'
        }), 500


@app.errorhandler(404)
def not_found(error):
    logger.warning(f"Ruta no encontrada: {request.path}")
    return jsonify({
        'error': 'Endpoint not found',
        'path': request.path
    }), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Error interno: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("=" * 60)
    
    if not model_loader.load_all():
        logger.error("No se pudieron cargar los recursos.")
        logger.error("Continuando en modo degradado...")
    
    logger.info(f"Estado de recursos: {model_loader.get_status()}")
    logger.info("=" * 60)
    logger.info("📍 Endpoints disponibles:")
    logger.info("   - GET  / (información)")
    logger.info("   - GET  /healthz (estado)")
    logger.info("   - POST /verify (verificación)")
    logger.info("=" * 60)
    logger.info(f"Servidor iniciado en http://{HOST}:{PORT}")
    logger.info("=" * 60)
    
    app.run(host=HOST, port=PORT, debug=DEBUG)