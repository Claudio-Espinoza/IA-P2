# Me Verifier

Sistema de verificación facial basado en Machine Learning para identificar si una persona en una imagen es "yo" o no.

## Estructura del Proyecto

```
me-verifier/
├── api/
│   └── app.py              # Flask API (/healthz, /verify)
├── models/
│   ├── model.joblib        # Modelo entrenado
│   └── scaler.joblib       # Escalador de features
├── data/
│   ├── me/                 # Fotos de "yo" (crudas)
│   ├── not_me/             # Fotos de otras personas
│   └── cropped/            # Rostros recortados
│       ├── me/
│       └── not_me/
├── scripts/
│   ├── crop_faces.py       # Detección y recorte de rostros
│   ├── embeddings.py       # Extracción de embeddings
│   └── run_gunicorn.sh     # Script de producción
├── reports/
│   ├── metrics.json        # Métricas del modelo
│   └── confusion_matrix.png
├── train.py                # Entrenamiento del modelo
├── evaluate.py             # Evaluación y reportes
├── tests/test_api.py       # Tests de la API
├── .env.example
├── README.md
└── requirements.txt
```

## Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Copiar archivo de configuración
cp .env.example .env
```

## Uso

### 1. Preparar datos

Coloca imágenes en las carpetas correspondientes:
- `data/me/`: Tus fotos
- `data/not_me/`: Fotos de otras personas

### 2. Preprocesamiento

```bash
# Detectar y recortar rostros
python scripts/crop_faces.py

# Extraer embeddings faciales
python scripts/embeddings.py
```

### 3. Entrenar modelo

```bash
python train.py
```

### 4. Evaluar modelo

```bash
python evaluate.py
```

Los reportes se guardarán en `reports/`.

### 5. Ejecutar API

#### Desarrollo
```bash
python api/app.py
```

#### Producción
```bash
chmod +x scripts/run_gunicorn.sh
./scripts/run_gunicorn.sh
```

## API Endpoints

### GET /healthz
Health check del servicio.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

### POST /verify
Verifica si la imagen contiene a la persona objetivo.

**Request:**
- Content-Type: multipart/form-data
- Body: image file

**Response:**
```json
{
  "is_me": true,
  "confidence": 0.95
}
```

## Testing

```bash
pytest tests/
```

## Dependencias Principales

- Flask: API web
- scikit-learn: Machine learning
- OpenCV: Procesamiento de imágenes
- joblib: Serialización de modelos
- gunicorn: Servidor WSGI de producción

## Notas

- Los modelos pre-entrenados (FaceNet, VGGFace) deben configurarse en `scripts/embeddings.py`
- Se recomienda tener al menos 50 imágenes por clase para un buen rendimiento
- El modelo usa SVM con kernel RBF por defecto, pero puede cambiarse en `train.py`

## Licencia

MIT
