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
# Todo esto debe ser dentro de me-verifier
# Si no quiere seguir los pasos anteirores puede usar este comando
python setup.py

# COmando para ejecutar
python -m api.app
```
