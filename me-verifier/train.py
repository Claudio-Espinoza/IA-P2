import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent))
from logger import setup_logger

logger = setup_logger(__name__)


def load_embeddings(embeddings_file='data/embeddings.npz'):
    try:
        logger.info(f"Cargando embeddings desde: {embeddings_file}")
        data = np.load(embeddings_file)
        X = data['embeddings']
        y = data['labels']
        logger.info(f"Se cargaron {len(X)} muestras")
        logger.info(f"Muestras positivas (yo): {sum(y == 1)}")
        logger.info(f"Muestras negativas (no yo): {sum(y == 0)}")
        return X, y
    except FileNotFoundError as e:
        logger.error(f"Archivo de embeddings no encontrado: {e}")
        raise
    except Exception as e:
        logger.error(f"Error al cargar embeddings: {e}")
        raise


def split_data(X, y, test_size=0.2, random_state=42):
    try:
        logger.info("Dividiendo datos en conjuntos de entrenamiento y prueba...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Muestras de entrenamiento: {len(X_train)}")
        logger.info(f"Muestras de prueba: {len(X_test)}")
        logger.info(f"Proporción de prueba: {test_size * 100:.1f}%")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error dividiendo datos: {e}")
        raise


def scale_data(X_train, X_test):
    try:
        logger.info("Escalando datos...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.debug("Datos escalados exitosamente")
        return X_train_scaled, X_test_scaled, scaler
    except Exception as e:
        logger.error(f"Error escalando datos: {e}")
        raise


def train_svm_model(X_train_scaled, y_train):
    try:
        logger.info("Entrenando modelo SVM...")
        logger.info("Parámetros: kernel='rbf', probability=True")
        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(X_train_scaled, y_train)
        logger.info("Modelo SVM entrenado exitosamente")
        return model
    except Exception as e:
        logger.error(f"Error entrenando modelo SVM: {e}")
        raise


def evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test):
    try:
        logger.info("Evaluando modelo...")
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        logger.info(f"=== Resultados de Evaluación ===")
        logger.info(f"Precisión en entrenamiento: {train_score:.4f}")
        logger.info(f"Precisión en prueba: {test_score:.4f}")
        if test_score < train_score:
            logger.warning(f"Posible sobreajuste detectado (diferencia: {(train_score - test_score):.4f})")
        return train_score, test_score
    except Exception as e:
        logger.error(f"Error evaluando modelo: {e}")
        raise


def save_model(model, scaler, models_dir='models'):
    try:
        logger.info("Guardando modelo y escalador...")
        models_path = Path(models_dir)
        models_path.mkdir(exist_ok=True, parents=True)
        model_file = models_path / 'model.joblib'
        scaler_file = models_path / 'scaler.joblib'
        joblib.dump(model, model_file)
        joblib.dump(scaler, scaler_file)
        logger.info(f"Modelo guardado en: {model_file}")
        logger.info(f"Escalador guardado en: {scaler_file}")
    except Exception as e:
        logger.error(f"Error guardando modelo: {e}")
        raise


def save_test_data(X_test, y_test, data_dir='data'):
    try:
        logger.info("Guardando datos de prueba...")
        data_path = Path(data_dir)
        data_path.mkdir(exist_ok=True, parents=True)
        test_file = data_path / 'test_data.npz'
        np.savez(test_file, X_test=X_test, y_test=y_test)
        logger.info(f"Datos de prueba guardados en: {test_file}")
        logger.info(f"Muestras de prueba guardadas: {len(y_test)}")
    except Exception as e:
        logger.error(f"Error guardando datos de prueba: {e}")
        raise


def train_model(embeddings_file='data/embeddings.npz'):
    logger.info("=== Iniciando entrenamiento del modelo ===")
    
    try:
        X, y = load_embeddings(embeddings_file)
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
        model = train_svm_model(X_train_scaled, y_train)
        evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
        save_model(model, scaler)
        save_test_data(X_test, y_test)
        logger.info("=== Entrenamiento completado exitosamente ===")
        return model, scaler
    except Exception as e:
        logger.error(f"Error en el entrenamiento: {e}")
        raise


if __name__ == '__main__':
    logger.info("Iniciando script de entrenamiento")
    
    try:
        train_model()
        logger.info("Script completado exitosamente")
    except Exception as e:
        logger.error(f"Error en el script: {e}")
        raise