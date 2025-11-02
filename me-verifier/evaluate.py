import numpy as np
import joblib
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from logger import setup_logger

logger = setup_logger(__name__)

def load_model_and_scaler(model_path='models/model.joblib', scaler_path='models/scaler.joblib'):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        logger.info("Model and scaler loaded successfully")
        return model, scaler
    except FileNotFoundError as e:
        logger.error(f"Failed to load model or scaler: {e}")
        raise


def load_test_data(test_data_file='data/test_data.npz'):
    try:
        data = np.load(test_data_file)
        X = data['X_test']
        y = data['y_test']
        logger.info(f"Test data loaded: {len(y)} samples")
        return X, y
    except FileNotFoundError as e:
        logger.error(f"Test data file not found: {e}")
        raise


def print_test_set_statistics(y):
    total = len(y)
    positive = sum(y == 1)
    negative = sum(y == 0)
    
    logger.info(f"=== Test Set Statistics ===")
    logger.info(f"Total samples: {total}")
    logger.info(f"Positive (me): {positive} ({positive/total*100:.1f}%)")
    logger.info(f"Negative (not_me): {negative} ({negative/total*100:.1f}%)")
    
    return positive, negative


def predict_and_calculate_metrics(model, scaler, X, y):
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)
    
    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    return y_pred, y_proba, accuracy, cm, report


def print_classification_report(y, y_pred):
    logger.info("\nClassification Report:")
    report_str = classification_report(y, y_pred, target_names=['not_me', 'me'])
    for line in report_str.split('\n'):
        logger.info(line)


def validate_test_set_size(positive, negative):
    if positive < 10 or negative < 10:
        logger.warning("Recommend at least 10 samples per class for valid evaluation.")
        return False
    return True


def save_metrics(accuracy, cm, report, y):
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    metrics = {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'test_set_size': {
            'total': int(len(y)),
            'positive': int(sum(y == 1)),
            'negative': int(sum(y == 0))
        }
    }
    
    try:
        with open(reports_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {reports_dir / 'metrics.json'}")
    except IOError as e:
        logger.error(f"Failed to save metrics: {e}")
        raise


def plot_confusion_matrix(cm):
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['not_me', 'me'],
                    yticklabels=['not_me', 'me'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(reports_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        logger.info(f"Confusion matrix saved to {reports_dir / 'confusion_matrix.png'}")
    except Exception as e:
        logger.error(f"Failed to save confusion matrix: {e}")
        raise


def evaluate_model(test_data_file='data/test_data.npz'):
    logger.info("Starting model evaluation...")
    
    try:
        model, scaler = load_model_and_scaler()
        X, y = load_test_data(test_data_file)
        
        positive, negative = print_test_set_statistics(y)
        validate_test_set_size(positive, negative)
        
        y_pred, y_proba, accuracy, cm, report = predict_and_calculate_metrics(model, scaler, X, y)
        print_classification_report(y, y_pred)
        
        save_metrics(accuracy, cm, report, y)
        plot_confusion_matrix(cm)
        
        logger.info("Model evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    evaluate_model()