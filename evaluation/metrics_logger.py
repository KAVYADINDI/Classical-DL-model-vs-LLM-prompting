# evaluation/metrics_logger.py
import os
import pandas as pd
from datetime import datetime

METRICS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "metrics_results.csv")

def log_metrics(model_name: str, accuracy: float, f1_score: float, 
                training_time: float = 0.0, model_type: str = "", 
                sample_size: int = 0, file_path: str = METRICS_FILE):
    """
    Enhanced metrics logger with additional parameters for comprehensive comparison.
    
    Args:
        model_name: Name of the model
        accuracy: Accuracy score
        f1_score: F1 score
        training_time: Training/inference time in seconds (optional)
        model_type: Type of model (Classical ML, Transformer, LLM) (optional)
        sample_size: Number of samples used for evaluation (optional)
        file_path: Path to save metrics CSV
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "accuracy": float(accuracy),
        "f1_score": float(f1_score),
        "training_time_seconds": float(training_time) if training_time > 0 else None,
        "training_time_minutes": float(training_time / 60) if training_time > 0 else None,
        "model_type": model_type if model_type else None,
        "sample_size": int(sample_size) if sample_size > 0 else None,
    }
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(file_path, index=False)
    print(f"📊 Saved metrics for {model_name} → {file_path}")

def get_latest_comparison(file_path: str = METRICS_FILE) -> pd.DataFrame:
    """
    Get the latest metrics for each model for comparison.
    
    Returns:
        DataFrame with latest metrics per model
    """
    if not os.path.exists(file_path):
        print(f"No metrics file found at {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    # Get latest entry per model
    latest = df.sort_values("timestamp").groupby("model").tail(1)
    return latest.reset_index(drop=True)

def clear_metrics(file_path: str = METRICS_FILE):
    """Clear all metrics for a fresh comparison run."""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"🗑️ Cleared metrics file: {file_path}")
