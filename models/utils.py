"""
Utility functions for the ABSA project
"""
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from config import ID_TO_LABEL


def calculate_metrics(y_true, y_pred, labels=None):
    """
    Calculate comprehensive metrics for model evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        
    Returns:
        Dictionary containing all metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }
    
    return metrics


def print_metrics(metrics, model_name="Model"):
    """Pretty print metrics"""
    print(f"\n{'='*60}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"{'='*60}\n")


def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix", save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        title: Plot title
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def measure_inference_time(model, data, predict_fn, num_runs=3):
    """
    Measure average inference time
    
    Args:
        model: The model to test
        data: Input data
        predict_fn: Function to call for prediction
        num_runs: Number of runs to average
        
    Returns:
        Average inference time in seconds
    """
    times = []
    for _ in range(num_runs):
        start = time.time()
        predict_fn(model, data)
        end = time.time()
        times.append(end - start)
    
    return np.mean(times)


def save_results(results, filepath):
    """Save results to a file"""
    import json
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filepath}")


def load_results(filepath):
    """Load results from a file"""
    import json
    with open(filepath, 'r') as f:
        return json.load(f)
