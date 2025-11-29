"""
Centralized Metrics Logging System
Logs model performance metrics to CSV for easy comparison
"""
import os
import pandas as pd
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COMPARE_DIR


class MetricsLogger:
    """Log and compare metrics across different models and datasets"""
    
    def __init__(self, log_file="model_metrics.csv"):
        self.log_file = os.path.join(COMPARE_DIR, log_file)
        self._initialize_log()
    
    def _initialize_log(self):
        """Initialize CSV file if it doesn't exist"""
        if not os.path.exists(self.log_file):
            df = pd.DataFrame(columns=[
                'timestamp',
                'model_name',
                'data_type',
                'accuracy',
                'f1_score',
                'precision_macro',
                'recall_macro',
                'tp',
                'tn',
                'fp',
                'fn',
                'training_time',
                'inference_time_per_sample',
                'notes'
            ])
            df.to_csv(self.log_file, index=False)
            print(f"Created metrics log: {self.log_file}")
    
    def log_metrics(self, model_name, data_type, accuracy, f1_score, 
                   precision_macro, recall_macro, training_time, 
                   tp=None, tn=None, fp=None, fn=None,
                   inference_time_per_sample=None, notes=""):
        """
        Log metrics for a model run
        
        Args:
            model_name: Name of the model (e.g., "TF-IDF+CNN", "BERT_Finetune", "LLM_ZeroShot")
            data_type: Dataset used ("laptop", "restaurant", or "combined")
            accuracy: Model accuracy
            f1_score: Macro F1 score
            precision_macro: Macro precision
            recall_macro: Macro recall
            training_time: Training time in seconds
            tp: True Positives (sum across all classes)
            tn: True Negatives (sum across all classes)
            fp: False Positives (sum across all classes)
            fn: False Negatives (sum across all classes)
            inference_time_per_sample: Average inference time per sample in seconds
            notes: Additional notes about the run
        """
        # Create new entry
        new_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': model_name,
            'data_type': data_type,
            'accuracy': float(accuracy),
            'f1_score': float(f1_score),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'tp': int(tp) if tp is not None else None,
            'tn': int(tn) if tn is not None else None,
            'fp': int(fp) if fp is not None else None,
            'fn': int(fn) if fn is not None else None,
            'training_time': float(training_time),
            'inference_time_per_sample': float(inference_time_per_sample) if inference_time_per_sample else None,
            'notes': notes
        }
        
        # Append to CSV
        df = pd.read_csv(self.log_file)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(self.log_file, index=False)
        
        print(f"\n[OK] Logged metrics for {model_name} on {data_type} dataset")
        print(f"  Accuracy: {accuracy:.4f} | F1: {f1_score:.4f} | Training Time: {training_time:.2f}s")
    
    def get_comparison(self, data_type=None):
        """
        Get comparison of all logged models
        
        Args:
            data_type: Filter by dataset type (optional)
        
        Returns:
            DataFrame with all metrics
        """
        df = pd.read_csv(self.log_file)
        
        if data_type:
            df = df[df['data_type'] == data_type]
        
        return df
    
    def print_comparison(self, data_type=None):
        """Print formatted comparison table"""
        df = self.get_comparison(data_type)
        
        if len(df) == 0:
            print("No metrics logged yet.")
            return
        
        print(f"\n{'='*100}")
        print(f"Model Performance Comparison" + (f" - {data_type} dataset" if data_type else ""))
        print(f"{'='*100}")
        
        # Select key columns for display
        display_cols = ['model_name', 'data_type', 'accuracy', 'f1_score', 
                       'precision_macro', 'recall_macro', 'training_time']
        
        display_df = df[display_cols].copy()
        
        # Format numeric columns
        for col in ['accuracy', 'f1_score', 'precision_macro', 'recall_macro']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        display_df['training_time'] = display_df['training_time'].apply(lambda x: f"{x:.2f}s")
        
        print(display_df.to_string(index=False))
        print(f"{'='*100}\n")
    
    def get_best_model(self, metric='f1_score', data_type=None):
        """Get the best performing model based on a metric"""
        df = self.get_comparison(data_type)
        
        if len(df) == 0:
            return None
        
        best_idx = df[metric].idxmax()
        best_model = df.loc[best_idx]
        
        return best_model


# Global logger instance
metrics_logger = MetricsLogger()


def log_metrics(model_name, data_type, accuracy, f1_score, precision_macro, 
                recall_macro, training_time, tp=None, tn=None, fp=None, fn=None,
                inference_time_per_sample=None, notes=""):
    """
    Convenience function to log metrics
    """
    metrics_logger.log_metrics(
        model_name=model_name,
        data_type=data_type,
        accuracy=accuracy,
        f1_score=f1_score,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        training_time=training_time,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        inference_time_per_sample=inference_time_per_sample,
        notes=notes
    )


def print_comparison(data_type=None):
    """Convenience function to print comparison"""
    metrics_logger.print_comparison(data_type)


def get_comparison(data_type=None):
    """Convenience function to get comparison DataFrame"""
    return metrics_logger.get_comparison(data_type)
