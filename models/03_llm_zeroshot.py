"""
Zero-Shot Local LLM for Aspect-Based Sentiment Analysis
Using Ollama with Mistral or Phi models
"""
import pandas as pd
import numpy as np
import time
import sys
import os
import json
import requests
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PROCESSED_DATA_DIR, RANDOM_SEED, ID_TO_LABEL, LABEL_TO_ID
)
from models.utils import calculate_metrics, print_metrics, plot_confusion_matrix, save_results
from compare.metrics_logger import log_metrics

# Set random seed
np.random.seed(RANDOM_SEED)

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"


class ZeroShotLLM:
    """Zero-Shot LLM for sentiment analysis using Ollama"""
    
    def __init__(self, model_name="phi:2.7b", temperature=0, max_tokens=10, batch_size=50, timeout=180):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.timeout = timeout
        self.prompt_template = self._create_prompt_template()
        
    def _create_prompt_template(self):
        """Create zero-shot prompt template"""
        template = """You are an expert sentiment analyzer. Your task is to determine the sentiment (positive, negative, or neutral) expressed about a specific aspect in a review.

IMPORTANT RULES:
1. Focus ONLY on the mentioned aspect, ignore other aspects in the review
2. Positive = praise, satisfaction, or approval
3. Negative = criticism, dissatisfaction, or problems
4. Neutral = neither positive nor negative, just factual or mixed
5. Answer with ONLY ONE WORD: positive, negative, or neutral

Review: {review}
Aspect: {aspect}

Sentiment (one word only):"""        
        return template
    
    def _call_ollama(self, prompt, retries=3):
        """Call Ollama API with retry logic"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        for attempt in range(retries):
            try:
                response = requests.post(OLLAMA_API_URL, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                return result.get('response', '').strip().lower()
            except requests.exceptions.Timeout:
                if attempt < retries - 1:
                    print(f"\nTimeout on attempt {attempt + 1}, retrying...")
                    time.sleep(2)
                    continue
                else:
                    print(f"\nTimeout after {retries} attempts, using default")
                    return "neutral"
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    print(f"\nError on attempt {attempt + 1}: {e}, retrying...")
                    time.sleep(2)
                    continue
                else:
                    print(f"\nError after {retries} attempts: {e}")
                    return "neutral"
    
    def _parse_sentiment(self, response):
        """Parse LLM response to sentiment label"""
        response = response.lower().strip()
        
        # Remove common prefixes
        response = response.replace('sentiment:', '').strip()
        response = response.replace('answer:', '').strip()
        
        # Extract first word if multiple words returned
        first_word = response.split()[0] if response else ""
        
        # Check first word for exact match
        if first_word == 'positive':
            return 'positive'
        elif first_word == 'negative':
            return 'negative'
        elif first_word == 'neutral':
            return 'neutral'
        
        # Check if sentiment word appears in first word (e.g., "positive.")
        if 'positive' in first_word:
            return 'positive'
        elif 'negative' in first_word:
            return 'negative'
        elif 'neutral' in first_word:
            return 'neutral'
        
        # Fallback: check entire response
        if 'positive' in response and 'negative' not in response:
            return 'positive'
        elif 'negative' in response:
            return 'negative'
        else:
            return 'neutral'
    
    def predict_single(self, review, aspect):
        """Predict sentiment for a single review-aspect pair"""
        prompt = self.prompt_template.format(review=review, aspect=aspect)
        response = self._call_ollama(prompt)
        sentiment = self._parse_sentiment(response)
        return LABEL_TO_ID[sentiment]
    
    def predict(self, reviews, aspects, cache_file=None):
        """Predict sentiment for multiple review-aspect pairs with batch processing"""
        predictions = []
        start_idx = 0
        
        # Load cached predictions if available
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached predictions from {cache_file}...")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                predictions = cached_data['predictions']
                start_idx = len(predictions)
                print(f"Resuming from sample {start_idx}/{len(reviews)}")
        
        total_samples = len(reviews)
        print(f"\nRunning Zero-Shot predictions with {self.model_name}...")
        print(f"Total samples: {total_samples}, Batch size: {self.batch_size}")
        
        for i in tqdm(range(start_idx, total_samples), initial=start_idx, total=total_samples):
            review = reviews[i]
            aspect = aspects[i]
            
            pred = self.predict_single(review, aspect)
            predictions.append(pred)
            
            # Save progress every batch_size samples
            if cache_file and (i + 1) % self.batch_size == 0:
                self._save_cache(cache_file, predictions)
                print(f"\n[Batch {(i+1)//self.batch_size}] Saved progress: {i+1}/{total_samples} samples")
        
        # Final save
        if cache_file:
            self._save_cache(cache_file, predictions)
            print(f"\nFinal save: {len(predictions)}/{total_samples} samples")
        
        return np.array(predictions)
    
    def _save_cache(self, cache_file, predictions):
        """Save predictions cache to file"""
        with open(cache_file, 'w') as f:
            json.dump({'predictions': predictions}, f)
    
    def evaluate(self, test_df, dataset_name, cache_file=None):
        """Evaluate model on test set"""
        print(f"\n{'='*80}")
        print(f"Evaluating Zero-Shot LLM on {dataset_name} Test Set")
        print(f"{'='*80}")
        
        X_test = test_df['clean_review'].values
        aspects_test = test_df['clean_aspect'].values
        y_test = test_df['sentiment_id'].values
        
        # Measure inference time
        start_time = time.time()
        y_pred = self.predict(X_test, aspects_test, cache_file=cache_file)
        inference_time = time.time() - start_time
        avg_inference_time = inference_time / len(X_test)
        
        print(f"\nTotal inference time: {inference_time:.4f} seconds")
        print(f"Average per sample: {avg_inference_time*1000:.4f} ms")
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        metrics['inference_time_total'] = inference_time
        metrics['inference_time_per_sample'] = avg_inference_time
        
        print_metrics(metrics, f"Zero-Shot LLM ({dataset_name})")
        
        # Per-class metrics
        print("\nPer-Class Metrics:")
        for i, label in ID_TO_LABEL.items():
            print(f"{label:8s}: Precision={metrics['precision_per_class'][i]:.4f}, "
                  f"Recall={metrics['recall_per_class'][i]:.4f}, "
                  f"F1={metrics['f1_per_class'][i]:.4f}")
        
        # Confusion matrix
        plot_confusion_matrix(
            y_test, y_pred,
            labels=list(ID_TO_LABEL.values()),
            title=f"Zero-Shot LLM - {dataset_name} Dataset",
            save_path=os.path.join('evaluation', f'llm_zeroshot_{dataset_name}_confusion_matrix.png')
        )
        
        return metrics, y_pred


def load_data(dataset_name):
    """Load preprocessed test data"""
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{dataset_name}_test.csv'))
    return test_df


def main():
    """Main evaluation pipeline"""
    print("="*80)
    print("Zero-Shot LLM Evaluation")
    print("="*80)
    
    # Choose model: 'phi:2.7b' (faster) or 'mistral:7b' (more accurate)
    model_name = 'phi:2.7b'  # Using Phi for faster inference
    
    # Choose dataset
    dataset_name = 'combined'  # Change to 'laptop' or 'restaurant' if needed
    
    # Sample limit for testing (set to None for full dataset)
    sample_limit = 100  # Test with 100 samples first, set to None for all 1173
    
    print(f"\nUsing model: {model_name}")
    print(f"Dataset: {dataset_name}")
    if sample_limit:
        print(f"Sample limit: {sample_limit} (for faster testing)")
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        print("✓ Ollama is running")
    except:
        print("✗ Error: Ollama is not running!")
        print("Please start Ollama service and try again.")
        return
    
    # Load test data
    test_df = load_data(dataset_name)
    
    # Limit samples if specified
    if sample_limit and sample_limit < len(test_df):
        test_df = test_df.head(sample_limit)
        print(f"\nUsing first {len(test_df)} samples for faster testing")
    
    print(f"Test set size: {len(test_df)}")
    
    # Cache file for progress saving
    cache_file = os.path.join('evaluation', f'llm_zeroshot_{dataset_name}_cache.json')
    
    # Initialize model with batch processing
    llm = ZeroShotLLM(
        model_name=model_name, 
        temperature=0,  # Deterministic output
        max_tokens=10,  # Enough for one word + buffer
        batch_size=20,  # Save progress every 20 samples
        timeout=180  # 3 minutes timeout
    )
    
    # Evaluate
    metrics, predictions = llm.evaluate(test_df, dataset_name, cache_file=cache_file)
    
    # Calculate TP, TN, FP, FN from confusion matrix
    from sklearn.metrics import confusion_matrix
    y_test = test_df['sentiment_id'].values
    cm = confusion_matrix(y_test, predictions)
    tp = int(np.diag(cm).sum())
    total = int(cm.sum())
    fp = int(cm.sum(axis=0).sum() - tp)
    fn = int(cm.sum(axis=1).sum() - tp)
    tn = int(total - tp - fp - fn)
    
    # Log metrics to CSV
    log_metrics(
        model_name=f"LLM_ZeroShot_{model_name.replace(':', '_')}",
        data_type=dataset_name,
        accuracy=metrics['accuracy'],
        f1_score=metrics['f1_score'],
        precision_macro=metrics['precision'],
        recall_macro=metrics['recall'],
        training_time=0,  # No training for zero-shot
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        inference_time_per_sample=metrics['inference_time_per_sample'],
        notes=f"Zero-Shot with {model_name}, Temp: 0.1"
    )
    
    # Save results
    results = {
        'model': f'Zero-Shot LLM ({model_name})',
        'dataset': dataset_name,
        'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in metrics.items() if not k.endswith('_per_class')},
        'training_time': 0,
        'params': {
            'model_name': model_name,
            'temperature': 0.1,
            'max_tokens': 10,
            'approach': 'zero-shot'
        }
    }
    
    save_results(results, os.path.join('evaluation', f'llm_zeroshot_{dataset_name}_results.json'))
    
    # Save sample predictions
    sample_df = test_df.head(20).copy()
    sample_df['predicted_sentiment'] = [ID_TO_LABEL[p] for p in predictions[:20]]
    sample_df['true_sentiment'] = [ID_TO_LABEL[t] for t in y_test[:20]]
    sample_df[['clean_review', 'clean_aspect', 'true_sentiment', 'predicted_sentiment']].to_csv(
        os.path.join('evaluation', f'llm_zeroshot_{dataset_name}_samples.csv'),
        index=False
    )
    
    print("\n" + "="*80)
    print("Zero-Shot LLM Evaluation Complete!")
    print("="*80)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"\nNext Step: Implement Chain-of-Thought LLM")


if __name__ == "__main__":
    main()
