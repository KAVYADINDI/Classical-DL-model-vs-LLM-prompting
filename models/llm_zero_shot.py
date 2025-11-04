# Zero-shot LLM for Aspect-Based Sentiment Analysis
# Simple and clean implementation

import json
import time
import requests
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Configuration
CSV_PATH = "data/test.csv"
TEXT_COL = "review_text"
LABEL_COL = "actual_sentiment"
MODEL_NAME = "llama3:8b"  # or "mistral:7b"
OLLAMA_URL = "http://localhost:11434/api/chat"
TIMEOUT = 60
N_SAMPLES = 5  # Start small for testing

# Simple prompt
SYSTEM_PROMPT = """You are an expert at sentiment analysis. 
Analyze the sentiment of the given review and respond with ONLY one word: positive, negative, or neutral."""

def load_data():
    """Load and validate data"""
    df = pd.read_csv(CSV_PATH)
    if N_SAMPLES:
        df = df.head(N_SAMPLES)
    return df

def get_sentiment(review_text):
    """Get sentiment prediction from LLM"""
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Review: {review_text}"}
            ],
            "stream": False
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        
        result = response.json()
        sentiment = result["message"]["content"].strip().lower()
        
        # Clean up response
        if "positive" in sentiment:
            return "positive"
        elif "negative" in sentiment:
            return "negative"
        else:
            return "neutral"
            
    except Exception as e:
        print(f"Error: {e}")
        return "neutral"  # Default fallback

def main():
    print("Starting zero-shot sentiment analysis...")
    
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} samples")
    
    # Get predictions
    predictions = []
    actuals = []
    
    for idx, row in enumerate(df.itertuples(), 1):
        review = str(row.review_text)
        actual = str(row.actual_sentiment).lower()
        
        print(f"Processing {idx}/{len(df)}...")
        pred = get_sentiment(review)
        
        predictions.append(pred)
        actuals.append(actual)
        
        time.sleep(0.5)  # Be gentle with API
    
    # Calculate metrics
    valid_pairs = [(a, p) for a, p in zip(actuals, predictions) 
                   if a in ['positive', 'negative', 'neutral']]
    
    if valid_pairs:
        true_labels, pred_labels = zip(*valid_pairs)
        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average='macro')
        
        print(f"\nResults:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1-Score: {f1:.3f}")
        
        # Save results
        results_df = df.copy()
        results_df['predicted_sentiment'] = predictions
        results_df.to_csv("data/zero_shot_results.csv", index=False)
        print("Results saved to data/zero_shot_results.csv")
    else:
        print("No valid predictions to evaluate")

if __name__ == "__main__":
    main()
