"""
Fine-tuned BERT Model for Aspect-Based Sentiment Analysis
"""
import pandas as pd
import numpy as np
import time
import sys
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_SEED,
    BERT_MODEL_NAME, MAX_LENGTH, BERT_BATCH_SIZE, 
    BERT_EPOCHS, BERT_LEARNING_RATE
)
from models.utils import calculate_metrics, print_metrics, plot_confusion_matrix, save_results
from compare.metrics_logger import log_metrics

# Set random seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class ABSADataset(Dataset):
    """Dataset for Aspect-Based Sentiment Analysis"""
    
    def __init__(self, texts, aspects, labels, tokenizer, max_length):
        self.texts = texts
        self.aspects = aspects
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        aspect = str(self.aspects[idx])
        label = self.labels[idx]
        
        # Combine text and aspect with [SEP] token
        combined_text = f"{text} [SEP] {aspect}"
        
        # Tokenize
        encoding = self.tokenizer(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTClassifier:
    """BERT-based classifier for sentiment analysis (supports BERT/DistilBERT)"""
    
    def __init__(self, num_classes=3, model_name=BERT_MODEL_NAME):
        self.num_classes = num_classes
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = device
        
    def build_model(self):
        """Initialize BERT/DistilBERT model"""
        print(f"Loading {self.model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes
        )
        self.model.to(self.device)
        return self.model
    
    def create_data_loader(self, texts, aspects, labels, batch_size, shuffle=True):
        """Create DataLoader"""
        dataset = ABSADataset(
            texts=texts,
            aspects=aspects,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=MAX_LENGTH
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )
    
    def train_model(self, train_texts, train_aspects, train_labels, 
              val_texts, val_aspects, val_labels, dataset_name="model"):
        """Train the BERT model"""
        print("\nTraining BERT model...")
        
        # Build model
        self.build_model()
        
        # Create data loaders
        train_loader = self.create_data_loader(
            train_texts, train_aspects, train_labels, BERT_BATCH_SIZE, shuffle=True
        )
        val_loader = self.create_data_loader(
            val_texts, val_aspects, val_labels, BERT_BATCH_SIZE, shuffle=False
        )
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=BERT_LEARNING_RATE, eps=1e-8)
        
        total_steps = len(train_loader) * BERT_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Loss function
        loss_fn = nn.CrossEntropyLoss().to(self.device)
        
        # Training loop
        best_val_accuracy = 0
        training_stats = []
        
        start_time = time.time()
        
        for epoch in range(BERT_EPOCHS):
            print(f'\n{"="*60}')
            print(f'Epoch {epoch + 1}/{BERT_EPOCHS}')
            print(f'{"="*60}')
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            train_iterator = tqdm(train_loader, desc="Training")
            for batch in train_iterator:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                train_iterator.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Validation phase
            val_accuracy, val_loss = self.evaluate_loader(val_loader, loss_fn)
            
            print(f'\nTrain Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}')
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save(dataset_name)
                print(f'[OK] Saved best model (Val Acc: {val_accuracy:.4f})')
            
            training_stats.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return training_time, training_stats
    
    def evaluate_loader(self, data_loader, loss_fn=None):
        """Evaluate model on a data loader"""
        self.model.eval()
        
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss().to(self.device)
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(data_loader)
        val_accuracy = val_correct / val_total
        
        return val_accuracy, avg_val_loss
    
    def predict(self, texts, aspects, batch_size=BERT_BATCH_SIZE):
        """Predict sentiment for texts"""
        self.model.eval()
        
        # Create dummy labels for dataset
        dummy_labels = np.zeros(len(texts), dtype=np.int64)
        
        data_loader = self.create_data_loader(
            texts, aspects, dummy_labels, batch_size, shuffle=False
        )
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                _, predicted = torch.max(logits, 1)
                all_predictions.extend(predicted.cpu().numpy())
        
        return np.array(all_predictions)
    
    def save(self, dataset_name="model"):
        """Save model and tokenizer"""
        save_dir = os.path.join(MODELS_DIR, f'bert_{dataset_name}')
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
    
    def load(self, dataset_name="model"):
        """Load model and tokenizer"""
        load_dir = os.path.join(MODELS_DIR, f'bert_{dataset_name}')
        
        self.model = AutoModelForSequenceClassification.from_pretrained(load_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        self.model.to(self.device)
        
        print(f"Model loaded from: {load_dir}")


def load_data(dataset_name):
    """Load preprocessed data from data folder"""
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{dataset_name}_train.csv'))
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{dataset_name}_val.csv'))
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{dataset_name}_test.csv'))
    
    return train_df, val_df, test_df


def evaluate_model(model, test_df, dataset_name):
    """Comprehensive model evaluation"""
    print(f"\n{'='*80}")
    print(f"Evaluating BERT on {dataset_name} Test Set")
    print(f"{'='*80}")
    
    X_test = test_df['clean_review'].values
    aspects_test = test_df['clean_aspect'].values
    y_test = test_df['sentiment_id'].values
    
    # Measure inference time
    start_time = time.time()
    y_pred = model.predict(X_test, aspects_test)
    inference_time = time.time() - start_time
    avg_inference_time = inference_time / len(X_test)
    
    print(f"\nTotal inference time: {inference_time:.4f} seconds")
    print(f"Average per sample: {avg_inference_time*1000:.4f} ms")
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    metrics['inference_time_total'] = inference_time
    metrics['inference_time_per_sample'] = avg_inference_time
    
    print_metrics(metrics, f"BERT ({dataset_name})")
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    from config import ID_TO_LABEL
    for i, label in ID_TO_LABEL.items():
        print(f"{label:8s}: Precision={metrics['precision_per_class'][i]:.4f}, "
              f"Recall={metrics['recall_per_class'][i]:.4f}, "
              f"F1={metrics['f1_per_class'][i]:.4f}")
    
    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        labels=list(ID_TO_LABEL.values()),
        title=f"BERT - {dataset_name} Dataset",
        save_path=os.path.join('evaluation', f'bert_{dataset_name}_confusion_matrix.png')
    )
    
    return metrics, y_pred


def main():
    """Main training and evaluation pipeline"""
    print("="*80)
    print("BERT Model Training and Evaluation")
    print("="*80)
    
    # Choose dataset
    dataset_name = 'combined'  # Change to 'laptop' or 'restaurant' if needed
    
    print(f"\nTraining on {dataset_name} dataset...")
    
    # Load data
    train_df, val_df, test_df = load_data(dataset_name)
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")
    
    # Initialize model
    model = BERTClassifier(num_classes=3)
    
    # Train model
    training_time, training_stats = model.train_model(
        train_df['clean_review'].values,
        train_df['clean_aspect'].values,
        train_df['sentiment_id'].values,
        val_df['clean_review'].values,
        val_df['clean_aspect'].values,
        val_df['sentiment_id'].values,
        dataset_name
    )
    
    # Load best model
    model.load(dataset_name)
    
    # Evaluate on test set
    metrics, predictions = evaluate_model(model, test_df, dataset_name)
    
    # Calculate TP, TN, FP, FN from confusion matrix
    from sklearn.metrics import confusion_matrix
    y_test = test_df['sentiment_id'].values
    cm = confusion_matrix(y_test, predictions)
    tp = int(np.diag(cm).sum())  # Sum of diagonal elements
    total = int(cm.sum())
    fp = int(cm.sum(axis=0).sum() - tp)  # Sum of columns minus diagonal
    fn = int(cm.sum(axis=1).sum() - tp)  # Sum of rows minus diagonal
    tn = int(total - tp - fp - fn)
    
    # Log metrics to CSV
    log_metrics(
        model_name="BERT_Finetune",
        data_type=dataset_name,
        accuracy=metrics['accuracy'],
        f1_score=metrics['f1_score'],
        precision_macro=metrics['precision'],
        recall_macro=metrics['recall'],
        training_time=training_time,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        inference_time_per_sample=metrics['inference_time_per_sample'],
        notes=f"Model: {BERT_MODEL_NAME}, Epochs: {BERT_EPOCHS}"
    )
    
    # Save results
    results = {
        'model': 'BERT',
        'dataset': dataset_name,
        'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in metrics.items() if not k.endswith('_per_class')},
        'training_time': training_time,
        'params': {
            'model_name': BERT_MODEL_NAME,
            'max_length': MAX_LENGTH,
            'batch_size': BERT_BATCH_SIZE,
            'epochs': BERT_EPOCHS,
            'learning_rate': BERT_LEARNING_RATE
        }
    }
    
    save_results(results, os.path.join('evaluation', f'bert_{dataset_name}_results.json'))
    
    print("\n" + "="*80)
    print("BERT Training Complete!")
    print("="*80)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("\nNext Step: Setup Local LLM")


if __name__ == "__main__":
    main()
