# models/bert_finetune.py
import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from evaluation.metrics_logger import log_metrics
from torch.utils.data import Dataset

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=32):  # Reduced from 64
        self.enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def run_bert_finetune(train_csv, test_csv, model_name="distilbert-base-uncased", epochs=1, batch_size=32, sample_size=1000):
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    
    # Sample data for faster training
    if sample_size:
        train = train.sample(n=min(sample_size, len(train)), random_state=42)
        test = test.sample(n=min(sample_size//4, len(test)), random_state=42)
        print(f"Using sample: {len(train)} training, {len(test)} test samples")

    le = LabelEncoder()
    y_train = le.fit_transform(train["detected_sentiment"])
    y_test = le.transform(test["detected_sentiment"])

    # Convert to lists explicitly using numpy
    y_train_list = np.array(y_train).tolist()
    y_test_list = np.array(y_test).tolist()

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    train_dataset = ReviewDataset(list(train["clean_text"].fillna("")), y_train_list, tokenizer)
    test_dataset = ReviewDataset(list(test["clean_text"].fillna("")), y_test_list, tokenizer)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))

    training_args = TrainingArguments(
        output_dir="./tmp-bert",
        per_device_train_batch_size=32,  # Increased from 16
        per_device_eval_batch_size=32,   # Increased from 16
        num_train_epochs=epochs,
        eval_strategy="no",
        logging_steps=10,               # Reduced logging frequency
        save_strategy="no",
        fp16=False,                     # Disable FP16 for CPU stability
        dataloader_num_workers=0,       # Single worker for simplicity
        remove_unused_columns=False,
        gradient_accumulation_steps=1,   # Reduced accumulation
        warmup_steps=10,                # Minimal warmup
        max_grad_norm=1.0,
        lr_scheduler_type="constant",    # Skip learning rate scheduling
        optim="adamw_torch",            # Use PyTorch optimizer
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, average="weighted"))
        }

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=test_dataset, 
        compute_metrics=compute_metrics
    )
    trainer.train()

    preds_output = trainer.predict(test_dataset)
    preds = np.argmax(preds_output.predictions, axis=1)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    log_metrics("BERT_Finetune", float(acc), float(f1))
    print(f"BERT Fine-tune → acc={acc:.4f}, f1={f1:.4f}")

if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    train_fp = os.path.join(base, "train.csv")
    test_fp = os.path.join(base, "test.csv")
    
    # Fast training with DistilBERT and sampling
    run_bert_finetune(train_fp, test_fp)
