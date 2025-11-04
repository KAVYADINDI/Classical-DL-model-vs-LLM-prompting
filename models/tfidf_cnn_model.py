# models/tfidf_cnn_model.py
import os
import sys
import numpy as np
import pandas as pd
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
# Import for type hints but use tf.keras at runtime
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Input, Reshape, Conv1D, GlobalMaxPooling1D, Dense  # type: ignore
from evaluation.metrics_logger import log_metrics

def run_tfidf_cnn(train_csv: str, test_csv: str, max_features: int = 5000, epochs: int = 3, batch_size: int = 64):
    """Train a simple TF-IDF + CNN sentiment classifier and log metrics."""
    
    start_time = time.time()
    
    # Load data
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_sparse = vectorizer.fit_transform(train["clean_text"].fillna(""))
    X_test_sparse = vectorizer.transform(test["clean_text"].fillna(""))

    # Convert to dense arrays
    X_train = X_train_sparse.toarray()  # type: ignore
    X_test = X_test_sparse.toarray()  # type: ignore

    input_dim = X_train.shape[1]
    print(f"TF-IDF feature dimension: {input_dim}")

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train["detected_sentiment"])
    y_test = le.transform(test["detected_sentiment"])
    num_classes = len(le.classes_)

    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Define CNN model
    model = Sequential([
        Input(shape=(input_dim,)),
        Reshape((input_dim, 1)),
        Conv1D(filters=64, kernel_size=3, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        X_train, y_train_cat,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose="auto"
    )

    preds = model.predict(X_test)
    preds_label = np.argmax(preds, axis=1)
    acc = float(accuracy_score(y_test, preds_label))
    f1 = float(f1_score(y_test, preds_label, average="weighted"))
    
    training_time = time.time() - start_time

    log_metrics("TF-IDF_CNN", acc, f1, 
                training_time=training_time, 
                model_type="Classical ML",
                sample_size=len(test))
    print(f"✅ TF-IDF+CNN → acc={acc:.4f}, f1={f1:.4f}, time={training_time:.1f}s")
    return acc, f1

def train_tfidf_cnn(train_csv: str, test_csv: str, sample_limit: int = 0):
    """Wrapper function for compatibility with comparison script."""
    acc, f1 = run_tfidf_cnn(train_csv, test_csv)
    return acc, f1

if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    train_fp = os.path.join(base, "train.csv")
    test_fp = os.path.join(base, "test.csv")
    run_tfidf_cnn(train_fp, test_fp)
