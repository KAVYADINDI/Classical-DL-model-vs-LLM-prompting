"""
TF-IDF + CNN Model for Aspect-Based Sentiment Analysis
This implements the baseline deep learning model
"""
import pandas as pd
import numpy as np
import pickle
import time
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_SEED,
    MAX_FEATURES, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,
    CNN_FILTERS, KERNEL_SIZE, DROPOUT_RATE,
    BATCH_SIZE, EPOCHS, LEARNING_RATE
)
from models.utils import calculate_metrics, print_metrics, plot_confusion_matrix, save_results
from compare.metrics_logger import log_metrics

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class TFIDFCNNModel:
    """TF-IDF + CNN classifier for sentiment analysis"""
    
    def __init__(self, max_features=MAX_FEATURES, max_len=MAX_SEQUENCE_LENGTH):
        self.max_features = max_features
        self.max_len = max_len
        self.vectorizer = None
        self.model = None
        self.history = None
        
    def build_model(self, num_classes=3):
        """
        Build CNN architecture
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(self.max_len, 1)),
            
            # First Conv Block
            layers.Conv1D(CNN_FILTERS, KERNEL_SIZE, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(DROPOUT_RATE),
            
            # Second Conv Block
            layers.Conv1D(CNN_FILTERS // 2, KERNEL_SIZE, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(DROPOUT_RATE),
            
            # Global pooling
            layers.GlobalMaxPooling1D(),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(DROPOUT_RATE),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(DROPOUT_RATE / 2),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def fit_vectorizer(self, texts):
        """Fit TF-IDF vectorizer on training texts"""
        print("Fitting TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        self.vectorizer.fit(texts)
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
    def transform_texts(self, texts):
        """Transform texts to TF-IDF features and pad/truncate"""
        # Transform to TF-IDF
        tfidf_matrix = self.vectorizer.transform(texts).toarray()
        
        # Pad or truncate to max_len
        if tfidf_matrix.shape[1] > self.max_len:
            features = tfidf_matrix[:, :self.max_len]
        else:
            features = np.pad(
                tfidf_matrix,
                ((0, 0), (0, self.max_len - tfidf_matrix.shape[1])),
                mode='constant'
            )
        
        # Reshape for CNN (add channel dimension)
        features = features.reshape(features.shape[0], features.shape[1], 1)
        return features
    
    def train(self, X_train, y_train, X_val, y_val, dataset_name="model"):
        """Train the model"""
        print("\nTraining TF-IDF + CNN model...")
        
        # Fit vectorizer on training data
        self.fit_vectorizer(X_train)
        
        # Transform data
        X_train_features = self.transform_texts(X_train)
        X_val_features = self.transform_texts(X_val)
        
        print(f"Training features shape: {X_train_features.shape}")
        print(f"Validation features shape: {X_val_features.shape}")
        
        # Build model
        self.build_model(num_classes=len(np.unique(y_train)))
        print(self.model.summary())
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(MODELS_DIR, f'tfidf_cnn_{dataset_name}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train
        start_time = time.time()
        self.history = self.model.fit(
            X_train_features, y_train,
            validation_data=(X_val_features, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return training_time
    
    def predict(self, texts):
        """Predict sentiment for texts"""
        features = self.transform_texts(texts)
        predictions = self.model.predict(features, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, texts):
        """Get prediction probabilities"""
        features = self.transform_texts(texts)
        return self.model.predict(features, verbose=0)
    
    def save(self, dataset_name="model"):
        """Save model and vectorizer"""
        model_path = os.path.join(MODELS_DIR, f'tfidf_cnn_{dataset_name}.h5')
        vectorizer_path = os.path.join(MODELS_DIR, f'tfidf_vectorizer_{dataset_name}.pkl')
        
        self.model.save(model_path)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Vectorizer saved to: {vectorizer_path}")
    
    def load(self, dataset_name="model"):
        """Load model and vectorizer"""
        model_path = os.path.join(MODELS_DIR, f'tfidf_cnn_{dataset_name}.h5')
        vectorizer_path = os.path.join(MODELS_DIR, f'tfidf_vectorizer_{dataset_name}.pkl')
        
        self.model = keras.models.load_model(model_path)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        print(f"Model loaded from: {model_path}")


def load_data(dataset_name):
    """Load preprocessed data from data folder"""
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{dataset_name}_train.csv'))
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{dataset_name}_val.csv'))
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{dataset_name}_test.csv'))
    
    return train_df, val_df, test_df


def evaluate_model(model, X_test, y_test, dataset_name, save_plots=True):
    """Comprehensive model evaluation"""
    print(f"\n{'='*80}")
    print(f"Evaluating TF-IDF + CNN on {dataset_name} Test Set")
    print(f"{'='*80}")
    
    # Measure inference time
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    avg_inference_time = inference_time / len(X_test)
    
    print(f"\nTotal inference time: {inference_time:.4f} seconds")
    print(f"Average per sample: {avg_inference_time*1000:.4f} ms")
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    metrics['inference_time_total'] = inference_time
    metrics['inference_time_per_sample'] = avg_inference_time
    metrics['training_time'] = model.training_time if hasattr(model, 'training_time') else 0
    
    print_metrics(metrics, f"TF-IDF + CNN ({dataset_name})")
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    from config import ID_TO_LABEL
    for i, label in ID_TO_LABEL.items():
        print(f"{label:8s}: Precision={metrics['precision_per_class'][i]:.4f}, "
              f"Recall={metrics['recall_per_class'][i]:.4f}, "
              f"F1={metrics['f1_per_class'][i]:.4f}")
    
    # Confusion matrix
    if save_plots:
        plot_confusion_matrix(
            y_test, y_pred,
            labels=list(ID_TO_LABEL.values()),
            title=f"TF-IDF + CNN - {dataset_name} Dataset",
            save_path=os.path.join('evaluation', f'tfidf_cnn_{dataset_name}_confusion_matrix.png')
        )
    
    return metrics, y_pred


def plot_training_history(history, dataset_name):
    """Plot training history"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title(f'Model Loss - {dataset_name}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title(f'Model Accuracy - {dataset_name}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join('evaluation', f'tfidf_cnn_{dataset_name}_training.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main training and evaluation pipeline"""
    print("="*80)
    print("TF-IDF + CNN Model Training and Evaluation")
    print("="*80)
    
    # Choose dataset: 'laptop', 'restaurant', or 'combined'
    dataset_name = 'combined'  # Change this to train on different datasets
    
    print(f"\nTraining on {dataset_name} dataset...")
    
    # Load data
    train_df, val_df, test_df = load_data(dataset_name)
    
    X_train = train_df['combined_input'].values
    y_train = train_df['sentiment_id'].values
    
    X_val = val_df['combined_input'].values
    y_val = val_df['sentiment_id'].values
    
    X_test = test_df['combined_input'].values
    y_test = test_df['sentiment_id'].values
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(X_train)}")
    print(f"Val: {len(X_val)}")
    print(f"Test: {len(X_test)}")
    
    # Initialize and train model
    model = TFIDFCNNModel(max_features=MAX_FEATURES, max_len=MAX_SEQUENCE_LENGTH)
    training_time = model.train(X_train, y_train, X_val, y_val, dataset_name)
    model.training_time = training_time
    
    # Plot training history
    plot_training_history(model.history, dataset_name)
    
    # Evaluate on test set
    metrics, predictions = evaluate_model(model, X_test, y_test, dataset_name)
    
    # Calculate TP, TN, FP, FN from confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, predictions)
    tp = int(np.diag(cm).sum())  # Sum of diagonal elements
    total = int(cm.sum())
    fp = int(cm.sum(axis=0).sum() - tp)  # Sum of columns minus diagonal
    fn = int(cm.sum(axis=1).sum() - tp)  # Sum of rows minus diagonal
    tn = int(total - tp - fp - fn)
    
    # Log metrics to CSV
    log_metrics(
        model_name="TF-IDF+CNN",
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
        notes=f"CNN filters: {CNN_FILTERS}, Max features: {MAX_FEATURES}"
    )
    
    # Save model
    model.save(dataset_name)
    
    # Save results
    results = {
        'model': 'TF-IDF + CNN',
        'dataset': dataset_name,
        'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in metrics.items() if not k.endswith('_per_class')},
        'training_time': training_time,
        'params': {
            'max_features': MAX_FEATURES,
            'max_sequence_length': MAX_SEQUENCE_LENGTH,
            'cnn_filters': CNN_FILTERS,
            'kernel_size': KERNEL_SIZE,
            'dropout_rate': DROPOUT_RATE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS
        }
    }
    
    save_results(results, os.path.join('evaluation', f'tfidf_cnn_{dataset_name}_results.json'))
    
    print("\n" + "="*80)
    print("TF-IDF + CNN Training Complete!")
    print("="*80)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("\nNext Step: Train BERT Model")


if __name__ == "__main__":
    main()
