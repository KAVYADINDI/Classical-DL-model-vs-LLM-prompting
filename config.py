"""
Configuration file for Aspect-Based Sentiment Analysis project
"""
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PREPROCESS_DIR = os.path.join(BASE_DIR, 'preprocess')
EVALUATION_DIR = os.path.join(BASE_DIR, 'evaluation')
COMPARE_DIR = os.path.join(BASE_DIR, 'compare')

# Dataset paths
LAPTOP_DATA = os.path.join(DATA_DIR, 'Laptop.csv')
RESTAURANT_DATA = os.path.join(DATA_DIR, 'Restaurants.csv')

# Processed data paths (train/val/test splits will be saved in data folder)
PROCESSED_DATA_DIR = DATA_DIR

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Sentiment labels
SENTIMENT_LABELS = ['positive', 'negative', 'neutral']
LABEL_TO_ID = {'negative': 0, 'neutral': 1, 'positive': 2}
ID_TO_LABEL = {0: 'negative', 1: 'neutral', 2: 'positive'}

# TF-IDF + CNN parameters
MAX_FEATURES = 5000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128
CNN_FILTERS = 128
KERNEL_SIZE = 5
DROPOUT_RATE = 0.5

# BERT parameters (using DistilBERT for faster CPU training)
BERT_MODEL_NAME = 'distilbert-base-uncased'  # 60% faster than bert-base-uncased
MAX_LENGTH = 64  # Reduced from 128 for faster processing
BERT_BATCH_SIZE = 32  # Increased for fewer iterations
BERT_EPOCHS = 2  # Reduced from 3 for faster training
BERT_LEARNING_RATE = 3e-5  # Slightly higher for faster convergence

# LLM parameters
LOCAL_LLM_MODEL = 'llama2'  # Can be changed to mistral, llama3, etc.
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 100

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
