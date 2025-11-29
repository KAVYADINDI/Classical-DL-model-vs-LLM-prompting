# Aspect-Based Sentiment Analysis: Deep Learning vs Open-Source LLMs

## Project Overview
This project compares classical deep learning models with open-source large language models (LLMs) for aspect-based sentiment analysis (ABSA) on laptop and restaurant reviews. The goal is to demonstrate that properly prompted LLMs can outperform fine-tuned deep learning models with zero training time while providing explainable reasoning.

**Key Finding**: LLM Few-Shot achieved **81% accuracy**, outperforming fine-tuned BERT (76.7%) with no training required.

## Team Members
- Kavya Kalidindi
- Varsha Palamuri
- Balaji Kurapati

## Project Objectives
1. Compare Deep Learning (TF-IDF+CNN, BERT) vs LLM approaches
2. Demonstrate LLM superiority through prompt engineering
3. Evaluate accuracy, inference speed, and explainability
4. Generate presentation-ready visualizations

## Project Structure
```
prj/
├── data/                          # Datasets
│   ├── Laptop.csv                # Raw laptop reviews (2,280 samples)
│   ├── Restaurants.csv           # Raw restaurant reviews (3,585 samples)
│   ├── combined_train.csv        # Combined training set (70%)
│   ├── combined_val.csv          # Combined validation set (15%)
│   ├── combined_test.csv         # Combined test set (15%)
│   └── label_mappings.pkl        # Sentiment label mappings
│
├── preprocess/                    # Data preprocessing
│   ├── 01_data_exploration.py    # EDA and statistics
│   └── 02_data_preprocessing.py  # Text cleaning and splits
│
├── models/                        # Model implementations
│   ├── 01_tfidf_cnn.py           # TF-IDF + CNN baseline
│   ├── 02_bert_model.py          # Fine-tuned DistilBERT
│   ├── 03_llm_zeroshot.py        # Zero-shot LLM (Phi 2.7B)
│   ├── 04_llm_fewshot.py         # Few-shot LLM (Phi 2.7B)
│   ├── utils.py                  # Evaluation utilities
│   ├── bert_combined/            # Saved BERT checkpoint
│   ├── tfidf_cnn_combined_best.h5   # Best CNN model
│   └── tfidf_vectorizer_combined.pkl # TF-IDF vectorizer
│
├── evaluation/                    # Results and visualizations
│   ├── comparison_accuracy.png   # Accuracy comparison chart
│   ├── comparison_metrics.png    # F1/Precision/Recall chart
│   ├── comparison_speed.png      # Inference time comparison
│   ├── comparison_tradeoff.png   # Accuracy vs speed scatter
│   ├── model_comparison_summary.csv  # Summary table
│   ├── *_confusion_matrix.png    # Confusion matrices per model
│   └── *_results.json            # Detailed results per model
│
├── compare/                       # Cross-model comparison
│   ├── metrics_logger.py         # Centralized CSV logging
│   ├── create_comparison.py      # Generate visualizations
│   └── model_metrics.csv         # All model metrics
│
├── config.py                      # Centralized configuration
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Models Implemented

### 1. **TF-IDF + CNN** (Baseline)
- **Accuracy**: 53.5%
- **Training**: ~10 minutes
- **Inference**: 1.4 ms/sample
- Classical NLP with convolutional neural network

### 2. **Fine-tuned BERT** (DistilBERT)
- **Accuracy**: 76.7%
- **Training**: 110 minutes (2 epochs)
- **Inference**: 224 ms/sample
- Transformer-based deep learning

### 3. **LLM Zero-Shot** (Phi 2.7B via Ollama)
- **Accuracy**: 74%
- **Training**: 0 seconds
- **Inference**: 15.5 s/sample
- Prompt-engineered with 5 explicit rules

### 4. **LLM Few-Shot** (Phi 2.7B via Ollama) - **BEST MODEL**
- **Accuracy**: 81%
- **Training**: 0 seconds
- **Inference**: 22.5 s/sample
- 8 detailed examples with reasoning analysis

## Key Results

| Model | Accuracy | F1-Score | Training Time | Inference Time |
|-------|----------|----------|---------------|----------------|
| TF-IDF+CNN | 53.5% | 37.2% | 10 min | 1.4 ms |
| BERT (DistilBERT) | 76.7% | 75.6% | 110 min | 224 ms |
| LLM Zero-Shot | 74.0% | 76.8% | 0 s | 15.5 s |
| **LLM Few-Shot** | **81.0%** | **81.4%** | **0 s** | **22.5 s** |

**Conclusion**: LLM Few-Shot outperforms fine-tuned BERT by 4.3% with zero training time and explainable reasoning.

## Installation

### Prerequisites
- Python 3.9+
- Ollama (for local LLM inference)
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Clone repository
git clone <repository-url>
cd prj

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install Ollama and download Phi model
# Visit: https://ollama.ai/download
ollama pull phi:2.7b
```

## Usage

### Complete Pipeline
```bash
# 1. Preprocess data (combine laptop + restaurant datasets)
python preprocess/02_data_preprocessing.py

# 2. Train all models
python models/01_tfidf_cnn.py      # TF-IDF+CNN (~10 min)
python models/02_bert_model.py     # BERT (~110 min)
python models/03_llm_zeroshot.py   # Zero-Shot (~25 min for 100 samples)
python models/04_llm_fewshot.py    # Few-Shot (~37 min for 100 samples)

# 3. Generate comparison visualizations
python compare/create_comparison.py
```

### Quick Evaluation (Using Saved Models)
```bash
# Models are automatically saved after training
# Re-run evaluation without retraining:
python models/02_bert_model.py  # Loads from models/bert_combined/
python models/01_tfidf_cnn.py   # Loads from models/tfidf_cnn_combined_best.h5
```

## Visualizations
All comparison charts are generated in `evaluation/`:
- **Accuracy Comparison**: Bar chart of model performance
- **Metrics Comparison**: F1, Precision, Recall for all models
- **Speed Comparison**: Inference time benchmarks
- **Accuracy vs Speed**: Trade-off scatter plot
- **Confusion Matrices**: Per-model classification analysis

## Technical Details

### Dataset
- **Total Samples**: 5,865 (2,280 laptop + 3,585 restaurant)
- **Classes**: positive (53%), negative (28%), neutral (18%)
- **Split**: 70% train, 15% validation, 15% test
- **Preprocessing**: Text cleaning, aspect term extraction, stratified splits

### Hyperparameters
- **TF-IDF+CNN**: max_features=5000, CNN filters=[128,64], epochs=50
- **BERT**: DistilBERT-base, max_length=64, batch_size=32, epochs=2
- **LLM**: Phi 2.7B, temperature=0.1, timeout=180-240s

### Metrics
- Accuracy, F1-score (weighted), Precision (macro), Recall (macro)
- True/False Positives/Negatives per model
- Training time, Inference time per sample

## Key Learnings
1. **Prompt Engineering is Critical**: Improved Zero-Shot from 16% → 74% with better prompts
2. **Few-Shot Examples Matter**: 8 detailed examples boosted accuracy to 81%
3. **LLMs Provide Explainability**: Reasoning traces show decision logic
4. **No Training Advantage**: LLMs skip 110-minute BERT training time
5. **CPU-Friendly**: Phi 2.7B runs efficiently on CPU (15-23s per sample)

## Citation
```bibtex
@misc{absa_dl_vs_llm_2025,
  title={Aspect-Based Sentiment Analysis: Deep Learning vs Open-Source LLMs},
  author={Kalidindi, Kavya and Palamuri, Varsha and Kurapati, Balaji},
  year={2025}
}
```

## License
This project is for academic purposes.

## Acknowledgments
- Hugging Face Transformers for BERT implementation
- Ollama for local LLM inference
- SemEval datasets for ABSA benchmarks
