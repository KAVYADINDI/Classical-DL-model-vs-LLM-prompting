"""
Data Preprocessing Pipeline for ABSA Project
Handles text cleaning, tokenization, and train/val/test splits
"""
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LAPTOP_DATA, RESTAURANT_DATA, RANDOM_SEED, TEST_SIZE, VAL_SIZE,
    LABEL_TO_ID, ID_TO_LABEL, PROCESSED_DATA_DIR
)


class TextPreprocessor:
    """Text preprocessing utilities"""
    
    @staticmethod
    def clean_text(text):
        """
        Clean and normalize text
        - Convert to lowercase
        - Remove special characters
        - Remove extra whitespace
        - Keep basic punctuation for context
        """
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Keep letters, numbers, and basic punctuation
        text = re.sub(r'[^a-z0-9\s.,!?\'-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def create_input_with_aspect(review, aspect):
        """
        Combine review text with aspect term for better context
        Format: "[CLS] review text [SEP] aspect term [SEP]"
        """
        clean_review = TextPreprocessor.clean_text(review)
        clean_aspect = TextPreprocessor.clean_text(aspect)
        return f"{clean_review} [SEP] {clean_aspect}"


def load_and_prepare_data(data_path, dataset_name, text_column):
    """
    Load dataset and perform initial cleaning
    
    Args:
        data_path: Path to CSV file
        dataset_name: Name of dataset (for logging)
        text_column: Name of column containing review text
        
    Returns:
        Cleaned DataFrame
    """
    print(f"\nLoading {dataset_name} dataset...")
    df = pd.read_csv(data_path)
    print(f"Original shape: {df.shape}")
    
    # Standardize column names
    if text_column not in df.columns:
        # Find review column (might be 'review_text' or 'review_txt')
        text_cols = [col for col in df.columns if 'review' in col.lower()]
        if text_cols:
            df.rename(columns={text_cols[0]: 'review_text'}, inplace=True)
    else:
        df.rename(columns={text_column: 'review_text'}, inplace=True)
    
    # Rename aspect column if needed
    if 'Aspect Term' in df.columns:
        df.rename(columns={'Aspect Term': 'aspect_term'}, inplace=True)
    
    # Rename sentiment column
    if 'actual_aspect_based_sentiment' in df.columns:
        df.rename(columns={'actual_aspect_based_sentiment': 'sentiment'}, inplace=True)
    
    # Remove rows with 'conflict' sentiment (ambiguous label)
    original_len = len(df)
    df = df[df['sentiment'] != 'conflict'].copy()
    print(f"Removed {original_len - len(df)} conflict labels")
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    print(f"Shape after removing duplicates: {df.shape}")
    
    # Clean text
    print("Cleaning text...")
    df['clean_review'] = df['review_text'].apply(TextPreprocessor.clean_text)
    df['clean_aspect'] = df['aspect_term'].apply(TextPreprocessor.clean_text)
    
    # Create combined input (review + aspect)
    df['combined_input'] = df.apply(
        lambda x: TextPreprocessor.create_input_with_aspect(
            x['review_text'], x['aspect_term']
        ), axis=1
    )
    
    # Convert sentiment labels to numeric
    df['sentiment_id'] = df['sentiment'].map(LABEL_TO_ID)
    
    # Remove any rows with invalid sentiments
    df = df[df['sentiment_id'].notna()].copy()
    
    # Remove very short reviews (likely noise)
    df = df[df['clean_review'].str.len() > 5].copy()
    
    print(f"Final shape: {df.shape}")
    print(f"\nSentiment distribution:")
    print(df['sentiment'].value_counts())
    
    return df


def create_splits(df, dataset_name):
    """
    Create train, validation, and test splits
    Stratified by sentiment to maintain class distribution
    """
    print(f"\nCreating train/val/test splits for {dataset_name}...")
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df['sentiment_id']
    )
    
    # Second split: train vs val
    val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)  # Adjust val size
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=RANDOM_SEED,
        stratify=train_val['sentiment_id']
    )
    
    print(f"Train set: {len(train)} samples")
    print(f"Val set:   {len(val)} samples")
    print(f"Test set:  {len(test)} samples")
    
    print(f"\nTrain sentiment distribution:")
    print(train['sentiment'].value_counts())
    
    return train, val, test


def save_processed_data(train, val, test, dataset_name):
    """Save processed data to CSV files in data folder"""
    output_dir = PROCESSED_DATA_DIR
    
    print(f"\nSaving processed data for {dataset_name}...")
    
    train.to_csv(os.path.join(output_dir, f'{dataset_name}_train.csv'), index=False)
    val.to_csv(os.path.join(output_dir, f'{dataset_name}_val.csv'), index=False)
    test.to_csv(os.path.join(output_dir, f'{dataset_name}_test.csv'), index=False)
    
    print(f"Saved to {output_dir}/")
    
    # Save label mappings
    mappings = {
        'label_to_id': LABEL_TO_ID,
        'id_to_label': ID_TO_LABEL
    }
    
    with open(os.path.join(output_dir, 'label_mappings.pkl'), 'wb') as f:
        pickle.dump(mappings, f)
    
    print("Label mappings saved")


def get_data_statistics(train, val, test, dataset_name):
    """Print comprehensive statistics"""
    print(f"\n{'='*80}")
    print(f"{dataset_name} Dataset Statistics")
    print(f"{'='*80}")
    
    for split_name, split_df in [('Train', train), ('Val', val), ('Test', test)]:
        print(f"\n{split_name} Split:")
        print(f"  Total samples: {len(split_df)}")
        print(f"  Avg review length: {split_df['clean_review'].str.len().mean():.2f} chars")
        print(f"  Avg word count: {split_df['clean_review'].str.split().str.len().mean():.2f} words")
        print(f"  Unique aspects: {split_df['aspect_term'].nunique()}")
        
        # Sentiment percentages
        sentiment_pct = split_df['sentiment'].value_counts(normalize=True) * 100
        for sentiment, pct in sentiment_pct.items():
            print(f"    {sentiment}: {pct:.2f}%")


def main():
    """Main preprocessing pipeline - Combined dataset only"""
    print("="*80)
    print("Aspect-Based Sentiment Analysis - Data Preprocessing")
    print("="*80)
    
    # Load Laptop dataset
    print("\n[1/3] Loading Laptop dataset...")
    laptop_df = load_and_prepare_data(LAPTOP_DATA, "Laptop", "review_text")
    
    # Load Restaurant dataset
    print("\n[2/3] Loading Restaurant dataset...")
    restaurant_df = load_and_prepare_data(RESTAURANT_DATA, "Restaurant", "review_txt")
    
    # Create Combined dataset
    print("\n[3/3] Creating Combined Dataset")
    print("="*80)
    
    combined_df = pd.concat([laptop_df, restaurant_df], ignore_index=True)
    print(f"\nCombined dataset shape: {combined_df.shape}")
    print(f"Total samples: {len(combined_df)}")
    print(f"\nCombined sentiment distribution:")
    print(combined_df['sentiment'].value_counts())
    print(combined_df['sentiment'].value_counts(normalize=True) * 100)
    
    # Create train/val/test splits for combined dataset
    combined_train, combined_val, combined_test = create_splits(combined_df, "Combined")
    save_processed_data(combined_train, combined_val, combined_test, "combined")
    get_data_statistics(combined_train, combined_val, combined_test, "Combined")
    
    print("\n" + "="*80)
    print("Data Preprocessing Complete!")
    print("="*80)
    print("\nProcessed files saved in:", PROCESSED_DATA_DIR)
    print("  - combined_train.csv")
    print("  - combined_val.csv")
    print("  - combined_test.csv")
    print("  - label_mappings.pkl")
    print("\nNext Step: Model Training")


if __name__ == "__main__":
    main()
