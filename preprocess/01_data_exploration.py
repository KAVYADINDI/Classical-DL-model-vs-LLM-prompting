"""
Data Exploration and Analysis for ABSA Project
This script performs exploratory data analysis on Laptop and Restaurant datasets
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config import LAPTOP_DATA, RESTAURANT_DATA, SENTIMENT_LABELS

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_datasets():
    """Load both datasets"""
    print("Loading datasets...")
    laptop_df = pd.read_csv(LAPTOP_DATA)
    restaurant_df = pd.read_csv(RESTAURANT_DATA)
    
    print(f"Laptop dataset shape: {laptop_df.shape}")
    print(f"Restaurant dataset shape: {restaurant_df.shape}")
    
    return laptop_df, restaurant_df


def explore_dataset(df, dataset_name):
    """Comprehensive exploration of a single dataset"""
    print(f"\n{'='*80}")
    print(f"Exploring {dataset_name} Dataset")
    print(f"{'='*80}\n")
    
    # Basic info
    print("Dataset Info:")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Sentiment distribution
    print(f"\nSentiment Distribution:")
    sentiment_counts = df['actual_aspect_based_sentiment'].value_counts()
    print(sentiment_counts)
    print(f"\nPercentage distribution:")
    print(df['actual_aspect_based_sentiment'].value_counts(normalize=True) * 100)
    
    # Review text statistics
    df['review_length'] = df.iloc[:, 1].astype(str).apply(len)
    df['word_count'] = df.iloc[:, 1].astype(str).apply(lambda x: len(x.split()))
    
    print(f"\nReview Text Statistics:")
    print(f"Average review length (characters): {df['review_length'].mean():.2f}")
    print(f"Average word count: {df['word_count'].mean():.2f}")
    print(f"Max review length: {df['review_length'].max()}")
    print(f"Min review length: {df['review_length'].min()}")
    
    # Aspect terms analysis
    print(f"\nTop 10 Most Common Aspect Terms:")
    aspect_counts = df['Aspect Term'].value_counts().head(10)
    print(aspect_counts)
    
    return df


def visualize_dataset(df, dataset_name):
    """Create visualizations for the dataset"""
    print(f"\nCreating visualizations for {dataset_name}...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset_name} Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Sentiment distribution
    sentiment_counts = df['actual_aspect_based_sentiment'].value_counts()
    axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values, 
                   color=['#ff6b6b', '#95e1d3', '#4ecdc4'])
    axes[0, 0].set_title('Sentiment Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Sentiment')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(sentiment_counts.values):
        axes[0, 0].text(i, v + 50, str(v), ha='center', va='bottom')
    
    # 2. Review length distribution
    axes[0, 1].hist(df['review_length'], bins=50, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Review Length Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Character Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(df['review_length'].mean(), color='red', 
                       linestyle='--', label=f"Mean: {df['review_length'].mean():.0f}")
    axes[0, 1].legend()
    
    # 3. Word count distribution
    axes[1, 0].hist(df['word_count'], bins=50, color='lightcoral', edgecolor='black')
    axes[1, 0].set_title('Word Count Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Word Count')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(df['word_count'].mean(), color='darkred', 
                       linestyle='--', label=f"Mean: {df['word_count'].mean():.0f}")
    axes[1, 0].legend()
    
    # 4. Top aspects by sentiment
    top_aspects = df['Aspect Term'].value_counts().head(10).index
    aspect_sentiment = df[df['Aspect Term'].isin(top_aspects)].groupby(
        ['Aspect Term', 'actual_aspect_based_sentiment']
    ).size().unstack(fill_value=0)
    
    aspect_sentiment.plot(kind='bar', stacked=True, ax=axes[1, 1], 
                         color=['#ff6b6b', '#95e1d3', '#4ecdc4'])
    axes[1, 1].set_title('Top 10 Aspects by Sentiment', fontweight='bold')
    axes[1, 1].set_xlabel('Aspect Term')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend(title='Sentiment')
    
    plt.tight_layout()
    plt.savefig(f'evaluation/{dataset_name}_eda.png', dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: evaluation/{dataset_name}_eda.png")
    plt.show()


def compare_datasets(laptop_df, restaurant_df):
    """Compare both datasets"""
    print(f"\n{'='*80}")
    print("Comparing Both Datasets")
    print(f"{'='*80}\n")
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Dataset Comparison', fontsize=16, fontweight='bold')
    
    # Sentiment distribution comparison
    laptop_sentiment = laptop_df['actual_aspect_based_sentiment'].value_counts()
    restaurant_sentiment = restaurant_df['actual_aspect_based_sentiment'].value_counts()
    
    x = np.arange(len(SENTIMENT_LABELS))
    width = 0.35
    
    laptop_counts = [laptop_sentiment.get(label, 0) for label in SENTIMENT_LABELS]
    restaurant_counts = [restaurant_sentiment.get(label, 0) for label in SENTIMENT_LABELS]
    
    axes[0].bar(x - width/2, laptop_counts, width, label='Laptop', color='#3498db')
    axes[0].bar(x + width/2, restaurant_counts, width, label='Restaurant', color='#e74c3c')
    axes[0].set_title('Sentiment Distribution Comparison', fontweight='bold')
    axes[0].set_xlabel('Sentiment')
    axes[0].set_ylabel('Count')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(SENTIMENT_LABELS)
    axes[0].legend()
    
    # Review length comparison
    axes[1].boxplot([laptop_df['review_length'], restaurant_df['review_length']], 
                    labels=['Laptop', 'Restaurant'])
    axes[1].set_title('Review Length Comparison', fontweight='bold')
    axes[1].set_ylabel('Character Count')
    
    plt.tight_layout()
    plt.savefig('evaluation/dataset_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison visualization saved to: evaluation/dataset_comparison.png")
    plt.show()
    
    # Print comparison statistics
    print("\nComparison Statistics:")
    print(f"{'Metric':<30} {'Laptop':<15} {'Restaurant':<15}")
    print(f"{'-'*60}")
    print(f"{'Total records':<30} {len(laptop_df):<15} {len(restaurant_df):<15}")
    print(f"{'Avg review length':<30} {laptop_df['review_length'].mean():<15.2f} {restaurant_df['review_length'].mean():<15.2f}")
    print(f"{'Avg word count':<30} {laptop_df['word_count'].mean():<15.2f} {restaurant_df['word_count'].mean():<15.2f}")
    print(f"{'Unique aspects':<30} {laptop_df['Aspect Term'].nunique():<15} {restaurant_df['Aspect Term'].nunique():<15}")


def check_data_quality(df, dataset_name):
    """Check data quality issues"""
    print(f"\nData Quality Check for {dataset_name}:")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    # Check for very short reviews
    short_reviews = (df['review_length'] < 10).sum()
    print(f"Very short reviews (<10 chars): {short_reviews}")
    
    # Check for empty aspect terms
    empty_aspects = df['Aspect Term'].isna().sum()
    print(f"Empty aspect terms: {empty_aspects}")
    
    # Check sentiment label consistency
    unique_sentiments = df['actual_aspect_based_sentiment'].unique()
    print(f"Unique sentiment labels: {unique_sentiments}")


def main():
    """Main execution function"""
    print("="*80)
    print("Aspect-Based Sentiment Analysis - Data Exploration")
    print("="*80)
    
    # Load datasets
    laptop_df, restaurant_df = load_datasets()
    
    # Explore Laptop dataset
    laptop_df = explore_dataset(laptop_df, "Laptop")
    check_data_quality(laptop_df, "Laptop")
    visualize_dataset(laptop_df, "Laptop")
    
    # Explore Restaurant dataset
    restaurant_df = explore_dataset(restaurant_df, "Restaurant")
    check_data_quality(restaurant_df, "Restaurant")
    visualize_dataset(restaurant_df, "Restaurant")
    
    # Compare datasets
    compare_datasets(laptop_df, restaurant_df)
    
    print("\n" + "="*80)
    print("Exploratory Data Analysis Complete!")
    print("="*80)
    print("\nKey Insights:")
    print("1. Both datasets contain aspect-level sentiment annotations")
    print("2. Check for class imbalance in sentiment distribution")
    print("3. Review lengths vary - may need padding/truncation")
    print("4. Multiple aspects can be extracted from single reviews")
    print("\nNext Step: Data Preprocessing")


if __name__ == "__main__":
    main()
