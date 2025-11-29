"""
Model Comparison and Visualization
Generate comprehensive comparison charts for the presentation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_metrics():
    """Load metrics from CSV"""
    df = pd.read_csv('compare/model_metrics.csv')
    return df

def plot_accuracy_comparison(df):
    """Plot accuracy comparison bar chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = df['model_name'].values
    accuracy = df['accuracy'].values * 100
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax.barh(models, accuracy, color=colors[:len(models)])
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracy)):
        ax.text(acc + 1, i, f'{acc:.1f}%', va='center', fontweight='bold')
    
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig('evaluation/comparison_accuracy.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: evaluation/comparison_accuracy.png")
    plt.close()

def plot_f1_precision_recall(df):
    """Plot F1, Precision, Recall comparison"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = df['model_name'].values
    x = np.arange(len(models))
    width = 0.25
    
    f1 = df['f1_score'].values * 100
    precision = df['precision_macro'].values * 100
    recall = df['recall_macro'].values * 100
    
    bars1 = ax.bar(x - width, f1, width, label='F1-Score', color='#3498db')
    bars2 = ax.bar(x, precision, width, label='Precision', color='#2ecc71')
    bars3 = ax.bar(x + width, recall, width, label='Recall', color='#e74c3c')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('F1-Score, Precision, and Recall Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation/comparison_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: evaluation/comparison_metrics.png")
    plt.close()

def plot_inference_time(df):
    """Plot inference time comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = df['model_name'].values
    inference_time = df['inference_time_per_sample'].values * 1000  # Convert to ms
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax.barh(models, inference_time, color=colors[:len(models)])
    
    # Add value labels
    for i, (bar, time) in enumerate(zip(bars, inference_time)):
        if time < 1000:
            label = f'{time:.1f} ms'
        else:
            label = f'{time/1000:.2f} s'
        ax.text(time + max(inference_time)*0.02, i, label, va='center', fontweight='bold')
    
    ax.set_xlabel('Inference Time per Sample (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Model Inference Speed Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('evaluation/comparison_speed.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: evaluation/comparison_speed.png")
    plt.close()

def plot_accuracy_vs_speed(df):
    """Scatter plot: Accuracy vs Inference Speed"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    models = df['model_name'].values
    accuracy = df['accuracy'].values * 100
    inference_time = df['inference_time_per_sample'].values * 1000  # ms
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    for i, (model, acc, time) in enumerate(zip(models, accuracy, inference_time)):
        ax.scatter(time, acc, s=300, c=colors[i], alpha=0.6, edgecolors='black', linewidth=2)
        ax.annotate(model, (time, acc), xytext=(10, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Inference Time per Sample (ms, log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Inference Speed Trade-off', fontsize=14, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation/comparison_tradeoff.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: evaluation/comparison_tradeoff.png")
    plt.close()

def create_summary_table(df):
    """Create a formatted summary table"""
    summary = df[['model_name', 'accuracy', 'f1_score', 'precision_macro', 'recall_macro', 
                   'inference_time_per_sample', 'training_time']].copy()
    
    # Format percentages
    summary['accuracy'] = (summary['accuracy'] * 100).round(2).astype(str) + '%'
    summary['f1_score'] = (summary['f1_score'] * 100).round(2).astype(str) + '%'
    summary['precision_macro'] = (summary['precision_macro'] * 100).round(2).astype(str) + '%'
    summary['recall_macro'] = (summary['recall_macro'] * 100).round(2).astype(str) + '%'
    
    # Format times
    summary['inference_time_per_sample'] = summary['inference_time_per_sample'].apply(
        lambda x: f'{x*1000:.2f} ms' if x < 1 else f'{x:.2f} s'
    )
    summary['training_time'] = summary['training_time'].apply(
        lambda x: f'{x:.2f} s' if x < 60 else f'{x/60:.2f} min'
    )
    
    # Rename columns
    summary.columns = ['Model', 'Accuracy', 'F1-Score', 'Precision', 'Recall', 
                       'Inference Time', 'Training Time']
    
    # Save to CSV
    summary.to_csv('evaluation/model_comparison_summary.csv', index=False)
    print("✓ Saved: evaluation/model_comparison_summary.csv")
    
    return summary

def main():
    """Generate all comparison visualizations"""
    print("="*80)
    print("Generating Model Comparison Visualizations")
    print("="*80)
    
    # Load metrics
    df = load_metrics()
    print(f"\nLoaded metrics for {len(df)} models")
    print(df[['model_name', 'accuracy', 'f1_score']].to_string(index=False))
    
    # Create visualizations
    print("\n" + "-"*80)
    print("Creating visualizations...")
    print("-"*80 + "\n")
    
    plot_accuracy_comparison(df)
    plot_f1_precision_recall(df)
    plot_inference_time(df)
    plot_accuracy_vs_speed(df)
    
    # Create summary table
    print("\n" + "-"*80)
    print("Creating summary table...")
    print("-"*80 + "\n")
    
    summary = create_summary_table(df)
    print("\nSummary Table:")
    print(summary.to_string(index=False))
    
    print("\n" + "="*80)
    print("Comparison Visualizations Complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  - evaluation/comparison_accuracy.png")
    print("  - evaluation/comparison_metrics.png")
    print("  - evaluation/comparison_speed.png")
    print("  - evaluation/comparison_tradeoff.png")
    print("  - evaluation/model_comparison_summary.csv")

if __name__ == "__main__":
    main()
