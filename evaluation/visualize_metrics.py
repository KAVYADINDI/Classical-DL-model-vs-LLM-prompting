# evaluation/visualize_metrics.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

METRICS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "metrics_results.csv")

def load_metrics(file_path=METRICS_FILE):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No metrics file found at {file_path}")
    return pd.read_csv(file_path)

def plot_latest_scores(file_path=METRICS_FILE, save_plot=None):
    """Original function - maintained for compatibility."""
    df = load_metrics(file_path)
    # select latest entry per model
    latest = df.sort_values("timestamp").groupby("model").tail(1)
    latest = latest.set_index("model")
    latest = latest[["accuracy", "f1_score"]]

    ax = latest.plot(kind="bar", figsize=(8,5))
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Latest Metrics by Model")
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save_plot:
        plt.savefig(save_plot)
        print("Saved plot to", save_plot)
    plt.show()

def create_comprehensive_comparison(file_path=METRICS_FILE, save_plots=True):
    """
    Create comprehensive comparison visualizations for all models.
    
    Args:
        file_path: Path to metrics CSV file
        save_plots: Whether to save plots to files
    """
    try:
        df = load_metrics(file_path)
    except FileNotFoundError:
        print("❌ No metrics file found. Please run some models first.")
        return
    
    # Get latest metrics per model
    latest = df.sort_values("timestamp").groupby("model").tail(1).reset_index(drop=True)
    
    if len(latest) == 0:
        print("❌ No metrics found in the file.")
        return
    
    print(f"📊 Creating comprehensive comparison for {len(latest)} models...")
    
    # Set up the plot style
    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        # Fallback if seaborn style not available
        plt.style.use('default')
        plt.rcParams.update({'figure.figsize': (16, 12)})
    fig = plt.figure(figsize=(16, 12))
    
    # Create color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FFB6C1', '#98FB98']
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(latest['model'])}
    
    # 1. Accuracy Comparison (Top Left)
    ax1 = plt.subplot(2, 3, 1)
    bars1 = ax1.bar(latest['model'], latest['accuracy'], 
                    color=[model_colors[m] for m in latest['model']])
    ax1.set_title('🎯 Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. F1-Score Comparison (Top Center)
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(latest['model'], latest['f1_score'], 
                    color=[model_colors[m] for m in latest['model']])
    ax2.set_title('📈 F1-Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('F1-Score')
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Training/Inference Time (Top Right) - if available
    ax3 = plt.subplot(2, 3, 3)
    if 'training_time_minutes' in latest.columns and latest['training_time_minutes'].notna().any():
        time_data = latest.dropna(subset=['training_time_minutes'])
        bars3 = ax3.bar(time_data['model'], time_data['training_time_minutes'], 
                        color=[model_colors[m] for m in time_data['model']])
        ax3.set_title('⏱️ Training/Inference Time', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Time (minutes)')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}m', ha='center', va='bottom', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'No timing data\navailable', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('⏱️ Training/Inference Time', fontsize=14, fontweight='bold')
    
    # 4. Performance vs Time Scatter (Bottom Left)
    ax4 = plt.subplot(2, 3, 4)
    type_colors = {'Classical ML': '#FF6B6B', 'Transformer': '#4ECDC4', 'LLM': '#45B7D1'}  # Define here for use in pie chart too
    
    if 'training_time_minutes' in latest.columns and latest['training_time_minutes'].notna().any():
        # Group by model type if available
        if 'model_type' in latest.columns and latest['model_type'].notna().any():
            for model_type in type_colors:
                mask = latest['model_type'] == model_type
                if mask.any():
                    subset = latest[mask].dropna(subset=['training_time_minutes'])
                    ax4.scatter(subset['training_time_minutes'], subset['accuracy'], 
                               c=type_colors[model_type], label=model_type, s=100, alpha=0.7)
            ax4.legend()
        else:
            time_data = latest.dropna(subset=['training_time_minutes'])
            ax4.scatter(time_data['training_time_minutes'], time_data['accuracy'], 
                       c=[model_colors[m] for m in time_data['model']], s=100, alpha=0.7)
        
        ax4.set_xlabel('Time (minutes)')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('🎯 Accuracy vs Time', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add model labels
        time_data = latest.dropna(subset=['training_time_minutes'])
        for _, row in time_data.iterrows():
            ax4.annotate(row['model'], (row['training_time_minutes'], row['accuracy']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'No timing data\navailable', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('🎯 Accuracy vs Time', fontsize=14, fontweight='bold')
    
    # 5. Model Type Distribution (Bottom Center) - if available
    ax5 = plt.subplot(2, 3, 5)
    if 'model_type' in latest.columns and latest['model_type'].notna().any():
        type_counts = latest['model_type'].value_counts()
        colors_pie = [type_colors.get(t, '#CCCCCC') for t in type_counts.index]
        ax5.pie(list(type_counts.values), labels=list(type_counts.index), autopct='%1.0f%%', 
                colors=colors_pie, startangle=90)
        ax5.set_title('🏷️ Model Type Distribution', fontsize=14, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'No model type\ndata available', ha='center', va='center', 
                transform=ax5.transAxes, fontsize=12)
        ax5.set_title('🏷️ Model Type Distribution', fontsize=14, fontweight='bold')
    
    # 6. Performance Summary Table (Bottom Right)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create summary table
    summary_data = []
    for _, row in latest.iterrows():
        summary_data.append([
            row['model'],
            f"{row['accuracy']:.3f}",
            f"{row['f1_score']:.3f}",
            f"{row['training_time_minutes']:.1f}m" if pd.notna(row.get('training_time_minutes')) else 'N/A'
        ])
    
    table = ax6.table(cellText=summary_data,
                     colLabels=['Model', 'Accuracy', 'F1-Score', 'Time'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax6.set_title('📋 Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save plots if requested
    if save_plots:
        plot_dir = os.path.dirname(file_path)
        plot_file = os.path.join(plot_dir, "comprehensive_comparison.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📈 Comprehensive comparison saved to: {plot_file}")
    
    plt.show()
    
    # Print best performers
    print("\n" + "="*60)
    print("🏆 BEST PERFORMERS")
    print("="*60)
    
    best_acc = latest.loc[latest['accuracy'].idxmax()]
    best_f1 = latest.loc[latest['f1_score'].idxmax()]
    
    print(f"🎯 Highest Accuracy: {best_acc['model']} ({best_acc['accuracy']:.4f})")
    print(f"📈 Highest F1-Score: {best_f1['model']} ({best_f1['f1_score']:.4f})")
    
    if 'training_time_minutes' in latest.columns and latest['training_time_minutes'].notna().any():
        fastest = latest.dropna(subset=['training_time_minutes']).loc[latest['training_time_minutes'].idxmin()]
        print(f"⚡ Fastest: {fastest['model']} ({fastest['training_time_minutes']:.2f} minutes)")
    
    print("="*60)

def plot_model_evolution(file_path=METRICS_FILE, save_plot=None):
    """
    Plot how model performance evolved over time.
    """
    try:
        df = load_metrics(file_path)
    except FileNotFoundError:
        print("❌ No metrics file found.")
        return
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy over time for each model
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.plot(model_data['timestamp'], model_data['accuracy'], 
                marker='o', label=f"{model} (Accuracy)", alpha=0.7)
    
    plt.title('📈 Model Performance Evolution Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        print(f"📈 Evolution plot saved to: {save_plot}")
    
    plt.show()

if __name__ == "__main__":
    print("🎨 Creating comprehensive model comparison visualizations...")
    create_comprehensive_comparison()
    print("\n🕒 Creating performance evolution plot...")
    plot_model_evolution()
