# run_comparison.py - ONE FILE FOR ALL MODEL COMPARISON NEEDS
import os
import sys
import pandas as pd
from time import time
import argparse
from typing import List, Optional

# Import enhanced metrics tools
from evaluation.metrics_logger import log_metrics, get_latest_comparison, clear_metrics
from evaluation.visualize_metrics import create_comprehensive_comparison, plot_model_evolution

# Import all model modules
from models.tfidf_cnn_model import run_tfidf_cnn
from models.bert_finetune import run_bert_finetune
from models.llm_zero_shot import run_zero_shot_prompting
from models.llm_few_shot import run_few_shot_prompting
from models.llm_chain_of_thought import run_chain_of_thought_prompting
from models.llm_self_consistency import run_self_consistency_prompting

def check_existing_results():
    """Check if we have existing results to visualize."""
    try:
        results = get_latest_comparison()
        if not results.empty:
            print(f"Found existing results for {len(results)} models:")
            for _, row in results.iterrows():
                time_str = f"{row['training_time_minutes']:.1f}m" if pd.notna(row.get('training_time_minutes')) else 'N/A'
                print(f"{row['model']:20} → Acc: {row['accuracy']:.3f}, F1: {row['f1_score']:.3f}, Time: {time_str}")
            return True
        else:
            print("No existing results found.")
            return False
    except:
        print("No existing results found.")
        return False

def run_models(sample_limit: int = 15, models_to_run: Optional[List[str]] = None):
    """
    Run the specified models (or all models if none specified).
    
    Args:
        sample_limit: Number of samples for testing
        models_to_run: List of model names to run, or None for all
    """
    
    print("=" * 80)
    print("RUNNING SENTIMENT ANALYSIS MODELS")
    print("=" * 80)
    
    # Data paths
    base_dir = os.path.join(os.path.dirname(__file__), "data")
    train_csv = os.path.join(base_dir, "train.csv")
    test_csv = os.path.join(base_dir, "test.csv")
    
    available_models = {
        'tfidf': ('TF-IDF + CNN', lambda: run_tfidf_cnn(train_csv, test_csv)),
        'bert': ('BERT Fine-tuning', lambda: run_bert_finetune(train_csv, test_csv, sample_size=sample_limit)),
        'zero_shot': ('LLM Zero-shot', lambda: run_zero_shot_prompting(test_csv, sample_limit=sample_limit)),
        'few_shot': ('LLM Few-shot', lambda: run_few_shot_prompting(test_csv, sample_limit=sample_limit)),
        'chain_of_thought': ('LLM Chain-of-Thought', lambda: run_chain_of_thought_prompting(test_csv, sample_limit=sample_limit)),
        'self_consistency': ('LLM Self-Consistency', lambda: run_self_consistency_prompting(test_csv, sample_limit=sample_limit))
    }
    
    # If no specific models requested, run all
    if models_to_run is None:
        models_to_run = list(available_models.keys())
    
    print(f"Testing with {sample_limit} samples per model\n")
    
    for i, model_key in enumerate(models_to_run, 1):
        if model_key in available_models:
            model_name, model_func = available_models[model_key]
            print(f"{i}️⃣ Running {model_name}...")
            try:
                start_time = time()
                model_func()
                elapsed = time() - start_time
                print(f"Completed in {elapsed:.1f} seconds\n")
            except Exception as e:
                print(f"Failed: {e}\n")
        else:
            print(f" Unknown model: {model_key}")

def show_analysis():
    """Show comprehensive analysis of all available results."""
    
    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # Create visualizations
    create_comprehensive_comparison()
    
    # Get latest results for analysis
    results = get_latest_comparison()
    
    if results.empty:
        print(" No results available for analysis.")
        return
    
    # Performance vs Efficiency vs Interpretability Analysis
    print("\n" + "=" * 60)
    print("PERFORMANCE vs EFFICIENCY vs INTERPRETABILITY")
    print("=" * 60)
    
    # Performance ranking
    print("\nPERFORMANCE RANKING (by Accuracy):")
    performance_sorted = results.sort_values('accuracy', ascending=False)
    for i, (_, row) in enumerate(performance_sorted.iterrows(), 1):
        print(f"  {i}. {row['model']:25} → {row['accuracy']:.4f}")
    
    # Speed ranking (if available)
    if 'training_time_minutes' in results.columns and results['training_time_minutes'].notna().any():
        print("\nSPEED RANKING (Fastest to Slowest):")
        speed_sorted = results.dropna(subset=['training_time_minutes']).sort_values('training_time_minutes')
        for i, (_, row) in enumerate(speed_sorted.iterrows(), 1):
            print(f"  {i}. {row['model']:25} → {row['training_time_minutes']:.1f} minutes")
    
    # Model type analysis
    if 'model_type' in results.columns and results['model_type'].notna().any():
        print(f"\nBY MODEL TYPE:")
        for model_type in ['Classical ML', 'Transformer', 'LLM']:
            type_models = results[results['model_type'] == model_type]
            if not type_models.empty:
                print(f"\n{model_type.upper()}:")
                avg_acc = type_models['accuracy'].mean()
                avg_f1 = type_models['f1_score'].mean()
                print(f"  Average Performance → Acc: {avg_acc:.3f}, F1: {avg_f1:.3f}")
                for _, row in type_models.iterrows():
                    time_str = f"{row['training_time_minutes']:.1f}m" if pd.notna(row.get('training_time_minutes')) else 'N/A'
                    print(f"    {row['model']:20} → Acc: {row['accuracy']:.3f}, F1: {row['f1_score']:.3f}, Time: {time_str}")
    
    # Best performers
    best_acc = results.loc[results['accuracy'].idxmax()]
    best_f1 = results.loc[results['f1_score'].idxmax()]
    
    print(f"\nCHAMPIONS:")
    print(f"Highest Accuracy: {best_acc['model']} ({best_acc['accuracy']:.4f})")
    print(f"Highest F1-Score: {best_f1['model']} ({best_f1['f1_score']:.4f})")
    
    if 'training_time_minutes' in results.columns and results['training_time_minutes'].notna().any():
        fastest = results.dropna(subset=['training_time_minutes']).loc[results['training_time_minutes'].idxmin()]
        print(f"Fastest: {fastest['model']} ({fastest['training_time_minutes']:.2f} minutes)")
    
    # Project insights
    print(f"\nPROJECT INSIGHTS:")
    print(f" Models Compared: {len(results)}")
    print(f" Results saved in: data/metrics_results.csv")
    print(f" Visualizations: data/comprehensive_comparison.png")
    
    # Recommendations based on results
    print(f"\nRECOMMENDATIONS:")
    best_acc_value = best_acc['accuracy']
    if isinstance(best_acc_value, pd.Series):
        best_acc_value = best_acc_value.iloc[0]
    
    if best_acc_value > 0.7:
        print(f"Strong performer found: {best_acc['model']}")
    else:
        print(f"All models showing moderate performance - consider tuning")
    
    if 'training_time_minutes' in results.columns and results['training_time_minutes'].notna().any():
        time_range = results['training_time_minutes'].max() - results['training_time_minutes'].min()
        if time_range > 10:
            print(f"Significant speed differences - consider use case requirements")
    
    print("=" * 80)

def main():
    """Main function with command line options."""
    
    parser = argparse.ArgumentParser(description='Comprehensive Sentiment Analysis Model Comparison')
    parser.add_argument('--run', nargs='*', 
                       help='Models to run: tfidf, bert, zero_shot, few_shot, chain_of_thought, self_consistency')
    parser.add_argument('--sample-limit', type=int, default=15,
                       help='Number of samples for testing (default: 15)')
    parser.add_argument('--clear', action='store_true',
                       help='Clear previous results before running')
    parser.add_argument('--show-only', action='store_true',
                       help='Only show analysis of existing results (don\'t run models)')
    
    args = parser.parse_args()
    
    print("SENTIMENT ANALYSIS MODEL COMPARISON TOOL")
    print("Project: Classical DL vs LLM Prompting Performance")
    print("=" * 60)
    
    # Clear previous results if requested
    if args.clear:
        clear_metrics()
        print("Cleared previous results\n")
    
    # Check existing results
    has_existing = check_existing_results()
    print()
    
    # Show only mode
    if args.show_only:
        if has_existing:
            show_analysis()
        else:
            print("No existing results to show. Run some models first.")
        return
    
    # Run models if requested
    if args.run is not None:
        if len(args.run) == 0:
            # --run with no arguments means run all
            run_models(sample_limit=args.sample_limit)
        else:
            # Run specific models
            run_models(sample_limit=args.sample_limit, models_to_run=args.run)
    elif not has_existing:
        # No existing results and no --run specified, ask user
        print("No existing results found.")
        response = input("Do you want to run all models? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            run_models(sample_limit=args.sample_limit)
        else:
            print("Use --run to specify which models to run, or --show-only to view existing results.")
            return
    
    # Always show analysis at the end
    show_analysis()

if __name__ == "__main__":
    # If no command line args, run in interactive mode
    if len(sys.argv) == 1:
        print("SENTIMENT ANALYSIS MODEL COMPARISON")
        print("Project: Classical DL vs LLM Prompting Performance")
        print("=" * 60)
        
        # Check existing results
        has_existing = check_existing_results()
        print()
        
        if has_existing:
            print("Options:")
            print("1. Show analysis of existing results")
            print("2. Run additional models")
            print("3. Clear results and run all models")
            
            choice = input("Choose option (1-3): ").strip()
            
            if choice == "1":
                show_analysis()
            elif choice == "2":
                print("\nAvailable models: tfidf, bert, zero_shot, few_shot, chain_of_thought, self_consistency")
                models_input = input("Enter models to run (space-separated) or 'all': ").strip()
                if models_input == 'all':
                    models_to_run = None
                else:
                    models_to_run = models_input.split()
                run_models(models_to_run=models_to_run)
                show_analysis()
            elif choice == "3":
                clear_metrics()
                run_models()
                show_analysis()
            else:
                print("Invalid choice. Showing existing results.")
                show_analysis()
        else:
            response = input("No existing results. Run all models? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                run_models()
                show_analysis()
            else:
                print("Exiting. Use --help for command line options.")
    else:
        main()