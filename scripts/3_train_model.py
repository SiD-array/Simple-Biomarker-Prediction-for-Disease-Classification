"""
Script 3: Model Training and Evaluation
=======================================
This script handles:
1. Loading the biomarker panel data
2. Training multiple classifiers with imbalance mitigation
3. Cross-validation evaluation with stratified sampling
4. Comprehensive model comparison and reporting

Goal: Train robust multi-class classifiers on the 1000-gene biomarker panel.

Author: RNA-Seq Biomarker Project
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
import joblib

from src.model_trainer import (
    ModelTrainer,
    plot_confusion_matrix,
    plot_model_comparison,
    save_results_summary
)


def main():
    """Main function for model training pipeline."""
    
    print("\n" + "="*70)
    print("  RNA-Seq BIOMARKER PROJECT - Step 3: Model Training & Evaluation")
    print("="*70)
    print(f"  Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # =========================================================================
    # Define Paths
    # =========================================================================
    project_root = Path(__file__).parent.parent
    processed_path = project_root / "data" / "processed"
    figures_path = project_root / "reports" / "figures"
    reports_path = project_root / "reports"
    
    # Create directories if needed
    figures_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 1: Loading Biomarker Panel Data")
    print("-"*70)
    
    input_file = processed_path / "final_biomarker_set.pkl"
    
    if not input_file.exists():
        print(f"  ✗ ERROR: {input_file} not found!")
        print("  Please run scripts/2_feature_selection.py first.")
        return None
    
    biomarker_data = pd.read_pickle(input_file)
    X = biomarker_data['data']
    y = biomarker_data['labels']
    
    print(f"  ✓ Loaded biomarker panel: {X.shape[0]} samples × {X.shape[1]} genes")
    print(f"  ✓ Class distribution:")
    for cls, count in y.value_counts().sort_index().items():
        pct = (count / len(y)) * 100
        print(f"     • {cls}: {count} samples ({pct:.1f}%)")
    
    # =========================================================================
    # STEP 2: Setup Model Trainer
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 2: Initializing Model Trainer")
    print("-"*70)
    print("\n  ⚠️  IMBALANCE MITIGATION STRATEGIES:")
    print("     • Using class_weight='balanced' for all classifiers")
    print("     • Using StratifiedKFold to maintain class proportions")
    print("     • Using F1-Macro as primary metric (penalizes poor minority class performance)")
    
    trainer = ModelTrainer(X, y, n_splits=5, random_state=42)
    
    # =========================================================================
    # STEP 3: Evaluate All Models
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 3: Cross-Validation Evaluation")
    print("-"*70)
    
    summary_df = trainer.evaluate_all_models()
    
    # Print summary table
    print("\n" + "="*60)
    print("  MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)
    
    # =========================================================================
    # STEP 4: Best Model Analysis
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 4: Best Model Analysis")
    print("-"*70)
    
    best_model_name, best_model = trainer.get_best_model()
    print(f"\n  🏆 Best Model: {best_model_name}")
    print(f"     • F1-Macro: {trainer.results[best_model_name]['f1_macro_mean']:.4f}")
    print(f"     • Accuracy: {trainer.results[best_model_name]['accuracy_mean']:.4f}")
    
    # Get detailed classification report
    print("\n  📋 Classification Report (Cross-Validated Predictions):")
    print("-"*60)
    report = trainer.get_classification_report(best_model_name)
    print(report)
    
    # Get detailed metrics
    metrics = trainer.get_detailed_metrics(best_model_name)
    
    # =========================================================================
    # STEP 5: Train Final Model
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 5: Training Final Model on Full Dataset")
    print("-"*70)
    
    final_model = trainer.train_final_model(best_model_name)
    print(f"  ✓ Final model trained on all {X.shape[0]} samples")
    
    # =========================================================================
    # STEP 6: Generate Visualizations
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 6: Generating Visualizations")
    print("-"*70)
    
    # Plot confusion matrix
    y_pred_best = trainer.cv_predictions[best_model_name]
    plot_confusion_matrix(
        trainer.y, y_pred_best,
        trainer.classes,
        save_path=str(figures_path / "confusion_matrix.png"),
        model_name=best_model_name
    )
    
    # Plot model comparison
    plot_model_comparison(
        summary_df,
        save_path=str(figures_path / "model_comparison.png")
    )
    
    # =========================================================================
    # STEP 7: Save Results
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 7: Saving Results")
    print("-"*70)
    
    # Save results summary
    save_results_summary(
        summary_df, report, metrics,
        save_path=str(reports_path / "model_results_summary.txt"),
        best_model_name=best_model_name
    )
    
    # Save summary table as CSV
    summary_df.to_csv(reports_path / "model_comparison.csv", index=False)
    print(f"   ✓ Saved model comparison to {reports_path / 'model_comparison.csv'}")
    
    # Save trained model
    model_file = processed_path / "trained_model.pkl"
    joblib.dump({
        'model': final_model,
        'model_name': best_model_name,
        'label_encoder': trainer.label_encoder,
        'feature_names': X.columns.tolist(),
        'classes': trainer.classes.tolist(),
        'cv_results': trainer.results[best_model_name],
        'training_date': datetime.now().isoformat()
    }, model_file)
    print(f"   ✓ Saved trained model to {model_file}")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "="*70)
    print("  MODEL TRAINING SUMMARY")
    print("="*70)
    print(f"""
  ✅ Pipeline Complete!
  
  Best Model: {best_model_name}
  ─────────────────────────────────────────
  • F1-Macro Score: {trainer.results[best_model_name]['f1_macro_mean']:.4f} (±{trainer.results[best_model_name]['f1_macro_std']:.4f})
  • Accuracy: {trainer.results[best_model_name]['accuracy_mean']:.4f} (±{trainer.results[best_model_name]['accuracy_std']:.4f})
  
  Per-Class Performance (F1-Score):
""")
    for i, cls in enumerate(trainer.classes):
        f1 = metrics['f1_per_class'][i]
        print(f"    • {cls}: {f1:.4f}")
    
    print(f"""
  Output Files:
  ─────────────────────────────────────────
  • Model Results: {reports_path / 'model_results_summary.txt'}
  • Model Comparison CSV: {reports_path / 'model_comparison.csv'}
  • Confusion Matrix: {figures_path / 'confusion_matrix.png'}
  • Model Comparison Plot: {figures_path / 'model_comparison.png'}
  • Trained Model: {model_file}
    """)
    print("="*70)
    print("  🎉 RNA-Seq Biomarker Classification Pipeline Complete!")
    print("="*70 + "\n")
    
    return trainer, final_model, summary_df


if __name__ == "__main__":
    trainer, model, summary = main()

