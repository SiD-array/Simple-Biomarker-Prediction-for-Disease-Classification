"""
Script 2: Feature Selection
===========================
This script handles:
1. Loading cleaned data and removing constant genes
2. PCA analysis for dimensionality assessment
3. ANOVA-based statistical feature selection
4. Creating a reduced biomarker panel (K=1000 genes)

Goal: Reduce feature space from 20,531 genes to ~1000 informative biomarkers.

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

from src.feature_tools import (
    drop_constant_genes,
    perform_pca_analysis,
    plot_scree_plot,
    select_top_genes_anova,
    plot_top_genes_scores,
    save_top_biomarkers
)


def main():
    """Main function for feature selection pipeline."""
    
    print("\n" + "="*70)
    print("  RNA-Seq BIOMARKER PROJECT - Step 2: Feature Selection")
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
    # STEP 1: Load Cleaned Data
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 1: Loading Cleaned Data")
    print("-"*70)
    
    input_file = processed_path / "cleaned_scaled_data.pkl"
    
    if not input_file.exists():
        print(f"  ✗ ERROR: {input_file} not found!")
        print("  Please run scripts/1_clean_and_eda.py first.")
        return None
    
    processed_data = pd.read_pickle(input_file)
    data = processed_data['data']
    labels = processed_data['labels']
    
    print(f"  ✓ Loaded data: {data.shape[0]} samples × {data.shape[1]} genes")
    print(f"  ✓ Classes: {labels.value_counts().to_dict()}")
    
    # =========================================================================
    # STEP 2: Remove Constant Genes
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 2: Removing Constant Genes (Zero Variance)")
    print("-"*70)
    
    data_filtered, removed_genes = drop_constant_genes(data)
    
    # =========================================================================
    # STEP 3: PCA Analysis
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 3: PCA Analysis (Dimensionality Assessment)")
    print("-"*70)
    
    # Perform PCA (compute up to n_samples components)
    pca_results = perform_pca_analysis(data_filtered)
    
    # Generate and save scree plot
    plot_scree_plot(
        pca_results, 
        save_path=str(figures_path / "pca_plot.png"),
        n_components_show=50
    )
    
    # Print PCA summary
    print(f"\n  📋 PCA Summary:")
    print(f"     • Total variance explained by first 10 PCs: {pca_results['cumulative_variance'][9]*100:.2f}%")
    print(f"     • Total variance explained by first 50 PCs: {pca_results['cumulative_variance'][49]*100:.2f}%")
    print(f"     • Components needed for 95% variance: {pca_results['n_components_95']}")
    print(f"     • Components needed for 99% variance: {pca_results['n_components_99']}")
    
    # =========================================================================
    # STEP 4: ANOVA Feature Selection
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 4: ANOVA Feature Selection (Top 1000 Genes)")
    print("-"*70)
    
    # Select top 1000 genes using ANOVA F-test
    K = 1000
    data_selected, gene_rankings = select_top_genes_anova(data_filtered, labels, k=K)
    
    # Plot top 50 genes by F-score
    plot_top_genes_scores(
        gene_rankings,
        save_path=str(figures_path / "top_genes_fscore.png"),
        n_top=50
    )
    
    # =========================================================================
    # STEP 5: Save Outputs
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 5: Saving Outputs")
    print("-"*70)
    
    # Save reduced dataset
    output_data = {
        'data': data_selected,
        'labels': labels,
        'gene_rankings': gene_rankings,
        'pca_results': {
            'n_components_95': pca_results['n_components_95'],
            'n_components_99': pca_results['n_components_99'],
            'cumulative_variance': pca_results['cumulative_variance'],
            'explained_variance_ratio': pca_results['explained_variance_ratio']
        },
        'removed_constant_genes': removed_genes,
        'selection_method': 'ANOVA F-test (SelectKBest)',
        'k_selected': K,
        'processing_date': datetime.now().isoformat()
    }
    
    output_file = processed_path / "final_biomarker_set.pkl"
    pd.to_pickle(output_data, output_file)
    print(f"  ✓ Saved biomarker dataset to {output_file}")
    print(f"    Shape: {data_selected.shape[0]} samples × {data_selected.shape[1]} genes")
    
    # Save top 50 biomarker names to text file
    biomarkers_file = reports_path / "top_50_biomarkers.txt"
    save_top_biomarkers(gene_rankings, str(biomarkers_file), n_top=50)
    
    # =========================================================================
    # Summary Report
    # =========================================================================
    print("\n" + "="*70)
    print("  FEATURE SELECTION SUMMARY")
    print("="*70)
    print(f"""
  Feature Reduction Pipeline:
    • Original features:     {data.shape[1]:,} genes
    • After removing constant: {data_filtered.shape[1]:,} genes (-{len(removed_genes)})
    • Final biomarker panel: {data_selected.shape[1]:,} genes (Top K={K})
    
  PCA Analysis Results:
    • Components for 95% variance: {pca_results['n_components_95']}
    • Components for 99% variance: {pca_results['n_components_99']}
    
  Top 5 Biomarker Candidates:
""")
    for i, row in gene_rankings.head(5).iterrows():
        print(f"    {int(row['rank'])}. {row['gene_name']} (F={row['f_score']:.2f}, p={row['p_value']:.2e})")
    
    print(f"""
  Output Files:
    • Biomarker dataset: {output_file}
    • Top 50 biomarkers: {biomarkers_file}
    • PCA scree plot: {figures_path / 'pca_plot.png'}
    • Top genes plot: {figures_path / 'top_genes_fscore.png'}
    
  ⚠️  MITIGATION REMINDER for Step 3 (Model Training):
    • Class imbalance detected: BRCA (37.5%) vs COAD (9.7%)
    • Must use: class_weight='balanced' in classifiers
    • Must use: StratifiedKFold for cross-validation
    • Must use: macro-averaged F1-score for evaluation
    """)
    print("="*70)
    print("  Step 2 Complete! Proceed to Step 3: Model Training")
    print("="*70 + "\n")
    
    return data_selected, labels, gene_rankings


if __name__ == "__main__":
    data_selected, labels, gene_rankings = main()

