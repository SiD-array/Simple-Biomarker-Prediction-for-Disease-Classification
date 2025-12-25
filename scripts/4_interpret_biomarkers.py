"""
Script 4: Biomarker Interpretation
==================================
This script handles:
1. Loading the trained Logistic Regression model
2. Extracting and analyzing model coefficients
3. Ranking genes by predictive power
4. Identifying class-specific biomarkers
5. Generating interpretive visualizations

Goal: Leverage model interpretability to identify the most potent biomarkers.

Author: RNA-Seq Biomarker Project
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime


def load_assets(processed_path: Path):
    """
    Load trained model and data assets.
    
    Returns
    -------
    Tuple containing model, data, labels, and metadata
    """
    print("\n📂 Loading Assets...")
    
    # Load trained model
    model_data = joblib.load(processed_path / "trained_model.pkl")
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    feature_names = model_data['feature_names']
    classes = model_data['classes']
    
    print(f"   ✓ Loaded trained model: {model_data['model_name']}")
    print(f"   ✓ Classes: {classes}")
    
    # Load biomarker data
    biomarker_data = pd.read_pickle(processed_path / "final_biomarker_set.pkl")
    X = biomarker_data['data']
    y = biomarker_data['labels']
    
    print(f"   ✓ Loaded expression data: {X.shape[0]} samples × {X.shape[1]} genes")
    
    return model, X, y, feature_names, classes, label_encoder


def extract_coefficients(model, feature_names: list, classes: list) -> pd.DataFrame:
    """
    Extract coefficient matrix from Logistic Regression model.
    
    Parameters
    ----------
    model : LogisticRegression
        Trained model
    feature_names : list
        List of gene names
    classes : list
        List of class names
        
    Returns
    -------
    pd.DataFrame
        Coefficient matrix (genes × classes)
    """
    print("\n🔬 Extracting Model Coefficients...")
    
    # Get coefficient matrix (shape: n_classes × n_features)
    coef_matrix = model.coef_
    
    print(f"   • Coefficient matrix shape: {coef_matrix.shape}")
    print(f"   • Classes: {len(classes)}, Features: {len(feature_names)}")
    
    # Create DataFrame with genes as rows and classes as columns
    coef_df = pd.DataFrame(
        coef_matrix.T,
        index=feature_names,
        columns=classes
    )
    coef_df.index.name = 'gene_name'
    
    return coef_df


def rank_genes_by_influence(coef_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank genes by their maximum absolute coefficient across all classes.
    
    Parameters
    ----------
    coef_df : pd.DataFrame
        Coefficient matrix
        
    Returns
    -------
    pd.DataFrame
        Ranked genes with coefficients and influence metrics
    """
    print("\n📊 Ranking Genes by Predictive Power...")
    
    # Calculate max absolute coefficient for each gene
    coef_df['max_abs_coef'] = coef_df.abs().max(axis=1)
    
    # Find which class has the maximum coefficient for each gene
    coef_df['dominant_class'] = coef_df.drop(columns=['max_abs_coef']).abs().idxmax(axis=1)
    
    # Get the actual coefficient value (with sign) for the dominant class
    coef_df['dominant_coef'] = coef_df.apply(
        lambda row: row[row['dominant_class']], axis=1
    )
    
    # Sort by max absolute coefficient
    ranked_df = coef_df.sort_values('max_abs_coef', ascending=False)
    ranked_df['rank'] = range(1, len(ranked_df) + 1)
    
    # Reorder columns
    cols = ['rank', 'max_abs_coef', 'dominant_class', 'dominant_coef'] + list(coef_df.columns[:-4])
    ranked_df = ranked_df[cols]
    
    print(f"   ✓ Ranked {len(ranked_df)} genes by influence")
    print(f"   • Top gene: {ranked_df.index[0]} (max |coef| = {ranked_df.iloc[0]['max_abs_coef']:.4f})")
    
    return ranked_df


def save_top_influential_biomarkers(ranked_df: pd.DataFrame, save_path: str, n_top: int = 50):
    """
    Save top influential biomarkers to text file.
    
    Parameters
    ----------
    ranked_df : pd.DataFrame
        Ranked genes DataFrame
    save_path : str
        Path to save the file
    n_top : int
        Number of top genes to save
    """
    top_genes = ranked_df.head(n_top)
    classes = [col for col in ranked_df.columns if col not in ['rank', 'max_abs_coef', 'dominant_class', 'dominant_coef']]
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"  Top {n_top} Most Influential Biomarkers (Logistic Regression Coefficients)\n")
        f.write("="*80 + "\n")
        f.write(f"Selection Method: Maximum Absolute Coefficient across 5 Cancer Classes\n")
        f.write(f"Model: Logistic Regression with class_weight='balanced'\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Rank':<6}{'Gene':<15}{'Max |Coef|':<12}{'Dominant Class':<15}{'Coefficient':<12}\n")
        f.write("-"*60 + "\n")
        
        for idx, row in top_genes.iterrows():
            sign = "+" if row['dominant_coef'] > 0 else ""
            f.write(f"{int(row['rank']):<6}{idx:<15}{row['max_abs_coef']:<12.4f}"
                    f"{row['dominant_class']:<15}{sign}{row['dominant_coef']:<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION GUIDE:\n")
        f.write("-"*80 + "\n")
        f.write("• Positive coefficient: Higher expression → Higher probability of that class\n")
        f.write("• Negative coefficient: Higher expression → Lower probability of that class\n")
        f.write("• Max |Coef|: Overall importance regardless of direction\n")
        f.write("• Dominant Class: The cancer type most strongly associated with this gene\n")
        f.write("="*80 + "\n")
    
    print(f"   ✓ Saved top {n_top} influential biomarkers to {save_path}")


def plot_gene_expression_boxplot(X: pd.DataFrame, y: pd.Series, gene_name: str, 
                                  rank: int, save_path: str):
    """
    Create boxplot showing gene expression across cancer classes.
    
    Parameters
    ----------
    X : pd.DataFrame
        Expression data
    y : pd.Series
        Class labels
    gene_name : str
        Name of gene to plot
    rank : int
        Rank of the gene
    save_path : str
        Path to save the figure
    """
    # Prepare data
    plot_data = pd.DataFrame({
        'Expression': X[gene_name],
        'Cancer Type': y
    })
    
    # Color palette
    colors = {
        'BRCA': '#E74C3C',
        'COAD': '#F39C12', 
        'KIRC': '#9B59B6',
        'LUAD': '#2ECC71',
        'PRAD': '#3498DB'
    }
    
    plt.figure(figsize=(10, 6))
    
    # Create boxplot
    ax = sns.boxplot(
        x='Cancer Type', 
        y='Expression', 
        data=plot_data,
        order=['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD'],
        palette=colors,
        width=0.6
    )
    
    # Add individual points
    sns.stripplot(
        x='Cancer Type',
        y='Expression',
        data=plot_data,
        order=['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD'],
        color='black',
        alpha=0.3,
        size=3,
        jitter=True
    )
    
    plt.title(f'Top {rank} Biomarker: {gene_name}\nExpression Distribution Across Cancer Types', 
              fontsize=12, fontweight='bold')
    plt.xlabel('Cancer Type', fontsize=11)
    plt.ylabel('Gene Expression (log-transformed)', fontsize=11)
    
    # Add sample counts
    for i, cancer in enumerate(['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD']):
        n = (y == cancer).sum()
        ax.text(i, ax.get_ylim()[0] - 0.5, f'n={n}', ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_top_genes_heatmap(ranked_df: pd.DataFrame, save_path: str, n_top: int = 20):
    """
    Create heatmap of top gene coefficients across classes.
    
    Parameters
    ----------
    ranked_df : pd.DataFrame
        Ranked genes DataFrame
    save_path : str
        Path to save the figure
    n_top : int
        Number of top genes to show
    """
    print(f"\n📊 Generating Coefficient Heatmap (Top {n_top} genes)...")
    
    # Get top genes and coefficient columns only
    classes = [col for col in ranked_df.columns if col not in ['rank', 'max_abs_coef', 'dominant_class', 'dominant_coef']]
    top_genes = ranked_df.head(n_top)
    coef_data = top_genes[classes]
    
    plt.figure(figsize=(10, 12))
    
    # Create heatmap
    sns.heatmap(
        coef_data,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Coefficient Value'}
    )
    
    plt.title(f'Top {n_top} Biomarker Coefficients by Cancer Type\n(Logistic Regression)', 
              fontsize=12, fontweight='bold')
    plt.xlabel('Cancer Type', fontsize=11)
    plt.ylabel('Gene (Ranked by Influence)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved coefficient heatmap to {save_path}")


def print_top_genes_summary(ranked_df: pd.DataFrame, n_top: int = 5):
    """Print summary table of top genes."""
    classes = [col for col in ranked_df.columns if col not in ['rank', 'max_abs_coef', 'dominant_class', 'dominant_coef']]
    
    print("\n" + "="*90)
    print(f"  TOP {n_top} MOST INFLUENTIAL BIOMARKERS")
    print("="*90)
    print(f"\n{'Rank':<6}{'Gene':<15}{'Max |Coef|':<12}", end="")
    for cls in classes:
        print(f"{cls:<12}", end="")
    print()
    print("-"*90)
    
    for idx, row in ranked_df.head(n_top).iterrows():
        print(f"{int(row['rank']):<6}{idx:<15}{row['max_abs_coef']:<12.4f}", end="")
        for cls in classes:
            coef = row[cls]
            print(f"{coef:>+10.4f}  ", end="")
        print()
    
    print("-"*90)
    print("\nInterpretation:")
    print("  • Positive coefficient → Gene upregulation associated with that cancer type")
    print("  • Negative coefficient → Gene downregulation associated with that cancer type")
    print("="*90)


def main():
    """Main function for biomarker interpretation pipeline."""
    
    print("\n" + "="*70)
    print("  RNA-Seq BIOMARKER PROJECT - Step 4: Biomarker Interpretation")
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
    # STEP 1: Load Assets
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 1: Loading Assets")
    print("-"*70)
    
    model, X, y, feature_names, classes, label_encoder = load_assets(processed_path)
    
    # =========================================================================
    # STEP 2: Extract Coefficients
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 2: Extracting Model Coefficients")
    print("-"*70)
    
    coef_df = extract_coefficients(model, feature_names, classes)
    
    # =========================================================================
    # STEP 3: Rank Genes by Influence
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 3: Ranking Genes by Predictive Power")
    print("-"*70)
    
    ranked_df = rank_genes_by_influence(coef_df)
    
    # Print top 5 summary
    print_top_genes_summary(ranked_df, n_top=5)
    
    # =========================================================================
    # STEP 4: Save Top 50 Influential Biomarkers
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 4: Saving Top 50 Influential Biomarkers")
    print("-"*70)
    
    save_top_influential_biomarkers(
        ranked_df, 
        save_path=str(reports_path / "top_50_influential_biomarkers.txt"),
        n_top=50
    )
    
    # =========================================================================
    # STEP 5: Generate Visualizations
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 5: Generating Visualizations")
    print("-"*70)
    
    # Plot boxplots for top 3 genes
    print("\n📊 Generating Box Plots for Top 3 Genes...")
    top_3_genes = ranked_df.head(3)
    
    for idx, (gene_name, row) in enumerate(top_3_genes.iterrows(), 1):
        save_path = figures_path / f"top{idx}_gene_{gene_name}_boxplot.png"
        plot_gene_expression_boxplot(X, y, gene_name, idx, str(save_path))
        print(f"   ✓ Saved boxplot for {gene_name} (Rank #{idx})")
    
    # Plot coefficient heatmap
    plot_top_genes_heatmap(
        ranked_df,
        save_path=str(figures_path / "biomarker_coefficient_heatmap.png"),
        n_top=20
    )
    
    # =========================================================================
    # STEP 6: Save Full Coefficient Rankings
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 6: Saving Full Coefficient Rankings")
    print("-"*70)
    
    # Save full rankings to CSV
    ranked_df.to_csv(reports_path / "gene_coefficient_rankings.csv")
    print(f"   ✓ Saved full rankings to {reports_path / 'gene_coefficient_rankings.csv'}")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "="*70)
    print("  BIOMARKER INTERPRETATION SUMMARY")
    print("="*70)
    
    # Class-specific top biomarkers
    print("\n  📋 Top Biomarker for Each Cancer Type:")
    print("-"*50)
    for cls in classes:
        # Find gene with highest positive coefficient for this class
        class_top = coef_df[cls].idxmax()
        class_coef = coef_df.loc[class_top, cls]
        print(f"    • {cls}: {class_top} (coef = +{class_coef:.4f})")
    
    print(f"""
  Output Files:
  ─────────────────────────────────────────
  • Top 50 Biomarkers: {reports_path / 'top_50_influential_biomarkers.txt'}
  • Full Rankings CSV: {reports_path / 'gene_coefficient_rankings.csv'}
  • Coefficient Heatmap: {figures_path / 'biomarker_coefficient_heatmap.png'}
  • Top Gene Boxplots: {figures_path / 'top1_gene_*_boxplot.png'}
                       {figures_path / 'top2_gene_*_boxplot.png'}
                       {figures_path / 'top3_gene_*_boxplot.png'}
    """)
    print("="*70)
    print("  Step 4 Complete! Biomarker interpretation analysis finished.")
    print("="*70 + "\n")
    
    return ranked_df, coef_df


if __name__ == "__main__":
    ranked_df, coef_df = main()

