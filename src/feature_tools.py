"""
Feature Tools Module
====================
Functions for dimensionality reduction and feature selection.
Includes PCA analysis and statistical filtering (ANOVA).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder


def drop_constant_genes(data: pd.DataFrame, variance_threshold: float = 0.0) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove genes with zero or low variance (constant genes).
    
    Parameters
    ----------
    data : pd.DataFrame
        Expression data (samples × genes)
    variance_threshold : float
        Minimum variance threshold. Default 0.0 removes only constant genes.
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - Filtered DataFrame with constant genes removed
        - List of removed gene names
    """
    print("\n🧹 Removing Constant Genes...")
    
    # Calculate variance for each gene
    gene_variance = data.var()
    
    # Identify constant genes (variance <= threshold)
    constant_genes = gene_variance[gene_variance <= variance_threshold].index.tolist()
    
    # Drop constant genes
    data_filtered = data.drop(columns=constant_genes)
    
    print(f"   • Original genes: {data.shape[1]}")
    print(f"   • Constant genes removed: {len(constant_genes)}")
    print(f"   • Remaining genes: {data_filtered.shape[1]}")
    
    return data_filtered, constant_genes


def perform_pca_analysis(data: pd.DataFrame, n_components: int = None) -> Dict[str, Any]:
    """
    Perform PCA analysis on gene expression data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Expression data (samples × genes)
    n_components : int, optional
        Number of components to compute. If None, computes min(n_samples, n_features).
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - pca: Fitted PCA object
        - explained_variance_ratio: Array of explained variance ratios
        - cumulative_variance: Array of cumulative explained variance
        - n_components_95: Number of components for 95% variance
        - n_components_99: Number of components for 99% variance
    """
    print("\n📊 Performing PCA Analysis...")
    
    # Fit PCA
    if n_components is None:
        n_components = min(data.shape[0], data.shape[1])
    
    pca = PCA(n_components=n_components)
    pca.fit(data)
    
    # Calculate cumulative variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Find components needed for different variance thresholds
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1
    
    results = {
        'pca': pca,
        'explained_variance_ratio': explained_variance,
        'cumulative_variance': cumulative_variance,
        'n_components_95': n_components_95,
        'n_components_99': n_components_99,
        'total_components': len(explained_variance)
    }
    
    print(f"   • Total components computed: {len(explained_variance)}")
    print(f"   • Components for 95% variance: {n_components_95}")
    print(f"   • Components for 99% variance: {n_components_99}")
    print(f"   • First 10 components explain: {cumulative_variance[9]*100:.2f}% variance")
    
    return results


def plot_scree_plot(pca_results: Dict[str, Any], save_path: str, n_components_show: int = 50):
    """
    Create and save a scree plot showing explained variance.
    
    Parameters
    ----------
    pca_results : Dict[str, Any]
        Results from perform_pca_analysis()
    save_path : str
        Path to save the figure
    n_components_show : int
        Number of components to show in the plot
    """
    print(f"\n📈 Generating Scree Plot...")
    
    explained_var = pca_results['explained_variance_ratio'][:n_components_show]
    cumulative_var = pca_results['cumulative_variance'][:n_components_show]
    n_components_95 = pca_results['n_components_95']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Individual explained variance
    ax1.bar(range(1, len(explained_var) + 1), explained_var * 100, 
            color='#3498DB', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Principal Component', fontsize=11)
    ax1.set_ylabel('Explained Variance (%)', fontsize=11)
    ax1.set_title('Scree Plot: Individual Explained Variance', fontsize=12, fontweight='bold')
    ax1.set_xlim(0.5, n_components_show + 0.5)
    
    # Plot 2: Cumulative explained variance
    ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var * 100, 
             'o-', color='#E74C3C', linewidth=2, markersize=4)
    ax2.axhline(y=95, color='#2ECC71', linestyle='--', linewidth=2, label='95% threshold')
    ax2.axvline(x=n_components_95, color='#9B59B6', linestyle='--', linewidth=2, 
                label=f'{n_components_95} components')
    ax2.fill_between(range(1, len(cumulative_var) + 1), cumulative_var * 100, 
                     alpha=0.3, color='#E74C3C')
    ax2.set_xlabel('Number of Principal Components', fontsize=11)
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=11)
    ax2.set_title('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_xlim(0.5, n_components_show + 0.5)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved scree plot to {save_path}")


def select_top_genes_anova(data: pd.DataFrame, labels: pd.Series, k: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select top K genes using ANOVA F-test (statistical filter).
    
    Parameters
    ----------
    data : pd.DataFrame
        Expression data (samples × genes)
    labels : pd.Series
        Class labels
    k : int
        Number of top genes to select
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - Filtered DataFrame with top K genes
        - DataFrame with gene rankings (gene_name, f_score, p_value, rank)
    """
    print(f"\n🔬 Performing ANOVA Feature Selection (Top {k} genes)...")
    
    # Encode labels if needed
    if labels.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(labels)
    else:
        y = labels.values
    
    # Perform ANOVA F-test
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(data, y)
    
    # Get scores and p-values
    f_scores = selector.scores_
    p_values = selector.pvalues_
    
    # Create ranking DataFrame
    gene_rankings = pd.DataFrame({
        'gene_name': data.columns,
        'f_score': f_scores,
        'p_value': p_values
    })
    gene_rankings['rank'] = gene_rankings['f_score'].rank(ascending=False).astype(int)
    gene_rankings = gene_rankings.sort_values('rank')
    
    # Get selected gene names
    selected_mask = selector.get_support()
    selected_genes = data.columns[selected_mask].tolist()
    
    # Filter data to selected genes
    data_selected = data[selected_genes]
    
    print(f"   • Original genes: {data.shape[1]}")
    print(f"   • Selected genes: {data_selected.shape[1]}")
    print(f"   • Top gene: {gene_rankings.iloc[0]['gene_name']} (F={gene_rankings.iloc[0]['f_score']:.2f})")
    print(f"   • Min F-score in top {k}: {gene_rankings.iloc[k-1]['f_score']:.2f}")
    
    return data_selected, gene_rankings


def plot_top_genes_scores(gene_rankings: pd.DataFrame, save_path: str, n_top: int = 50):
    """
    Plot F-scores of top genes.
    
    Parameters
    ----------
    gene_rankings : pd.DataFrame
        Gene rankings from select_top_genes_anova()
    save_path : str
        Path to save the figure
    n_top : int
        Number of top genes to show
    """
    print(f"\n📊 Plotting Top {n_top} Gene F-Scores...")
    
    top_genes = gene_rankings.head(n_top)
    
    plt.figure(figsize=(12, 10))
    
    # Horizontal bar plot
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, n_top))
    bars = plt.barh(range(n_top), top_genes['f_score'].values, color=colors, edgecolor='black', linewidth=0.5)
    
    plt.yticks(range(n_top), top_genes['gene_name'].values, fontsize=8)
    plt.xlabel('ANOVA F-Score', fontsize=11)
    plt.ylabel('Gene Name', fontsize=11)
    plt.title(f'Top {n_top} Biomarker Candidates by ANOVA F-Score', fontsize=12, fontweight='bold')
    
    # Invert y-axis so rank 1 is at top
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved top genes plot to {save_path}")


def save_top_biomarkers(gene_rankings: pd.DataFrame, save_path: str, n_top: int = 50):
    """
    Save top biomarker gene names to a text file.
    
    Parameters
    ----------
    gene_rankings : pd.DataFrame
        Gene rankings from select_top_genes_anova()
    save_path : str
        Path to save the text file
    n_top : int
        Number of top genes to save
    """
    top_genes = gene_rankings.head(n_top)
    
    with open(save_path, 'w') as f:
        f.write(f"Top {n_top} Biomarker Candidates for Cancer Classification\n")
        f.write("=" * 60 + "\n")
        f.write(f"Selection Method: ANOVA F-test (SelectKBest)\n")
        f.write(f"Classes: BRCA, COAD, KIRC, LUAD, PRAD\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Rank':<6}{'Gene Name':<20}{'F-Score':<15}{'P-Value':<15}\n")
        f.write("-" * 60 + "\n")
        
        for _, row in top_genes.iterrows():
            f.write(f"{int(row['rank']):<6}{row['gene_name']:<20}{row['f_score']:<15.2f}{row['p_value']:<15.2e}\n")
    
    print(f"   ✓ Saved top {n_top} biomarkers to {save_path}")

