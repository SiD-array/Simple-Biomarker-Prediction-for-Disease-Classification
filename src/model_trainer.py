"""
Model Trainer Module
====================
Class and functions for model training, cross-validation, and evaluation.
Implements class imbalance mitigation strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score, 
    accuracy_score,
    precision_score,
    recall_score
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    A class to train and evaluate multiple classifiers with imbalance mitigation.
    
    Attributes
    ----------
    X : pd.DataFrame
        Feature matrix (samples × genes)
    y : np.ndarray
        Encoded labels
    classes : np.ndarray
        Original class names
    cv : StratifiedKFold
        Cross-validation splitter
    models : Dict
        Dictionary of model instances
    results : Dict
        Dictionary to store results
    """
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42):
        """
        Initialize the ModelTrainer.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Class labels
        n_splits : int
            Number of cross-validation folds
        random_state : int
            Random seed for reproducibility
        """
        self.X = X
        self.random_state = random_state
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(y)
        self.classes = self.label_encoder.classes_
        
        # Initialize stratified k-fold
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Initialize models with class_weight='balanced'
        self.models = self._initialize_models()
        
        # Results storage
        self.results = {}
        self.cv_predictions = {}
        
        print(f"  ✓ ModelTrainer initialized")
        print(f"    • Samples: {X.shape[0]}, Features: {X.shape[1]}")
        print(f"    • Classes: {list(self.classes)}")
        print(f"    • CV Folds: {n_splits}")
    
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize all models with balanced class weights."""
        models = {
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state,
                solver='lbfgs'
            ),
            'Support Vector Classifier': SVC(
                class_weight='balanced',
                kernel='rbf',
                random_state=self.random_state,
                probability=True
            ),
            'Random Forest': RandomForestClassifier(
                class_weight='balanced',
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        return models
    
    def evaluate_model(self, model_name: str) -> Dict[str, float]:
        """
        Evaluate a single model using cross-validation.
        
        Parameters
        ----------
        model_name : str
            Name of the model to evaluate
            
        Returns
        -------
        Dict[str, float]
            Dictionary with evaluation metrics
        """
        model = self.models[model_name]
        
        # Cross-validation scores
        f1_scores = cross_val_score(
            model, self.X, self.y, 
            cv=self.cv, 
            scoring='f1_macro',
            n_jobs=-1
        )
        
        accuracy_scores = cross_val_score(
            model, self.X, self.y,
            cv=self.cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Get cross-validated predictions for confusion matrix
        y_pred_cv = cross_val_predict(model, self.X, self.y, cv=self.cv, n_jobs=-1)
        self.cv_predictions[model_name] = y_pred_cv
        
        results = {
            'f1_macro_mean': f1_scores.mean(),
            'f1_macro_std': f1_scores.std(),
            'accuracy_mean': accuracy_scores.mean(),
            'accuracy_std': accuracy_scores.std(),
            'f1_scores': f1_scores,
            'accuracy_scores': accuracy_scores
        }
        
        self.results[model_name] = results
        return results
    
    def evaluate_all_models(self) -> pd.DataFrame:
        """
        Evaluate all models and return summary DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Summary of model performances
        """
        print("\n📊 Evaluating Models with Stratified 5-Fold Cross-Validation")
        print("   (Using class_weight='balanced' for imbalance mitigation)")
        print("-" * 60)
        
        summary_data = []
        
        for model_name in self.models.keys():
            print(f"\n  🔄 Evaluating {model_name}...")
            results = self.evaluate_model(model_name)
            
            print(f"     • F1-Macro: {results['f1_macro_mean']:.4f} (±{results['f1_macro_std']:.4f})")
            print(f"     • Accuracy: {results['accuracy_mean']:.4f} (±{results['accuracy_std']:.4f})")
            
            summary_data.append({
                'Model': model_name,
                'F1-Macro (Mean)': results['f1_macro_mean'],
                'F1-Macro (Std)': results['f1_macro_std'],
                'Accuracy (Mean)': results['accuracy_mean'],
                'Accuracy (Std)': results['accuracy_std']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('F1-Macro (Mean)', ascending=False)
        
        return summary_df
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model based on F1-Macro score.
        
        Returns
        -------
        Tuple[str, Any]
            Best model name and instance
        """
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1_macro_mean'])
        return best_model_name, self.models[best_model_name]
    
    def train_final_model(self, model_name: str = None) -> Any:
        """
        Train the final model on the full dataset.
        
        Parameters
        ----------
        model_name : str, optional
            Name of model to train. If None, uses best model.
            
        Returns
        -------
        Any
            Trained model
        """
        if model_name is None:
            model_name, _ = self.get_best_model()
        
        print(f"\n🏆 Training Final Model: {model_name}")
        model = self.models[model_name]
        model.fit(self.X, self.y)
        
        return model
    
    def get_classification_report(self, model_name: str = None) -> str:
        """
        Generate classification report for cross-validated predictions.
        
        Parameters
        ----------
        model_name : str, optional
            Model name. If None, uses best model.
            
        Returns
        -------
        str
            Classification report
        """
        if model_name is None:
            model_name, _ = self.get_best_model()
        
        y_pred = self.cv_predictions[model_name]
        report = classification_report(
            self.y, y_pred, 
            target_names=self.classes,
            digits=4
        )
        return report
    
    def get_detailed_metrics(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get detailed per-class metrics.
        
        Parameters
        ----------
        model_name : str, optional
            Model name. If None, uses best model.
            
        Returns
        -------
        Dict[str, Any]
            Detailed metrics dictionary
        """
        if model_name is None:
            model_name, _ = self.get_best_model()
        
        y_pred = self.cv_predictions[model_name]
        
        metrics = {
            'precision_per_class': precision_score(self.y, y_pred, average=None),
            'recall_per_class': recall_score(self.y, y_pred, average=None),
            'f1_per_class': f1_score(self.y, y_pred, average=None),
            'precision_macro': precision_score(self.y, y_pred, average='macro'),
            'recall_macro': recall_score(self.y, y_pred, average='macro'),
            'f1_macro': f1_score(self.y, y_pred, average='macro'),
            'accuracy': accuracy_score(self.y, y_pred)
        }
        
        return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          classes: np.ndarray, save_path: str,
                          model_name: str = "Best Model"):
    """
    Plot and save confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    classes : np.ndarray
        Class names
    save_path : str
        Path to save the figure
    model_name : str
        Model name for title
    """
    print(f"\n📊 Generating Confusion Matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Count-based confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=axes[0],
                cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].set_title(f'Confusion Matrix (Counts)\n{model_name}', fontsize=12, fontweight='bold')
    
    # Plot 2: Normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
                xticklabels=classes, yticklabels=classes, ax=axes[1],
                cbar_kws={'label': 'Percentage'}, vmin=0, vmax=1)
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].set_title(f'Confusion Matrix (Normalized)\n{model_name}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved confusion matrix to {save_path}")


def plot_model_comparison(summary_df: pd.DataFrame, save_path: str):
    """
    Plot model comparison bar chart.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Model performance summary
    save_path : str
        Path to save the figure
    """
    print(f"\n📊 Generating Model Comparison Plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(summary_df))
    width = 0.35
    
    # F1-Macro bars
    bars1 = ax.bar(x - width/2, summary_df['F1-Macro (Mean)'], width, 
                   yerr=summary_df['F1-Macro (Std)'], 
                   label='F1-Macro', color='#3498DB', capsize=5, alpha=0.8)
    
    # Accuracy bars
    bars2 = ax.bar(x + width/2, summary_df['Accuracy (Mean)'], width,
                   yerr=summary_df['Accuracy (Std)'],
                   label='Accuracy', color='#2ECC71', capsize=5, alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Model Performance Comparison\n(5-Fold Stratified Cross-Validation)', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Model'], rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved model comparison plot to {save_path}")


def save_results_summary(summary_df: pd.DataFrame, classification_report: str,
                         metrics: Dict, save_path: str, best_model_name: str):
    """
    Save comprehensive results summary to text file.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Model performance summary
    classification_report : str
        Classification report string
    metrics : Dict
        Detailed metrics
    save_path : str
        Path to save the file
    best_model_name : str
        Name of the best model
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("  RNA-Seq BIOMARKER PROJECT - Model Training Results\n")
        f.write("="*70 + "\n\n")
        
        f.write("EVALUATION SETTINGS\n")
        f.write("-"*70 + "\n")
        f.write("• Cross-Validation: 5-Fold Stratified\n")
        f.write("• Primary Metric: F1-Macro (handles class imbalance)\n")
        f.write("• Imbalance Mitigation: class_weight='balanced'\n\n")
        
        f.write("MODEL COMPARISON\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Model':<30}{'F1-Macro':<15}{'Accuracy':<15}\n")
        f.write("-"*70 + "\n")
        for _, row in summary_df.iterrows():
            f.write(f"{row['Model']:<30}{row['F1-Macro (Mean)']:.4f} ± {row['F1-Macro (Std)']:.4f}  "
                    f"{row['Accuracy (Mean)']:.4f} ± {row['Accuracy (Std)']:.4f}\n")
        
        f.write(f"\n🏆 BEST MODEL: {best_model_name}\n\n")
        
        f.write("DETAILED CLASSIFICATION REPORT (Best Model)\n")
        f.write("-"*70 + "\n")
        f.write(classification_report)
        f.write("\n")
        
        f.write("PER-CLASS METRICS (Best Model)\n")
        f.write("-"*70 + "\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro Precision: {metrics['precision_macro']:.4f}\n")
        f.write(f"Macro Recall: {metrics['recall_macro']:.4f}\n")
        f.write(f"Macro F1-Score: {metrics['f1_macro']:.4f}\n")
        f.write("="*70 + "\n")
    
    print(f"   ✓ Saved results summary to {save_path}")

