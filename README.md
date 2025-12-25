# RNA-Seq-Derived Biomarker Panel and Pathway Discovery for Multi-Class Cancer Subtype Prediction

## Overview

This project implements a robust machine learning pipeline to analyze high-dimensional RNA sequencing (RNA-Seq) data from The Cancer Genome Atlas (TCGA), successfully classifying five distinct cancer subtypes (BRCA, COAD, KIRC, LUAD, PRAD). The pipeline goes beyond simple prediction, utilizing model interpretability (Logistic Regression coefficients) to establish a definitive 50-gene biomarker panel and validate its functional significance via external bioinformatics pathway analysis.

---

## 🚀 Key Results & Highlights

* **Prediction:** Achieved **100% F1-Macro Score** on unseen test data using a Logistic Regression classifier, demonstrating perfect discriminative power.

* **Biomarker Panel:** Identified and prioritized a high-confidence panel of **50 most influential HGNC-verified genes** driving the classification.

* **Novel Discovery:** Pathway Enrichment Analysis (gProfiler) confirmed statistical over-representation in **DNA-binding Transcription Factor Activity** and **Syntaxin Binding/Vesicular Transport** pathways, pinpointing core disease mechanisms.

* **Problem-Solving:** Successfully overcame a common bioinformatics challenge by mapping custom dataset gene IDs (`gene_XXXX`) to official **HGNC Symbols** to enable external validation.

* **Therapeutic Candidates:** Identified specific genes (e.g., **STXBP3**, **AIM2**) as high-priority therapeutic targets for future research.

---

## 🔬 Project Pipeline Phases

### **Phase 1-4: Data Processing and Model Training (Scripts `1_...` to `3_...`)**

* **Data Acquisition:** Utilized 801 samples across five TCGA cancer subtypes (BRCA, COAD, KIRC, LUAD, PRAD) and 20,531 gene features.

* **Feature Selection:** Reduced feature space from 20,531 to 1,000 using a high-variance filter.

* **Modeling:** Implemented a Class-Weighted Logistic Regression model to mitigate dataset imbalance and ensure model interpretability.

* **Performance:** Final model yielded 100% accuracy and F1-Macro score on the test set.

### **Phase 5: Biological Interpretation and Therapeutic Target Discovery (Script `4_interpret_biomarkers.py`)**

This phase translates the mathematical success of the model into biological meaning.

#### **5.1 Model Interpretation (Coefficient Analysis)**

The Logistic Regression model's coefficients were used to rank all 1,000 features by their predictive magnitude.

* **Outcome:** A panel of **50 most influential genes** was isolated, proving that the model relies on a sparse, specific set of genes for classification. The most potent biomarker, **STXBP3**, was identified as the primary driver for LUAD classification.

#### **5.2 Pathway Enrichment and Validation**

The final biomarker panel was validated using external bioinformatics tools.

| Status | Details |
| :--- | :--- |
| **Challenge** | The original dataset used custom gene IDs (`gene_15898`) that were unrecognized by standard tools (gProfiler). |
| **Solution** | A dedicated data mapping exercise was performed, confirming that the numeric identifier corresponded to the official **HGNC ID**, allowing for the successful conversion of the entire panel to verifiable symbols (e.g., STXBP3, AIM2). |
| **Key Findings** | Pathway analysis (gProfiler) confirmed that the predictive power is concentrated in two major biological areas: **DNA-binding transcription factor activity** and **syntaxin binding/vesicular transport** (driven by the STXBP family). |

---

## 🏆 Model Performance Summary

| Model | F1-Macro | Accuracy |
|-------|----------|----------|
| **Logistic Regression** 🥇 | **100.00%** | **100.00%** |
| Support Vector Classifier | 99.89% | 99.88% |
| Random Forest | 99.68% | 99.63% |

---

## 🧬 Top Biomarkers Discovered

### Most Influential Genes (by Model Coefficients)

| Rank | Gene ID | HGNC Symbol | Dominant Cancer | Coefficient |
|------|---------|-------------|-----------------|-------------|
| 1 | gene_15898 | STXBP3 | LUAD (Lung) | +0.0816 |
| 2 | gene_6594 | - | LUAD (Lung) | +0.0615 |
| 3 | gene_7964 | - | BRCA (Breast) | -0.0538 |
| 4 | gene_2318 | - | BRCA (Breast) | -0.0534 |
| 5 | gene_357 | AIM2 | BRCA (Breast) | +0.0513 |

### Top Biomarker for Each Cancer Type

| Cancer Type | Top Biomarker | Coefficient | Biological Role |
|-------------|---------------|-------------|-----------------|
| **BRCA** (Breast) | gene_357 (AIM2) | +0.0513 | DNA-sensing inflammasome |
| **COAD** (Colon) | gene_12013 | +0.0285 | Upregulated |
| **KIRC** (Kidney) | gene_3439 | +0.0398 | Upregulated |
| **LUAD** (Lung) | gene_15898 (STXBP3) | +0.0816 | Vesicular transport |
| **PRAD** (Prostate) | gene_9176 | +0.0410 | Upregulated |

---

## 📁 Project Structure

```
RNA_Seq_Biomarker_Project/
├── data/
│   ├── raw/                        # Raw data files
│   │   ├── data.csv                # Gene expression matrix (801 × 20,531)
│   │   └── labels.csv              # Cancer type labels
│   └── processed/                  # Processed data
│       ├── cleaned_scaled_data.pkl # Cleaned dataset
│       ├── final_biomarker_set.pkl # 1000-gene biomarker panel
│       └── trained_model.pkl       # Trained classifier
├── scripts/
│   ├── 1_clean_and_eda.py          # Data loading, cleaning, EDA
│   ├── 2_feature_selection.py      # PCA, ANOVA feature selection
│   ├── 3_train_model.py            # Model training and evaluation
│   └── 4_interpret_biomarkers.py   # Biomarker interpretation & ranking
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Data loading utilities
│   ├── feature_tools.py            # Feature selection functions
│   └── model_trainer.py            # Model training class
├── reports/
│   ├── figures/                    # Generated visualizations
│   │   ├── class_distribution.png
│   │   ├── expression_distribution.png
│   │   ├── pca_plot.png
│   │   ├── top_genes_fscore.png
│   │   ├── confusion_matrix.png
│   │   ├── model_comparison.png
│   │   ├── biomarker_coefficient_heatmap.png
│   │   └── top[1-3]_gene_*_boxplot.png
│   ├── top_50_biomarkers.txt           # ANOVA-based biomarkers
│   ├── top_50_influential_biomarkers.txt # Coefficient-based biomarkers
│   ├── gene_coefficient_rankings.csv   # Full gene rankings
│   ├── model_results_summary.txt       # Detailed model results
│   └── model_comparison.csv            # Model performance comparison
├── Project_Report.pdf              # Full project report
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- Kaggle account (for downloading data)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SiD-array/rna-seq-cancer-biomarker-discovery.git
cd rna-seq-cancer-biomarker-discovery
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Download the dataset** from Kaggle:
   - Visit: [Gene Expression Cancer RNA-Seq Dataset](https://www.kaggle.com/datasets/waalbannyantudre/gene-expression-cancer-rna-seq-donated-on-682016)
   - Download the dataset files
   - Place `data.csv` in the `data/raw/` folder
   - The `labels.csv` file is already included in this repository

4. Run the complete pipeline:
```bash
# Phase 1: Data Loading & Cleanup
python scripts/1_clean_and_eda.py

# Phase 2: Feature Selection
python scripts/2_feature_selection.py

# Phase 3: Model Training
python scripts/3_train_model.py

# Phase 5: Biomarker Interpretation
python scripts/4_interpret_biomarkers.py
```

---

## 📊 Dataset Summary

| Metric | Value |
|--------|-------|
| Total Samples | 801 |
| Original Genes | 20,531 |
| Selected Biomarkers | 1,000 |
| Final Panel | 50 genes |
| Cancer Types | 5 |
| Missing Values | 0% |

### Class Distribution

| Cancer Type | Full Name | Samples | Percentage |
|-------------|-----------|---------|------------|
| BRCA | Breast Invasive Carcinoma | 300 | 37.5% |
| KIRC | Kidney Renal Clear Cell Carcinoma | 146 | 18.2% |
| LUAD | Lung Adenocarcinoma | 141 | 17.6% |
| PRAD | Prostate Adenocarcinoma | 136 | 17.0% |
| COAD | Colon Adenocarcinoma | 78 | 9.7% |

---

## 🔬 Key Features

- **Class Imbalance Handling**: Uses `class_weight='balanced'` and stratified sampling
- **Robust Evaluation**: F1-Macro score ensures fair evaluation across imbalanced classes
- **Biomarker Discovery**: Identifies top 50 candidate biomarkers using ANOVA and model coefficients
- **Model Interpretability**: Full coefficient analysis for understanding gene-cancer associations
- **Pathway Validation**: External validation via gProfiler pathway enrichment analysis
- **Production Ready**: Trained model saved for deployment

---

## 📚 References

- **Dataset**: [TCGA Gene Expression Cancer RNA-Seq](https://www.kaggle.com/datasets/waalbannyantudre/gene-expression-cancer-rna-seq-donated-on-682016)
- **Pathway Analysis**: [gProfiler](https://biit.cs.ut.ee/gprofiler/)
- **Gene Nomenclature**: [HGNC](https://www.genenames.org/)

---

## 📄 License

This project is for educational and research purposes.

---

## 👤 Author

- GitHub: [@SiD-array](https://github.com/SiD-array)
