# Simple Biomarker Prediction for Disease Classification

A complete machine learning pipeline for identifying biomarkers from RNA-Seq gene expression data to classify different cancer types with **100% accuracy**.

## 🎯 Project Overview

This project analyzes RNA-Seq expression data from **801 samples** across **5 cancer types** to identify potential biomarkers for disease classification:

- **BRCA** - Breast Invasive Carcinoma (300 samples)
- **KIRC** - Kidney Renal Clear Cell Carcinoma (146 samples)
- **LUAD** - Lung Adenocarcinoma (141 samples)
- **PRAD** - Prostate Adenocarcinoma (136 samples)
- **COAD** - Colon Adenocarcinoma (78 samples)

## 🏆 Results

| Model | F1-Macro | Accuracy |
|-------|----------|----------|
| **Logistic Regression** 🥇 | **100.00%** | **100.00%** |
| Support Vector Classifier | 99.89% | 99.88% |
| Random Forest | 99.68% | 99.63% |

The 1000-gene biomarker panel achieves perfect classification across all 5 cancer types using 5-fold stratified cross-validation.

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
│   └── 3_train_model.py            # Model training and evaluation
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
│   │   └── model_comparison.png
│   ├── top_50_biomarkers.txt       # Top biomarker candidates
│   ├── model_results_summary.txt   # Detailed model results
│   └── model_comparison.csv        # Model performance comparison
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- Kaggle account (for downloading data)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SiD-array/Simple-Biomarker-Prediction-for-Disease-Classification.git
cd Simple-Biomarker-Prediction-for-Disease-Classification
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
# Step 1: Data Loading & Cleanup
python scripts/1_clean_and_eda.py

# Step 2: Feature Selection
python scripts/2_feature_selection.py

# Step 3: Model Training
python scripts/3_train_model.py
```

## 📊 Dataset Summary

| Metric | Value |
|--------|-------|
| Total Samples | 801 |
| Original Genes | 20,531 |
| Selected Biomarkers | 1,000 |
| Cancer Types | 5 |
| Missing Values | 0% |

### Class Distribution

| Cancer Type | Samples | Percentage |
|-------------|---------|------------|
| BRCA | 300 | 37.5% |
| KIRC | 146 | 18.2% |
| LUAD | 141 | 17.6% |
| PRAD | 136 | 17.0% |
| COAD | 78 | 9.7% |

## 📈 Pipeline Steps

### Step 1: Data Loading & Initial Cleanup ✅
- Load RNA-Seq expression data (801 samples × 20,531 genes)
- Validate data format and quality
- Check for missing values (none found)
- Remove 267 constant genes
- Generate initial visualizations

### Step 2: Feature Selection ✅
- PCA analysis (478 components for 95% variance)
- ANOVA F-test for statistical feature selection
- Select top 1,000 discriminative genes
- Identify biomarker candidates

### Step 3: Model Training ✅
- Train 3 classifiers with class imbalance mitigation:
  - Logistic Regression
  - Support Vector Classifier
  - Random Forest
- 5-fold stratified cross-validation
- Comprehensive evaluation with F1-Macro scoring

## 🔬 Key Features

- **Class Imbalance Handling**: Uses `class_weight='balanced'` and stratified sampling
- **Robust Evaluation**: F1-Macro score ensures fair evaluation across imbalanced classes
- **Biomarker Discovery**: Identifies top 50 candidate biomarkers for cancer classification
- **Production Ready**: Trained model saved for deployment

## 📄 License

This project is for educational purposes.

## 👤 Author

- GitHub: [@SiD-array](https://github.com/SiD-array)



