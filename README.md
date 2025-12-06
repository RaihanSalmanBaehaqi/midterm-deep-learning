# Deep Learning Midterm Project

---

## Student Identification

| Field | Information |
|-------|-------------|
| **Name** | RAIHAN SALMAN BAEHAQI |
| **Class** |  |
| **Student ID (NIM)** | 1103220180 |

---

## Repository Purpose

This repository contains my Deep Learning midterm assignment, which demonstrates the implementation of end-to-end deep learning models for three different machine learning tasks: classification, regression, and clustering. The purpose is to apply deep learning techniques to real-world datasets and evaluate model performance using appropriate metrics.

---

## Project Overview

This project includes three main tasks:

### 1. Fraud Detection (Classification)
- **Goal**: Predict whether an online transaction is fraudulent
- **Dataset**: Transaction data with 394 features (transaction amount, time, card info, etc.)
- **Challenge**: Highly imbalanced dataset (~3.5% fraud cases)

### 2. Music Release Year Prediction (Regression)
- **Goal**: Predict the release year of songs based on audio features
- **Dataset**: Audio features from songs (90 features including timbre characteristics)
- **Challenge**: Predicting temporal information from audio signals

### 3. Customer Segmentation (Clustering)
- **Goal**: Group customers based on credit card usage behavior
- **Dataset**: Customer credit card data (17 features including balance, purchases, payments)
- **Challenge**: Finding meaningful customer segments without labels

---

## Models and Metrics

### Fraud Detection

**Models Used:**
- Baseline Deep Neural Network (DNN)
  - Architecture: Input → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
- Improved DNN with Batch Normalization
  - Architecture: Input → Dense(256, ReLU) → BatchNorm → Dropout(0.4) → Dense(128, ReLU) → BatchNorm → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)

**Metrics:**
- **Accuracy**: Overall prediction correctness
- **Precision**: Proportion of predicted frauds that are actually fraudulent
- **Recall**: Proportion of actual frauds that are detected
- **F1-Score**: Harmonic mean of Precision and Recall (main metric due to class imbalance)
- **ROC-AUC**: Model's ability to distinguish between classes

**Results:**
```
Model:
- Accuracy:
- Precision:
- Recall:
- F1-Score:
- ROC-AUC:
```

---

### Music Year Prediction

**Models Used:**
- Baseline Regression DNN
  - Architecture: Input → Dense(128, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Linear)
- Deep Regression Model with Regularization
  - Architecture: Input → Dense(256, ReLU) → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Linear)

**Metrics:**
- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual years
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in years (main metric)
- **MAE (Mean Absolute Error)**: Average absolute difference in years
- **R² Score**: Proportion of variance explained by the model

**Results:**
```
Model:
- MSE:
- RMSE: years
- MAE: years
- R² Score:
```

---

### Customer Segmentation

**Models Used:**
- K-Means Clustering
- K-Means on Autoencoder Features
  - Autoencoder: Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Latent(8) → Dense(16, ReLU) → Dense(32, ReLU) → Dense(64, ReLU) → Output
- DBSCAN (optional)
- Hierarchical Clustering (optional)

**Metrics:**
- **Silhouette Score**: Measures how similar objects are to their own cluster vs other clusters (range: -1 to 1, higher is better)
- **Davies-Bouldin Index**: Average similarity ratio of clusters (lower is better)
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster dispersion (higher is better)
- **Inertia (WCSS)**: Within-cluster sum of squares (used in Elbow method)

**Results:**
```
Model:
- Number of Clusters:
- Silhouette Score:
- Davies-Bouldin Index:
- Calinski-Harabasz Score:

Cluster Descriptions:
- Cluster 0: [e.g., "High spenders with frequent installment purchases"]
- Cluster 1: [e.g., "Low balance, cash advance users"]
- ...
```

---

## How to Navigate This Repository

### Repository Structure
```
midterm-deep-learning/
│
├── README.md # This file
├── 01.Transaction-Midterm/ # Fraud Detection Project
│ ├── 01_transaction_midterm.ipynb # Notebook for fraud detection
│
├── 02.Regresi-Midterm/ # Music Year Prediction Project
│ ├── 02_regresi_midterm.ipynb # Notebook for music year prediction
│
└── 03.Clustering-Midterm/ # Customer Segmentation Project
├── 03_clustering_midterm.ipynb # Notebook for customer segmentation
```

### Running the Notebooks

**Step 1: Access the Notebooks**
- Click on any `.ipynb` file in the repository
- Open it in Google Colab or Jupyter Notebook

**Step 2: Download Datasets**
- Fraud Detection: [Download here](https://drive.google.com/drive/folders/1JvI5xhPfN3VmjpWYZk9fCHG41xG697um)
- Music Regression: [Download here](https://drive.google.com/file/d/1f8eaAZY-7YgFxLcrL3OkvSRa3onNNLb9/view)
- Customer Clustering: [Download here](https://drive.google.com/drive/folders/1FsQtOI_QES15zZLmEw099MGAR5-rnsOP)

**Step 3: Run the Code**
- If using Google Colab: Upload dataset or mount Google Drive
- Enable GPU: Runtime → Change runtime type → GPU
- Run cells sequentially from top to bottom

### Notebook Contents

**01_transaction_midterm.ipynb** - Fraud Detection
1. Data Loading & EDA
2. Data Preprocessing (missing values, encoding, scaling)
3. Handling Class Imbalance (SMOTE, class weights)
4. Model Building (DNN architecture)
5. Training & Evaluation
6. Results & Predictions

**02_regresi_midterm.ipynb** - Music Year Prediction
1. Data Loading & EDA
2. Data Preprocessing (scaling, outliers)
3. Model Building (Regression DNN)
4. Training & Evaluation
5. Results & Predictions

**03_clustering_midterm.ipynb** - Customer Segmentation
1. Data Loading & EDA
2. Data Preprocessing (scaling, missing values)
3. Dimensionality Reduction (optional: Autoencoder)
4. Clustering (K-Means, DBSCAN, etc.)
5. Evaluation & Cluster Analysis
6. Cluster Interpretation

---

## Additional Information

**Datasets Used:**
- Transaction dataset: 590,540 records, 394 features
- Music dataset: 515,345 records, 90 audio features
- Customer dataset: 8,950 records, 17 credit card features

**Tools & Libraries:**
- TensorFlow/Keras or PyTorch
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
