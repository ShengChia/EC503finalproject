README for RFE and K-Means Clustering in Exploration of Feature Selection Project

Overview
This code (rfekeam.m) implements Recursive Feature Elimination (RFE) and K-Means Clustering for feature selection on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. The primary goal is to evaluate and compare the performance of these feature selection methods on a supervised classification task using logistic regression.

Dataset
- Wisconsin Diagnostic Breast Cancer (WDBC) dataset
- Contains 30 features extracted from digitized images of fine needle aspirate (FNA) of breast masses. The dataset includes labels for diagnosis (‘M’ for malignant, ‘B’ for benign).
- Source: UCI Machine Learning Repository

Data Preprocessing
1. Data Loading:
   - The data is loaded using MATLAB’s `readtable` function.
2. Normalization:
   - All features are normalized using the `normalize` function to ensure uniform scale across features.
3. Label Encoding:
   - Diagnosis labels are converted from categorical (‘M’/‘B’) to binary (1 for ‘M’, 0 for ‘B’).

Code Functions

1. Recursive Feature Elimination (RFE)
Functionality:
- Iteratively removes the least important feature based on weights learned by logistic regression.
- Produces a ranked list of features and identifies the most significant features using a threshold.

How to Use:
- Input: Normalized feature matrix `X`, binary labels `y`, and a threshold value `thresh`.
- Output: Ranked features, weights of selected features, and their indices.

Dependencies:
- MATLAB’s `fitclinear` for training logistic regression models.
- `abs` for computing feature scores from model weights.

2. K-Means Clustering for Feature Selection
Functionality:
- Groups features into clusters using a correlation matrix.
- Selects one representative feature per cluster based on the highest average correlation.

How to Use:
- Input: Correlation matrix of features, number of clusters k, and maximum iterations `maxIter`.
- Output: Selected feature indices per cluster and their respective weights.

Dependencies:
- `pdist2` for computing distances between features and cluster centers.
- `mean` for updating cluster centers and computing representative features.

3. Evaluation Metrics
- Evaluates selected features using logistic regression on a held-out test set.
- Computes accuracy, precision, recall, F1-score, and confusion matrix.

How to Use:
- Input: Training and testing datasets, selected feature indices.
- Output: Classification performance metrics.

Dependencies:
- MATLAB’s `cvpartition` for creating train-test splits.
- `fitclinear` for training logistic regression models.
- `confusionmat` for confusion matrix computation.

Note:
1. Ensure the `wdbc.data` file is available in the working directory.
2. Set a threshold value for feature selection (in the code, `thresh = 0.5`).
3. Run the code

