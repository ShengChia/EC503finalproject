README for MCFS:
Data preparation:
1.Load the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, lists all 32 feature names, including ID (identifier) and Diagnosis (target label).
2.The diagnosis column (M for malignant, B for benign) is mapped to binary values, Malignant (M) → 1, Benign (B) → 0
3.Check for Missing Values
4.Separate Features and Labels
5. Normalize Features: The features are normalized to have zero mean and unit variance using StandardScaler.

Function used:
construct_W2(X, **kwargs):
Purpose: Constructs an affinity matrix 
W for the dataset X using k-nearest neighbors (KNN) or other specified neighbor modes. This matrix represents the similarity between samples in the dataset.
Inputs:
X: Input data matrix (samples × features).
kwargs: Parameters for constructing W, including:
metric, neighbor mode, weightmode
neighbor_mode: Neighbor mode
k: Number of neighbors (default: 5).
NumPy, SciPy (pairwise_distances for distance computation, sparse matrix operations).


mcfs2(X, y=None, n_selected_features=None, mode='rank', **kwargs):
Purpose: Implements the MCFS algorithm to compute feature weights based on cluster assignments.
1.Constructs the affinity matrix W 
2.Computes the Laplacian matrix L and performs generalized eigen-decomposition to find eigenvectors corresponding to the smallest eigenvalues.
3.Solves K-L1-regularized regression problems to determine feature weights.
4.Ranks features based on their weights.
NumPy, SciPy (eigh for eigen-decomposition), Scikit-learn (LARS for regression).

feature_ranking(W):
Purpose: Ranks features based on their maximum absolute weights in the weight matrix W, computed from MCFS.
Inputs:
W: A feature weight matrix, where each row corresponds to a feature, and each column corresponds to a cluster.
Outputs:
Indices of features sorted in descending order of importance.
NumPy

Libraries:
NumPy: For matrix operations.
SciPy: For sparse matrices and eigen-decomposition.
Pandas: For dataset handling.

Scikit-learn:
pairwise_distances: Distance computation.
StandardScaler: Feature scaling.
Lars: L1-regularized regression.
KMeans: Clustering.
Metrics (adjusted_rand_score, confusion_matrix, classification_report, accuracy_score).








