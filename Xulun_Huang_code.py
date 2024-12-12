import sys
import os
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from group_lasso import GroupLasso
import matplotlib.pyplot as plt
import time
import json

from sklearn.datasets import fetch_california_housing

# Parse dataset argument
if "-d" not in sys.argv:
    print("Error: No dataset specified. Use -d bc, -d adult, or -d housing.")
    sys.exit(1)

dataset_index = sys.argv.index("-d") + 1
if dataset_index >= len(sys.argv):
    print("Error: No dataset name after -d. Use -d bc, -d adult, or -d housing.")
    sys.exit(1)

dataset_flag = sys.argv[dataset_index].lower()

if dataset_flag not in ["bc", "adult", "housing"]:
    print("Error: Invalid dataset. Use -d bc, -d adult, or -d housing.")
    sys.exit(1)

if dataset_flag == "bc":
    # BREAST CANCER DATASET
    results_dir = "results_bc"
    column_names = [
        'ID', 'Diagnosis',
        'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness',
        'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension',
        'SE Radius', 'SE Texture', 'SE Perimeter', 'SE Area', 'SE Smoothness',
        'SE Compactness', 'SE Concavity', 'SE Concave Points', 'SE Symmetry', 'SE Fractal Dimension',
        'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area', 'Worst Smoothness',
        'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal Dimension'
    ]

    data = pd.read_csv('../datas/wdbc.data', names=column_names)
    data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})
    data.drop('ID', axis=1, inplace=True)

    X = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']
    feature_names = X.columns

    # Grouping for BC: Natural grouping by feature category (Radius, Texture, etc.)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    groups = []
    for col in X.columns:
        if 'Radius' in col:
            groups.append(0)
        elif 'Texture' in col:
            groups.append(1)
        elif 'Perimeter' in col:
            groups.append(2)
        elif 'Area' in col:
            groups.append(3)
        elif 'Smoothness' in col:
            groups.append(4)
        elif 'Compactness' in col:
            groups.append(5)
        elif 'Concavity' in col:
            groups.append(6)
        elif 'Concave Points' in col:
            groups.append(7)
        elif 'Symmetry' in col:
            groups.append(8)
        elif 'Fractal Dimension' in col:
            groups.append(9)
        else:
            groups.append(10)
    groups = np.array(groups)

elif dataset_flag == "adult":
    # ADULT DATASET
    results_dir = "results_adult"
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
        'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
        'hours_per_week', 'native_country', 'income'
    ]

    data = pd.read_csv('../datas/adult.data', names=column_names, skipinitialspace=True)
    data = data.replace('?', np.nan).dropna()
    data['income'] = data['income'].map({'<=50K':0, '>50K':1})

    X = data.drop('income', axis=1)
    y = data['income']

    numeric_cols = ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    feature_names = X_encoded.columns
    X = X_encoded

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Grouping for ADULT:
    # Here we group features by their original column name before encoding.
    groups = []
    group_map = {}
    group_counter = 0
    for col in X.columns:
        original_feature = col.split('_')[0]
        if original_feature not in group_map:
            group_map[original_feature] = group_counter
            group_counter += 1
        groups.append(group_map[original_feature])
    groups = np.array(groups)

elif dataset_flag == "housing":
   
    results_dir = "results_housing"
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target

    median_val = np.median(y)
    y = (y > median_val).astype(int)

    feature_names = X.columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    
    # Each feature is in its own group since there's no natural grouping given.
    groups = np.arange(X.shape[1])

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Baseline Model
start = time.time()
baseline_clf = LogisticRegression(max_iter=1000)
baseline_clf.fit(X_train, y_train)
baseline_infer_start = time.time()
y_pred_base = baseline_clf.predict(X_test)
baseline_infer_time = time.time() - baseline_infer_start
baseline_f1 = f1_score(y_test, y_pred_base)
baseline_num_features = X_train.shape[1]
baseline_train_time = time.time() - start

lasso_alpha = 0.025
group_lasso_group_reg = 0.025
en_alpha = 0.025
en_l1_ratio = 0.5

# LASSO
lasso_final = Lasso(alpha=lasso_alpha, max_iter=10000)
start = time.time()
lasso_final.fit(X_train, y_train)
lasso_train_time = time.time() - start
lasso_coef = lasso_final.coef_
sel_lasso = np.where(lasso_coef != 0)[0]
lasso_num_features = len(sel_lasso) if len(sel_lasso)>0 else X_train.shape[1]

if len(sel_lasso) == 0:
    X_train_l = X_train
    X_test_l = X_test
else:
    X_train_l = X_train[:, sel_lasso]
    X_test_l = X_test[:, sel_lasso]

clf_l = LogisticRegression(max_iter=1000)
clf_l.fit(X_train_l, y_train)
lasso_infer_start = time.time()
y_pred_lasso = clf_l.predict(X_test_l)
lasso_infer_time = time.time() - lasso_infer_start
lasso_f1 = f1_score(y_test, y_pred_lasso)

# Group LASSO
gl_final = GroupLasso(groups=groups, group_reg=group_lasso_group_reg, n_iter=1000, scale_reg="group_size", supress_warning=True, fit_intercept=True)
start = time.time()
gl_final.fit(X_train, y_train)
gl_train_time = time.time() - start
mask = gl_final.sparsity_mask_.flatten()
sel_gl = np.where(mask)[0]
gl_num_features = len(sel_gl) if len(sel_gl)>0 else X_train.shape[1]

if len(sel_gl) == 0:
    X_train_gl = X_train
    X_test_gl = X_test
else:
    X_train_gl = X_train[:, sel_gl]
    X_test_gl = X_test[:, sel_gl]

clf_gl_final = LogisticRegression(max_iter=1000)
clf_gl_final.fit(X_train_gl, y_train)
gl_infer_start = time.time()
y_pred_gl_final = clf_gl_final.predict(X_test_gl)
gl_infer_time = time.time() - gl_infer_start
gl_f1 = f1_score(y_test, y_pred_gl_final)
gl_coefs = gl_final.coef_.flatten()

# ElasticNet
en_final = ElasticNet(alpha=en_alpha, l1_ratio=en_l1_ratio, max_iter=10000)
start = time.time()
en_final.fit(X_train, y_train)
en_train_time = time.time() - start
en_coef = en_final.coef_
sel_en = np.where(en_coef != 0)[0]
en_num_features = len(sel_en) if len(sel_en)>0 else X_train.shape[1]

if len(sel_en) == 0:
    X_train_en = X_train
    X_test_en = X_test
else:
    X_train_en = X_train[:, sel_en]
    X_test_en = X_test[:, sel_en]

clf_en_final = LogisticRegression(max_iter=1000)
clf_en_final.fit(X_train_en, y_train)
en_infer_start = time.time()
y_pred_en_final = clf_en_final.predict(X_test_en)
en_infer_time = time.time() - en_infer_start
en_f1 = f1_score(y_test, y_pred_en_final)

methods = ['Baseline', 'LASSO', 'Group LASSO', 'ElasticNet']
f1_scores = [baseline_f1, lasso_f1, gl_f1, en_f1]

plt.figure(figsize=(6,4))
plt.bar(methods, f1_scores, color=['gray','blue','green','red'])
plt.title("F1 Score Comparison")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "f1_score_comparison.png"))
plt.close()

feature_counts = [baseline_num_features, lasso_num_features, gl_num_features, en_num_features]

plt.figure(figsize=(6,4))
plt.bar(methods, feature_counts, color=['gray','blue','green','red'])
plt.title("Model Complexity (Number of Features Selected)")
plt.ylabel("Feature Count")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "feature_count_comparison.png"))
plt.close()


def plot_feature_importance(coefs, selected, fname):
    if len(selected) == 0:
        selected = np.arange(len(coefs))
    abs_coefs = np.abs(coefs[selected])
    sorted_idx = np.argsort(abs_coefs)
    
    if len(sorted_idx) > 20:
        sorted_idx = sorted_idx[-20:]
    
    plt.figure(figsize=(6,4))
    plt.barh(range(len(sorted_idx)), abs_coefs[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[selected[i]] for i in sorted_idx])
    plt.title("Feature Importance by Absolute Coefficient")
    plt.xlabel("Absolute Coefficient")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

# LASSO importance
plot_feature_importance(lasso_coef, sel_lasso, os.path.join(results_dir, "lasso_feature_importance.png"))
# Group LASSO importance
plot_feature_importance(gl_coefs, sel_gl, os.path.join(results_dir, "group_lasso_feature_importance.png"))
# ElasticNet importance
plot_feature_importance(en_coef, sel_en, os.path.join(results_dir, "elasticnet_feature_importance.png"))

total_features = X_train.shape[1]
results = {
    "baseline": {
        "f1": baseline_f1,
        "features_selected": baseline_num_features,
        "total_features": total_features,
        "proportion_selected": baseline_num_features/total_features,
        "train_time": baseline_train_time,
        "infer_time": baseline_infer_time
    },
    "lasso": {
        "f1": lasso_f1,
        "alpha": lasso_alpha,
        "features_selected": lasso_num_features,
        "total_features": total_features,
        "proportion_selected": lasso_num_features/total_features,
        "train_time": lasso_train_time,
        "infer_time": lasso_infer_time
    },
    "group_lasso": {
        "f1": gl_f1,
        "group_reg": group_lasso_group_reg,
        "features_selected": gl_num_features,
        "total_features": total_features,
        "proportion_selected": gl_num_features/total_features,
        "train_time": gl_train_time,
        "infer_time": gl_infer_time
    },
    "elasticnet": {
        "f1": en_f1,
        "alpha": en_alpha,
        "l1_ratio": en_l1_ratio,
        "features_selected": en_num_features,
        "total_features": total_features,
        "proportion_selected": en_num_features/total_features,
        "train_time": en_train_time,
        "infer_time": en_infer_time
    }
}

with open(os.path.join(results_dir, "results.json"), "w") as f:
    json.dump(results, f, indent=4)

