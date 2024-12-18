# -*- coding: utf-8 -*-
"""MI.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fMOG08t7k7HqcDtveRgE0FBYpAOX-7kP
"""

# 安装必要的包
!pip install ucimlrepo matplotlib pandas
!pip install -U scikit-learn --use-deprecated=legacy-resolver


# 导入必要的库
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

# 加载数据集
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# 提取特征和目标
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# 查看数据的基本信息
print("Dataset Features:\n", X.head())
print("\nTarget Distribution:\n", y.value_counts())


# 计算特征与目标变量的互信息
mi_scores = mutual_info_classif(X, y, random_state=42)

# 创建一个 DataFrame 保存特征名称和对应的 MI 分数
mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
mi_df.sort_values(by='MI Score', ascending=False, inplace=True)

# 打印互信息得分最高的特征
print("\nTop 10 Features by MI:\n", mi_df.head(10))


# 绘制所有特征的 MI 分数
plt.figure(figsize=(12, 6))
plt.bar(mi_df['Feature'], mi_df['MI Score'])
plt.xticks(rotation=90)
plt.title('Feature Importance by Mutual Information')
plt.xlabel('Features')
plt.ylabel('Mutual Information Score')
plt.tight_layout()
plt.show()



# 设定互信息分数的阈值
threshold = 0.1
selected_features = mi_df[mi_df['MI Score'] > threshold]['Feature'].tolist()

# 打印选择的特征
print(f"\nSelected Features (MI Score > {threshold}):\n", selected_features)

# 构建新的特征数据集
X_selected = X[selected_features]




# 按选定特征绘制柱状图
plt.figure(figsize=(10, 6))
selected_mi = mi_df[mi_df['Feature'].isin(selected_features)]
plt.bar(selected_mi['Feature'], selected_mi['MI Score'], color='red')
plt.xticks(rotation=90)
plt.title('Selected Features by Mutual Information')
plt.xlabel('Selected Features')
plt.ylabel('Mutual Information Score')
plt.tight_layout()
plt.show()

# ---------------------
# 计算分类准确率
# ---------------------

# 分割数据集为训练集和测试集
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 使用选定的特征数据集
X_selected = X[selected_features]

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 训练分类模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 预测并计算准确率
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 打印准确率
print(f"\nAccuracy after MI Feature Selection: {accuracy:.4f}")