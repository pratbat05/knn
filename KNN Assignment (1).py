# -*- coding: utf-8 -*-
"""KNN Assignment"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import os

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names


iris_df = pd.DataFrame(data=X, columns=feature_names)
iris_df['species'] = y
iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("First 5 rows of the Iris DataFrame:")
print(iris_df.head())
print("\nDataFrame Info:")
iris_df.info()

output_dir = 'plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("\nGenerating Pairplot...")
pair_plot = sns.pairplot(iris_df, hue='species', palette='viridis')
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
pair_plot.savefig(os.path.join(output_dir, 'pairplot.png'))
plt.show()


print("\nGenerating Correlation Heatmap...")
plt.figure(figsize=(8, 6))
correlation_matrix = iris_df[feature_names].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Iris Features')
plt.savefig(os.path.join(output_dir, 'heatmap.png'))
plt.show()



print("\nGenerating Boxplots...")
plt.figure(figsize=(12, 8))
for i, feature in enumerate(feature_names):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='species', y=feature, data=iris_df, hue='species', palette='viridis', legend=False)
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'boxplots.png'))
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


k_values = [1, 3, 5, 7]


scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1_score': make_scorer(f1_score, average='weighted')
}

print("\n--- Model Performance Results with 5-fold Cross-Validation ---")

results = []
for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)


    cv_results = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')

    print(f"\nResults for k = {k}:")
    print(f"  Cross-Validation Accuracy: {cv_results}")
    print(f"  Average Cross-Validation Accuracy: {np.mean(cv_results):.4f} (+/- {np.std(cv_results) * 2:.4f})")


    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results.append({
        'k': k,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

    print(f"  Test Set Metrics:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall: {recall:.4f}")
    print(f"    F1 Score: {f1:.4f}")
