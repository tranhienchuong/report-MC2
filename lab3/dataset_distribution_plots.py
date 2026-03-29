import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================
# 1. IRIS DATASET (2D gốc)
# =========================
iris = load_iris()
X_iris = iris.data[:, :2]  # Sepal length, Sepal width
y_iris = iris.target

plt.figure(figsize=(6, 5))
scatter = plt.scatter(
    X_iris[:, 0], X_iris[:, 1],
    c=y_iris, edgecolor='k'
)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Distribution of Iris Dataset (First 2 Features)')
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=iris.target_names.tolist()
)
plt.grid(True)
plt.show()


# ==================================
# 2. BREAST CANCER DATASET (PCA 2D)
# ==================================
cancer = load_breast_cancer()
scaler = StandardScaler()
X_cancer_std = scaler.fit_transform(cancer.data)

pca = PCA(n_components=2)
X_cancer_pca = pca.fit_transform(X_cancer_std)
y_cancer = cancer.target

plt.figure(figsize=(6, 5))
scatter2 = plt.scatter(
    X_cancer_pca[:, 0], X_cancer_pca[:, 1],
    c=y_cancer, edgecolor='k', alpha=0.6
)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Distribution of Breast Cancer Dataset (PCA Projection)')
plt.legend(
    handles=scatter2.legend_elements()[0],
    labels=['Malignant', 'Benign']
)
plt.grid(True)
plt.show()
