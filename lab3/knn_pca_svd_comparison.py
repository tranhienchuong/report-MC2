from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

datasets = [
    ("Dataset 1 (Iris)", load_iris()),
    ("Dataset 2 (Breast Cancer)", load_breast_cancer())
]

for name, data in datasets:
    print(f"--- EXPERIMENT: {name} ---")
    
    # 1. Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
    
    # 2. Chuẩn hóa dữ liệu (Bắt buộc cho PCA)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    # Lấy accuracy gốc (đã chuẩn hóa) để so sánh
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_std, y_train)
    acc_orig = accuracy_score(y_test, knn.predict(X_test_std))
    
    # 3. Áp dụng PCA (Giảm xuống 2 chiều)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    
    # Chạy k-NN trên PCA
    knn.fit(X_train_pca, y_train)
    acc_pca = accuracy_score(y_test, knn.predict(X_test_pca))
    
    # 4. Áp dụng SVD (Giảm xuống 2 chiều)
    svd = TruncatedSVD(n_components=2)
    X_train_svd = svd.fit_transform(X_train_std)
    X_test_svd = svd.transform(X_test_std)
    
    # Chạy k-NN trên SVD
    knn.fit(X_train_svd, y_train)
    acc_svd = accuracy_score(y_test, knn.predict(X_test_svd))
    
    print(f"Accuracy (Original Normalized): {acc_orig:.4f}")
    print(f"Accuracy (PCA - 2 components):  {acc_pca:.4f}")
    print(f"Accuracy (SVD - 2 components):  {acc_svd:.4f}")
    print(f"Variance Explained (PCA):       {sum(pca.explained_variance_ratio_):.2f}")
    print("-" * 30 + "\n")