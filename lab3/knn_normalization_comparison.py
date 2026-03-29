from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    
    # 2. Chạy k-NN trên dữ liệu GỐC (Raw Data) - k=5
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    acc_raw = accuracy_score(y_test, knn.predict(X_test))
    
    # 3. Chuẩn hóa dữ liệu (Normalization)
    scaler = StandardScaler()
    # Lưu ý: Chỉ fit trên tập Train, rồi transform cả Train và Test để tránh "data leakage"
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    # 4. Chạy k-NN trên dữ liệu ĐÃ CHUẨN HÓA (Normalized Data)
    knn.fit(X_train_norm, y_train)
    acc_norm = accuracy_score(y_test, knn.predict(X_test_norm))
    
    print(f"Accuracy (Raw):        {acc_raw:.4f}")
    print(f"Accuracy (Normalized): {acc_norm:.4f}")
    print(f"Improvement:           {(acc_norm - acc_raw):.4f}")
    print("-" * 30 + "\n")