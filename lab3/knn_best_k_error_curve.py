import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Danh sách datasets
datasets = [
    ("Dataset 1 (Iris)", load_iris()),
    ("Dataset 2 (Breast Cancer)", load_breast_cancer())
]

for name, data in datasets:
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42
    )

    error_rates = []
    k_range = range(1, 31)

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        error_rates.append(1 - accuracy_score(y_test, y_pred))

    # Tìm k tốt nhất
    min_error = min(error_rates)
    best_k = k_range[error_rates.index(min_error)]
    print(f"{name} - Best k: {best_k}, Min Error: {min_error:.4f}")

    # Vẽ MỖI DATASET MỘT ẢNH
    plt.figure(figsize=(7, 5))
    plt.plot(k_range, error_rates, linestyle='dashed', marker='o')
    plt.title(f'Error Rate vs. K Value - {name}')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.grid(True)
    plt.show()
