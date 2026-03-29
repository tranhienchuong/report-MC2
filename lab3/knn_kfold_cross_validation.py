from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

datasets = [
    ("Dataset 1 (Iris)", load_iris()),
    ("Dataset 2 (Breast Cancer)", load_breast_cancer())
]

# Số lượng fold (k trong k-fold)
num_folds = 5

print(f"--- k-FOLD CROSS VALIDATION (k={num_folds}) ---")

for name, data in datasets:
    # Khởi tạo mô hình k-NN (k=5)
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # Thực hiện Cross Validation
    # cv=5: Chia dữ liệu làm 5 phần, chạy 5 lần.
    scores = cross_val_score(knn, data.data, data.target, cv=num_folds, scoring='accuracy')
    
    # Tính trung bình và độ lệch chuẩn
    mean_acc = scores.mean()
    std_acc = scores.std()
    
    print(f"Dataset: {name}")
    print(f"Accuracy scores for each fold: {scores}")
    print(f"Average Accuracy: {mean_acc:.4f} (+/- {std_acc*2:.4f})")
    print("-" * 30)