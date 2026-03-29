from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

datasets = [
    ("Dataset 1 (Iris)", load_iris()),
    ("Dataset 2 (Breast Cancer)", load_breast_cancer())
]

print(f"--- LEAVE-ONE-OUT CROSS VALIDATION ---")

for name, data in datasets:
    # Khởi tạo mô hình k-NN (k=5)
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # Khởi tạo phương pháp Leave-One-Out
    loo = LeaveOneOut()
    
    # Thực hiện Cross Validation
    # scores trả về 1 (đúng) hoặc 0 (sai) cho từng mẫu
    scores = cross_val_score(knn, data.data, data.target, cv=loo, scoring='accuracy')
    
    # Tính Accuracy trung bình
    mean_acc = scores.mean()
    # Tính Error Rate
    error_rate = 1 - mean_acc
    
    print(f"Dataset: {name}")
    print(f"Number of iterations (n): {len(data.data)}")
    print(f"LOOCV Accuracy:    {mean_acc:.4f}")
    print(f"LOOCV Error Rate:  {error_rate:.4f}")
    print("-" * 30)