import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Danh sách datasets
datasets = [
    ("Dataset 1 (Iris)", load_iris()),
    ("Dataset 2 (Breast Cancer)", load_breast_cancer())
]

for name, data in datasets:
    print(f"--- EXPERIMENT: Random Forest on {name} ---")
    
    # 1. Chia dữ liệu 80% Train - 20% Test (Giống hệt phần DT để so sánh công bằng)
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    # 2. Xây dựng Random Forest
    # n_estimators=100: Tạo K=100 cây (tương ứng với 100 tập bootstrap)
    rf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
    rf.fit(X_train, y_train)
    
    # 3. Dự đoán và tính lỗi
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    error = 1 - acc
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Classification Error: {error:.4f}")
    
    # 4. Vẽ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'RF Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    print("-" * 30 + "\n")