import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Danh sách datasets
datasets = [
    ("Dataset 1 (Iris)", load_iris()),
    ("Dataset 2 (Breast Cancer)", load_breast_cancer())
]

for name, data in datasets:
    print(f"--- EXPERIMENT: Decision Tree on {name} ---")
    
    # 1. Chia dữ liệu 80% Train - 20% Test (Theo đề bài Lab 4)
    # random_state cố định để kết quả không đổi khi chạy lại
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    # 2. Xây dựng mô hình Decision Tree
    # criterion='entropy': Sử dụng độ đo Entropy (Information Gain)
    dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dt.fit(X_train, y_train)
    
    # 3. Dự đoán và tính lỗi
    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    error = 1 - acc
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Classification Error: {error:.4f}")
    
    # 4. Vẽ cây quyết định (Decision Tree Visualization)
    plt.figure(figsize=(12, 8))
    plot_tree(dt, filled=True, feature_names=data.feature_names, class_names=data.target_names, rounded=True)
    plt.title(f'Decision Tree Structure - {name}')
    plt.show()
    
    # 5. Vẽ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    print("-" * 30 + "\n")