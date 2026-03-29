from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình tham số tốt nhất (Giả định từ bước II.2)
# Bạn có thể thay đổi C hoặc gamma nếu kết quả GridSearch của bạn khác
best_configs = [
    {
        "name": "Dataset 1 (Iris)",
        "data": load_iris(),
        "params": {'kernel': 'linear', 'C': 1} # Iris thường tốt nhất với Linear
    },
    {
        "name": "Dataset 2 (Breast Cancer)",
        "data": load_breast_cancer(),
        "params": {'kernel': 'rbf', 'C': 10, 'gamma': 0.1} # Cancer cần RBF
    }
]

for config in best_configs:
    name = config['name']
    data = config['data']
    params = config['params']
    
    print(f"--- EVALUATING SVM FOR: {name} ---")
    print(f"Parameters used: {params}")
    
    # 1. Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
    
    # 2. Chuẩn hóa (Bắt buộc cho SVM)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 3. Huấn luyện SVM
    # **params: Tự động điền các tham số kernel, C, gamma vào hàm SVC
    svm = SVC(**params)
    svm.fit(X_train, y_train)
    
    # 4. Đánh giá
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    error = 1 - acc
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Classification Error: {error:.4f}")
    
    # 5. Vẽ Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title(f'SVM Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    print("-" * 30 + "\n")