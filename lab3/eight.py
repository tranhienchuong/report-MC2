from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

datasets = [
    ("Dataset 1 (Iris)", load_iris()),
    ("Dataset 2 (Breast Cancer)", load_breast_cancer())
]

for name, data in datasets:
    print(f"--- TUNING PARAMETERS FOR: {name} ---")
    
    # 1. Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
    
    # 2. Chuẩn hóa (Rất quan trọng với SVM)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 3. Thiết lập lưới tham số để thử nghiệm
    # C: Trade-off giữa margin rộng và phân loại đúng.
    # gamma: Độ ảnh hưởng của từng điểm dữ liệu (chỉ dùng cho kernel RBF/Poly).
    # kernel: Loại hàm biến đổi không gian.
    param_grid = [
        # Trường hợp 1: Dùng Linear Kernel (Chỉ cần chỉnh C)
        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
        
        # Trường hợp 2: Dùng RBF Kernel (Cần chỉnh cả C và gamma)
        {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
    ]
    
    # 4. Chạy Grid Search (Tìm kiếm lưới)
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5)
    grid.fit(X_train, y_train)
    
    # 5. In kết quả tốt nhất
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Best Cross-Validation Score: {grid.best_score_:.4f}")
    print("-" * 30 + "\n")