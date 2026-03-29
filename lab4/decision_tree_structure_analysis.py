from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Chạy cho cả 2 dataset
datasets = [
    ("Dataset 1 (Iris)", load_iris()),
    ("Dataset 2 (Breast Cancer)", load_breast_cancer())
]

for name, data in datasets:
    # Train lại model nhanh gọn
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dt.fit(X_train, y_train)

    # 1. Lấy tên đặc trưng ở Root Node (Nút gốc)
    # dt.tree_.feature[0] là chỉ số của feature tại nút gốc
    root_feature = data.feature_names[dt.tree_.feature[0]]
    
    # 2. Lấy độ sâu của cây
    depth = dt.get_depth()

    print(f"--- {name} ---")
    print(f"[INSERT FEATURE NAME] = {root_feature}")
    print(f"[INSERT DEPTH]        = {depth}")
    print("-" * 30)