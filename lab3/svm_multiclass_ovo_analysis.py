from sklearn.datasets import load_iris
from sklearn.svm import SVC

# Tải dữ liệu Iris (3 lớp)
iris = load_iris()
X, y = iris.data, iris.target

print(f"--- MULTI-CLASS SVM ANALYSIS (IRIS) ---")
print(f"Number of classes: {len(iris.target_names)} ({iris.target_names})")

# Huấn luyện SVM với chế độ mặc định (decision_function_shape='ovr' là output, nhưng nội tại là ovo)
svm = SVC(kernel='linear', decision_function_shape='ovr')
svm.fit(X, y)

# Kiểm tra số lượng bộ phân loại con (Binary Classifiers)
# intercept_ sẽ có kích thước bằng số lượng bộ phân loại
num_classifiers = svm.intercept_.shape[0]

print(f"Shape of intercept_ matrix: {svm.intercept_.shape}")
print(f"Number of internal binary classifiers constructed: {num_classifiers}")

# Tính toán lý thuyết: One-vs-One cần N*(N-1)/2
n = 3
theoretical_ovo = int(n * (n - 1) / 2)
print(f"Theoretical classifiers needed for One-vs-One (3 classes): {theoretical_ovo}")

if num_classifiers == theoretical_ovo:
    print("=> CONCLUSION: The model uses 'One-vs-One' strategy.")
else:
    print("=> CONCLUSION: The model uses 'One-vs-Rest' strategy.")