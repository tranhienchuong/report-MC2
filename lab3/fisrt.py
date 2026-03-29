# 1. Import các thư viện cần thiết
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Tải dữ liệu (Chọn 1 trong 2 dòng dưới)
# Dữ liệu 1: Hoa Diên Vĩ (Iris) - Dễ, ít dữ liệu
data = load_iris()

# Dữ liệu 2: Ung thư vú (Breast Cancer) - Khó hơn, nhiều dữ liệu hơn
# data = load_breast_cancer() 

# 3. Tách lấy X (Dữ liệu đầu vào) và y (Nhãn/Đáp án)
X = data.data    # Đây là các số đo (Features)
y = data.target  # Đây là tên loài hoa/bệnh (Labels)

# 4. Chia dữ liệu thành Train (để học) và Test (để thi)
# test_size=0.3 nghĩa là: Lấy 70% để học (Train), 30% để thi (Test)
# random_state=42: Giúp kết quả chia lần nào cũng giống nhau (quan trọng để viết báo cáo)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- ĐẾN ĐÂY LÀ BẠN ĐÃ CÓ X_train, y_train ĐỂ DÙNG CHO ĐOẠN CODE CŨ ---

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 2. Dự đoán
y_pred = knn.predict(X_test)

# 3. Tính toán Error Rate
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
print(f"Accuracy: {accuracy:.4f}")
print(f"Error Rate: {error_rate:.4f}")

# 4. In báo cáo chi tiết (Precision, Recall...)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. Vẽ Confusion Matrix (Lưu ảnh này lại để chèn vào báo cáo)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (k={k})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()