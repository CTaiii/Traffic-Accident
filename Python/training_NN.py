import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
import pickle

# Đọc dữ liệu từ file Excel
file_path = 'data/All_MH.xlsx'  # Cập nhật đường dẫn đúng nếu cần
data = pd.read_excel(file_path)

# Chuẩn bị dữ liệu
X = data.drop(columns=['NguyenNhan'])  # Cột dự đoán
y = data['NguyenNhan']

# Chuyển đổi các biến phân loại thành số
X = pd.get_dummies(X)

# Cân bằng dữ liệu bằng Undersampling
under_sampler = RandomUnderSampler()
X_resampled, y_resampled = under_sampler.fit_resample(X, y)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Khởi tạo mô hình Logistic Regression
model = LogisticRegression(max_iter=1000)  # Tăng số lần lặp nếu cần

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Lưu mô hình vào file
with open('data/model_NN.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Mô hình dự đoán nguyên nhân đã được lưu vào file 'model_NN.pkl'")
