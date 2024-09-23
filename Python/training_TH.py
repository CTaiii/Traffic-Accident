import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Đọc dữ liệu từ file Excel
file_path = 'data/All_MH.xlsx'  # Cập nhật đường dẫn đúng nếu cần
data = pd.read_excel(file_path)

# Chuẩn bị dữ liệu
X = data.drop(columns=['ThietHai'])
y = data['ThietHai']

# Chuyển đổi các biến phân loại thành số
X = pd.get_dummies(X)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình
model = RandomForestClassifier()

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Lưu mô hình vào file
with open('data/model_TH.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Mô hình đã được lưu vào file 'model_TH.pkl'")
