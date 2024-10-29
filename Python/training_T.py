import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

# Đọc dữ liệu từ file Excel trên máy cục bộ
file_path = 'data/All_MH.xlsx'  # Đảm bảo cập nhật đường dẫn đúng
data = pd.read_excel(file_path)

# Dữ liệu đầu vào và nhãn cần dự đoán
X = data.drop(columns=['Tuoi'])  # Thay đổi 'Tuoi' thành cột dự đoán
y = data['Tuoi']

# Chuyển đổi các biến phân loại thành số
X = pd.get_dummies(X)

# Cân bằng dữ liệu bằng Undersampling
under_sampler = RandomUnderSampler()
X_resampled, y_resampled = under_sampler.fit_resample(X, y)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Lưu mô hình
import pickle
with open('data/model_Tuoi.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Mô hình đã được lưu vào file 'model_Tuoi.pkl'")