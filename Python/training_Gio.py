import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from imblearn.under_sampling import RandomUnderSampler

# Đọc dữ liệu từ file Excel
file_path = 'data/All_MH.xlsx'  # Cập nhật đường dẫn đúng nếu cần
data = pd.read_excel(file_path)

# Loại bỏ các dòng có giá trị 'X' hoặc rỗng trong cột 'Gio'
data = data[(data['Gio'] != 'X') & (data['Gio'].notnull())]

# Giả sử bạn muốn dự đoán cột 'Gio'
X = data.drop(columns=['Gio'])  # Thay đổi cột dự đoán
y = data['Gio']

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

# Lưu mô hình vào file
with open('data/model_Gio.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Mô hình đã được lưu vào file 'model_Gio.pkl'")

# Lưu tên cột của X sau khi đã thực hiện get_dummies
feature_columns = X.columns.tolist()  # Lưu trữ tên các cột cho dự đoán

