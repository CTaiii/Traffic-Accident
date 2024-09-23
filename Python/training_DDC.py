import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering
import pickle

# Đọc dữ liệu từ Excel
file_path = 'data/All.xlsx'
df = pd.read_excel(file_path)

# Xử lý dữ liệu
df_cleaned = df.dropna()

# Khởi tạo LabelEncoder cho từng cột
label_encoders = {}
for column in df_cleaned.columns:
    if df_cleaned[column].dtype == 'object':
        encoder = LabelEncoder()
        df_cleaned[column] = encoder.fit_transform(df_cleaned[column])
        label_encoders[column] = encoder

# Tiền xử lý dữ liệu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_cleaned)

# Huấn luyện mô hình phân cụm
model_DDC = AgglomerativeClustering(n_clusters=3)
model_DDC.fit(scaled_data)

# Lưu mô hình và các công cụ tiền xử lý vào file
with open('data/model_DDC.pkl', 'wb') as file:
    pickle.dump(model_DDC, file)
with open('data/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
with open('data/label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

print("Mô hình phân cụm và các công cụ tiền xử lý đã được lưu")
