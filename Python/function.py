
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.preprocessing import LabelEncoder
import pickle
from training_DDC import label_encoders, df_cleaned


# Đọc dữ liệu từ file Excel
def read_excel(file_path):
    """Đọc dữ liệu từ file Excel và trả về một DataFrame"""
    df = pd.read_excel(file_path)
    if df.empty:
        raise ValueError("DataFrame is empty. Please provide a valid Excel file.")
    return df


def analyze_frequency(df):
    """Phân tích tần suất của cặp 'Nguyên nhân' và 'Đường'"""
    if 'Nguyên nhân' not in df.columns or 'Đường' not in df.columns:
        raise ValueError("DataFrame does not contain required columns 'Nguyên nhân' and 'Đường'")

    frequency = df.groupby(['Nguyên nhân', 'Đường']).size().reset_index(name='Tần suất')
    total_by_nguyen_nhan = df['Nguyên nhân'].value_counts().reset_index()
    total_by_nguyen_nhan.columns = ['Nguyên nhân', 'Tổng số lần xuất hiện']
    frequency = pd.merge(frequency, total_by_nguyen_nhan, on='Nguyên nhân')
    frequency['Tần suất (%)'] = (frequency['Tần suất'] / frequency['Tổng số lần xuất hiện']) * 100
    top_10_frequency = frequency.sort_values(by='Tần suất', ascending=False).head(10)

    return top_10_frequency, None


def cluster_and_save(file_path, output_file_path, n_clusters=3):
    """Thực hiện gom cụm dữ liệu và lưu kết quả vào file Excel"""
    df = pd.read_excel(file_path)
    df_cleaned = df.dropna()
    label_encoder = LabelEncoder()
    for column in df_cleaned.columns:
        df_cleaned[column] = label_encoder.fit_transform(df_cleaned[column])

    cluster_model = AgglomerativeClustering(n_clusters=n_clusters)
    df_cleaned['Cluster'] = cluster_model.fit_predict(df_cleaned)

    silhouette_avg = silhouette_score(df_cleaned.drop(columns=['Cluster']), df_cleaned['Cluster'])
    print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg}")

    df_cleaned.to_excel(output_file_path, index=False)

    # Collect cluster info for display
    cluster_info = {}
    for cluster_id in range(n_clusters):
        cluster_data = df_cleaned[df_cleaned['Cluster'] == cluster_id]
        cluster_info[f'Cụm {cluster_id + 1}'] = {
            'data': cluster_data.head().to_dict(orient='records'),
            'count': len(cluster_data)
        }

    # Vẽ biểu đồ phân tán của các cụm và lưu vào bộ nhớ
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df_cleaned, x=df_cleaned.columns[0], y=df_cleaned.columns[1], hue='Cluster', palette='viridis',
                    s=100, ax=ax)
    ax.set_xlabel(df_cleaned.columns[0])
    ax.set_ylabel(df_cleaned.columns[1])
    ax.set_title(f'Biểu Đồ Phân Tán Các Cụm (Số cụm: {n_clusters})')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig)

    return cluster_info, img_base64


# Tải mô hình đã huấn luyện từ file
with open('data/model_TH.pkl', 'rb') as file:
    model_TH = pickle.load(file)


def predict_damages(input_data):
    """Dự đoán thiệt hại dựa trên dữ liệu đầu vào"""
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_TH.feature_names_in_, fill_value=0)
    prediction = model_TH.predict(input_df)
    return prediction[0]


with open('data/model_NN.pkl', 'rb') as file:
    model_NN = pickle.load(file)


def predict_reason(input_data):
    """Dự đoán nguyên nhân dựa trên dữ liệu đầu vào"""
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_NN.feature_names_in_, fill_value=0)
    prediction = model_NN.predict(input_df)
    return prediction[0]


# Đọc mô hình và các công cụ từ file
with open('data/model_DDC.pkl', 'rb') as file:
    model_DDC = pickle.load(file)
with open('data/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('data/label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# Lưu thông tin cột vào file
with open('data/columns.pkl', 'wb') as file:
    pickle.dump(df_cleaned.columns.tolist(), file)

# Lưu thông tin cột vào biến toàn cục
columns = df_cleaned.columns.tolist()  # Đảm bảo biến columns chứa danh sách các cột

# Hàm tiền xử lý dữ liệu
def preprocess_input(input_data, encoder_dict, scaler):
    # Chuyển đổi dữ liệu đầu vào thành DataFrame
    input_df = pd.DataFrame([input_data])

    # Sử dụng LabelEncoder để chuyển đổi các giá trị phân loại
    for column, encoder in encoder_dict.items():
        input_df[column] = encoder.transform(input_df[column])

    # Nếu cần thiết, thêm bước xử lý cho các cột còn lại, ví dụ:
    # input_df['Gio'] = input_df['Gio'].map({'Sáng': 0, 'Trưa': 1, 'Chiều': 2, 'Tối': 3, 'Khuya': 4})

    # Chọn các cột cần thiết
    # input_df = input_df[['Duong', 'Gio', 'Quan', 'Tuoi', 'ThietHai', 'NguyenNhan']]

    # Chuẩn hóa dữ liệu
    input_scaled = scaler.transform(input_df)

    return input_df