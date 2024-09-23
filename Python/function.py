
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

from training_DDC import label_encoders


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


def preprocess_input(input_data):
    """Tiền xử lý dữ liệu đầu vào để phù hợp với mô hình phân cụm"""
    damage_mapping = {'Nhẹ': 1, 'Nặng': 2, 'Rất nặng': 3}
    df = pd.DataFrame([input_data])

    # Chuyển đổi các nhãn thành mã số dựa trên encoder
    for column in df.columns:
        if df[column].dtype == 'object':
            if column in label_encoders:
                df[column] = label_encoders[column].transform(df[column])
            else:
                # Nếu chưa có encoder cho cột này, tạo mới
                encoder = LabelEncoder()
                df[column] = encoder.fit_transform(df[column])
                label_encoders[column] = encoder

    # Chuyển đổi cột 'ThietHai' nếu cần
    if 'ThietHai' in df.columns:
        df['ThietHai'] = df['ThietHai'].map(damage_mapping)

    return df


def predict_cluster(input_data):
    """Dự đoán cụm dựa trên dữ liệu đầu vào"""
    # Tải các công cụ đã lưu
    with open('data/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('data/label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)
    with open('data/model_DDC.pkl', 'rb') as file:
        model_DDC = pickle.load(file)

    # Tiền xử lý dữ liệu đầu vào
    input_df = preprocess_input(input_data)

    # Đảm bảo dữ liệu đầu vào có tất cả các cột giống như dữ liệu huấn luyện
    all_columns = pd.read_excel('data/All.xlsx').columns
    for column in all_columns:
        if column not in input_df.columns:
            input_df[column] = 0

    input_df = input_df.reindex(columns=all_columns, fill_value=0)

    # Tiến hành tiền xử lý dữ liệu
    input_scaled = scaler.transform(input_df)

    # Dự đoán cụm cho dữ liệu đầu vào
    cluster_prediction = model_DDC.fit_predict(input_scaled)
    return cluster_prediction[0]