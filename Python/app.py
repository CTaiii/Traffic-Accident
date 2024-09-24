from flask import Flask, render_template, request
from function import read_excel, analyze_frequency, cluster_and_save, predict_damages, predict_reason, preprocess_input
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'analyze_data' in request.form:
        file_path = './data/DDxNN.xlsx'
        try:
            df = read_excel(file_path)
            top_10_frequency, _ = analyze_frequency(df)
            result_dict = top_10_frequency.to_dict(orient='records')
            return render_template('index.html', result=result_dict)
        except ValueError as e:
            return render_template('index.html', error=str(e))
        except Exception as e:
            return render_template('index.html', error="Đã xảy ra lỗi: " + str(e))

@app.route('/cluster', methods=['POST'])
def cluster():
    if 'cluster_data' in request.form:
        input_file_path = './data/DDxNN.xlsx'
        output_file_path = './data/DDxNN_MaHoa.xlsx'
        try:
            cluster_info, plot_img = cluster_and_save(input_file_path, output_file_path)
            return render_template('index.html', cluster_info=cluster_info, plot_img=plot_img, message="Đã thực hiện gom cụm và lưu kết quả.")
        except Exception as e:
            return render_template('index.html', error="Đã xảy ra lỗi: " + str(e))

@app.route('/predict_TH', methods=['POST'])
def predict_TH():
    input_data = {
        'Gio': request.form.get('Gio'),
        'Duong': request.form.get('Duong'),
        'Quan': request.form.get('Quan'),
        'Tuoi': int(request.form.get('Tuoi')),
        'NguyenNhan': request.form.get('NguyenNhan')
    }
    try:
        prediction = predict_damages(input_data)
        return render_template('index.html', prediction_TH=prediction)
    except Exception as e:
        return render_template('index.html', error="Đã xảy ra lỗi: " + str(e))

@app.route('/predict_NN', methods=['POST'])
def predict_NN():
    input_data = {
        'Gio': request.form.get('Gio'),
        'Duong': request.form.get('Duong'),
        'Quan': request.form.get('Quan'),
        'Tuoi': int(request.form.get('Tuoi')),
        'NguyenNhan': request.form.get('NguyenNhan')
    }
    try:
        prediction = predict_reason(input_data)
        return render_template('index.html', prediction_NN=prediction)
    except Exception as e:
        return render_template('index.html', error="Đã xảy ra lỗi: " + str(e))

@app.route('/predict_cluster', methods=['POST'])
def predict_cluster():
    try:
        # Lấy dữ liệu từ form
        input_data = {
            'Gio': request.form.get('Gio'),
            'Duong': request.form.get('Duong'),
            'Quan': request.form.get('Quan'),
            'Tuoi': int(request.form.get('Tuoi')),
            'ThietHai': request.form.get('ThietHai'),
            'NguyenNhan': request.form.get('NguyenNhan')
        }

        # In dữ liệu nhận được từ form
        print("Data received from HTML form:")
        for key, value in input_data.items():
            print(f"{key}: {value}")

        # Tiền xử lý dữ liệu
        input_df = preprocess_input(input_data, encoder_dict, scaler)

        # Chuyển đổi dữ liệu thành định dạng mà mô hình dự đoán yêu cầu
        input_scaled = scaler.transform(input_df)

        # Dự đoán cụm
        cluster = model_DDC.predict(input_scaled)

        return f"Dữ liệu thuộc cụm: {cluster[0]}"

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return render_template('index.html', error="Đã xảy ra lỗi: " + str(e))

if __name__ == '__main__':
    # Tải các encoder, scaler và mô hình từ file
    with open('data/label_encoders.pkl', 'rb') as f:
        encoder_dict = pickle.load(f)

    with open('data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('data/model_DDC.pkl', 'rb') as f:
        model_DDC = pickle.load(f)

    with open('data/columns.pkl', 'rb') as file:
        columns = pickle.load(file)

    app.run(debug=True)