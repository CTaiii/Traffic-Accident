from flask import Flask, render_template, request
from function import read_excel, analyze_frequency, cluster_and_save, predict_damages, predict_reason, predict_cluster
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
def predict_cluster_route():
    input_data = {
        'Gio': request.form.get('Gio'),
        'Duong': request.form.get('Duong'),
        'Quan': request.form.get('Quan'),
        'Tuoi': int(request.form.get('Tuoi')),
        'NguyenNhan': request.form.get('NguyenNhan')
    }
    try:
        cluster = predict_cluster(input_data)
        return render_template('index.html', prediction_cluster=f"Cụm dự đoán: {cluster + 1}")
    except Exception as e:
        return render_template('index.html', error="Đã xảy ra lỗi: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
