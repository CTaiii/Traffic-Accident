import pickle

def check_files():
    # Kiểm tra model
    try:
        with open('data/model_DDC.pkl', 'rb') as file:
            model_DDC = pickle.load(file)
        print("KMeans model loaded successfully")
    except Exception as e:
        print(f"Error loading KMeans model: {e}")

    # Kiểm tra scaler
    try:
        with open('data/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        print("Scaler loaded successfully")
    except Exception as e:
        print(f"Error loading scaler: {e}")

    # Kiểm tra label_encoders
    try:
        with open('data/label_encoders.pkl', 'rb') as file:
            label_encoders = pickle.load(file)
        print("Label Encoders loaded successfully")
    except Exception as e:
        print(f"Error loading label encoders: {e}")

    # Kiểm tra columns
    try:
        with open('data/columns.pkl', 'rb') as file:
            columns = pickle.load(file)
        print("Columns loaded successfully:", columns)
    except Exception as e:
        print(f"Error loading columns: {e}")

if __name__ == '__main__':
    check_files()