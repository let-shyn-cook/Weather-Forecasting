from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import tensorflow as tf
from utils import preprocess_input, inverse_transform_predictions, get_tavg_sample_data, get_sequence_sample_data

app = Flask(__name__)

# Load các model
tavg_models = {
    'linear': joblib.load('models/tavg/linear_model.pkl'),
    'random_forest': joblib.load('models/tavg/random_forest_model.pkl'),
    'xgboost': joblib.load('models/tavg/xgboost_model.pkl'),
    'svr': joblib.load('models/tavg/svr_model.pkl'),
    'mlp': joblib.load('models/tavg/mlp_model.pkl')
}

sequence_models = {
    'LSTM': tf.keras.models.load_model('models/sequence/lstm_model.keras'),
    'XGBoost': joblib.load('models/sequence/xgb_model.pkl')
}

scaler = joblib.load('models/sequence/scaler.pkl')

@app.route('/')
def home():
    return render_template('tavg_prediction.html')

@app.route('/tavg')
def tavg_prediction():
    return render_template('tavg_prediction.html')

@app.route('/sequence')
def sequence_prediction():
    return render_template('sequence_prediction.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/api/predict_tavg', methods=['POST'])
def predict_tavg():
    data = request.json
    processed_input = preprocess_input(data)
    
    predictions = {}
    for name, model in tavg_models.items():
        pred = model.predict([processed_input])[0]
        predictions[name] = round(float(pred), 2)
    
    return jsonify(predictions)

@app.route('/api/predict_sequence', methods=['POST'])
def predict_sequence():
    data = request.json
    sequence = data['sequence']
    days_to_predict = data['days_to_predict']
    
    # Load models
    models = {
        'LSTM': tf.keras.models.load_model('models/sequence/lstm_model.keras'),
        'XGBoost': joblib.load('models/sequence/xgb_model.pkl')
    }
    
    scaler = joblib.load('models/sequence/scaler.pkl')
    features = ['tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'pres']
    
    # Chuẩn bị dữ liệu đầu vào
    input_sequence = np.array([[day[feature] for feature in features] for day in sequence])
    scaled_sequence = scaler.transform(input_sequence)
    
    # Dự đoán với từng model
    predictions = {}
    for model_name, model in models.items():
        current_sequence = scaled_sequence.copy()
        model_predictions = []
        
        for _ in range(days_to_predict):
            if model_name == 'XGBoost':
                X_flat = current_sequence.reshape(1, -1)
                pred = model.predict(X_flat)
                pred = pred.reshape(1, len(features))
            else:  # LSTM
                X = current_sequence.reshape(1, len(current_sequence), len(features))
                pred = model.predict(X, verbose=0)
                if len(pred.shape) == 3:
                    pred = pred[:, -1:, :]
                pred = pred.reshape(1, len(features))
            
            # Thêm prediction vào chuỗi hiện tại
            current_sequence = np.vstack([current_sequence[1:], pred])
            
            # Inverse transform prediction để có giá trị thực
            pred_original = scaler.inverse_transform(pred)
            prediction_dict = {
                feature: float(pred_original[0][i]) 
                for i, feature in enumerate(features)
            }
            model_predictions.append(prediction_dict)
        
        predictions[model_name] = model_predictions

    return jsonify(predictions)

@app.route('/api/get_tavg_sample')
def get_tavg_sample():
    return jsonify(get_tavg_sample_data())

@app.route('/api/get_sequence_sample')
def get_sequence_sample():
    return jsonify(get_sequence_sample_data())

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',  # Cho phép truy cập từ mọi IP trong mạng LAN
        port=5000,       # Port mặc định
        debug=True       # Tắt debug mode khi chạy production
    ) 