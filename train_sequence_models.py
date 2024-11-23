import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU, SimpleRNN, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json
import tensorflow as tf
# Thiết lập seed cho reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Tạo thư mục models/sequence nếu chưa tồn tại
if not os.path.exists('models/sequence'):
    os.makedirs('models/sequence')

# Thêm EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    mode='min'
)

# Định nghĩa không gian tìm kiếm siêu tham số
param_distributions = {
    'LSTM': {
        'units': [64],
        'learning_rate': [0.001],
        'batch_size': [32],
        'dropout_rate': [0.2]
    }
}

# Đọc và xử lý dữ liệu
df = pd.read_csv('data_thoi_tiet.csv', thousands=',')
features = ['tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'pres']

# Xử lý missing values
df[features] = df[features].fillna(df[features].mean())

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])
joblib.dump(scaler, 'models/sequence/scaler.pkl')

# Chuẩn bị dữ liệu cho sequence prediction
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 7
X, y = create_sequences(scaled_data, seq_length)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Dictionary để lưu metrics
model_metrics = {}

def create_model(model_type, params):
    if model_type == 'LSTM':
        model = Sequential([
            LSTM(units=params['units'], input_shape=(seq_length, len(features)), return_sequences=True),
            BatchNormalization(),
            Dropout(params['dropout_rate']),
            LSTM(units=params['units']//2),
            BatchNormalization(),
            Dropout(params['dropout_rate']),
            Dense(len(features))
        ])
    return model

# Thực hiện tìm kiếm siêu tham số cho các mô hình neural network
best_params = {}
best_models = {}

# Thực hiện cross-validation và huấn luyện
tscv = TimeSeriesSplit(n_splits=5)

for model_name in ['LSTM']:
    print(f'Tuning {model_name} hyperparameters...')
    best_val_loss = float('inf')
    
    # Tạo callbacks cho model hiện tại
    callbacks = [
        early_stopping,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'models/sequence/best_{model_name.lower()}.keras',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    for units in param_distributions[model_name]['units']:
        for lr in param_distributions[model_name]['learning_rate']:
            for batch_size in param_distributions[model_name]['batch_size']:
                for dropout_rate in param_distributions[model_name]['dropout_rate']:
                    params = {
                        'units': units,
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'dropout_rate': dropout_rate
                    }
                    
                    # Huấn luyện mô hình trên toàn bộ tập train
                    model = create_model(model_name, params)
                    history = model.fit(
                        X_train, y_train,
                        epochs=100,
                        batch_size=batch_size,
                        validation_split=0.2,
                        callbacks=callbacks,
                        verbose=1
                    )
                    
                    val_loss = min(history.history['val_loss'])
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params[model_name] = params
                        best_models[model_name] = model

    # Đánh giá và lưu mô hình tốt nhất
    model = best_models[model_name]
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    train_mse = mean_squared_error(y_train.reshape(-1), y_train_pred.reshape(-1))
    train_r2 = r2_score(y_train.reshape(-1), y_train_pred.reshape(-1))
    test_mse = mean_squared_error(y_test.reshape(-1), y_pred.reshape(-1))
    test_r2 = r2_score(y_test.reshape(-1), y_pred.reshape(-1))
    
    model_metrics[model_name] = {
        'train_mse': float(train_mse),
        'train_r2': float(train_r2),
        'test_mse': float(test_mse),
        'test_r2': float(test_r2)
    }
    
    print(f'{model_name}:')
    print(f'Train - MSE: {train_mse:.4f}, R2: {train_r2:.4f}')
    print(f'Test - MSE: {test_mse:.4f}, R2: {test_r2:.4f}')
    
    # Lưu mô hình
    model.save(f'models/sequence/{model_name.lower()}_model.keras')

# Lưu metrics và parameters
with open('models/sequence/model_metrics.json', 'w') as f:
    json.dump(model_metrics, f, indent=4)

with open('models/sequence/best_params.json', 'w') as f:
    json.dump(best_params, f, indent=4)

# Huấn luyện XGBoost
print('Training XGBoost model...')
X_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

xgb_model = MultiOutputRegressor(xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
))
xgb_model.fit(X_flat, y_train)
joblib.dump(xgb_model, 'models/sequence/xgb_model.pkl')

# Đánh giá XGBoost
y_pred = xgb_model.predict(X_test_flat)
mse = mean_squared_error(y_test.reshape(-1), y_pred.reshape(-1))
r2 = r2_score(y_test.reshape(-1), y_pred.reshape(-1))
model_metrics['XGBoost'] = {
    'test_mse': float(mse),
    'test_r2': float(r2)
}
print(f'XGBoost - Test MSE: {mse:.4f}, Test R2: {r2:.4f}')

# Lưu metrics
with open('models/sequence/model_metrics.json', 'w') as f:
    json.dump(model_metrics, f, indent=4) 

def ensemble_predict(models, X):
    predictions = []
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    return np.mean(predictions, axis=0) 