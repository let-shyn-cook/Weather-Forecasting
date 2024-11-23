import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json

# Tạo thư mục models
if not os.path.exists('models/tavg'):
    os.makedirs('models/tavg')

# Đọc và xử lý dữ liệu
df = pd.read_csv('data_thoi_tiet.csv')
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

features = ['tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'month', 'day']
target = 'tavg'

X = df[features]
y = df[target]

# Xử lý missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=features)
y = pd.Series(imputer.fit_transform(y.values.reshape(-1, 1)).ravel())

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo dictionary để lưu metrics
model_metrics = {}

# Train và lưu các mô hình
models = {
    'linear': LinearRegression(),
    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'xgboost': xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        reg_lambda=1,
        reg_alpha=0.5,
        random_state=42
    ),
    'svr': SVR(
        kernel='rbf',
        C=10,
        gamma='scale',
        epsilon=0.1
    ),
    'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
}

for name, model in models.items():
    print(f'Training {name} model...')
    model.fit(X_train, y_train)
    
    # Lưu model
    joblib.dump(model, f'models/tavg/{name}_model.pkl')
    
    # Tính toán và lưu metrics
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'train_score': model.score(X_train, y_train),
        'test_score': model.score(X_test, y_test),
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    model_metrics[name] = metrics
    print(f'{name} - Train score: {metrics["train_score"]:.4f}, Test score: {metrics["test_score"]:.4f}')

# Lưu metrics vào file JSON
with open('models/tavg/model_metrics.json', 'w') as f:
    json.dump(model_metrics, f, indent=4) 