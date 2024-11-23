import numpy as np

def preprocess_input(data):
    features = ['tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'month', 'day']
    return [float(data[feature]) for feature in features]

def inverse_transform_predictions(predictions, scaler):
    predictions = np.array(predictions)
    inverse_predictions = scaler.inverse_transform(predictions)
    return {
        'tavg': inverse_predictions[:, 0].tolist(),
        'tmin': inverse_predictions[:, 1].tolist(),
        'tmax': inverse_predictions[:, 2].tolist(),
        'prcp': inverse_predictions[:, 3].tolist(),
        'wdir': inverse_predictions[:, 4].tolist(),
        'wspd': inverse_predictions[:, 5].tolist(),
        'pres': inverse_predictions[:, 6].tolist()
    } 

def get_tavg_sample_data():
    return {
        'tmin': 22.5,
        'tmax': 31.2,
        'prcp': 0.5,
        'wdir': 180,
        'wspd': 2.5,
        'month': 6,
        'day': 15
    }

def get_sequence_sample_data():
    return {
        'input_data': [
            {
                'tavg': 26.5,
                'tmin': 22.0,
                'tmax': 31.0,
                'prcp': 0.0,
                'wdir': 180,
                'wspd': 2.5,
                'pres': 1008.5
            },
            {
                'tavg': 27.0,
                'tmin': 22.5,
                'tmax': 31.5,
                'prcp': 0.0,
                'wdir': 185,
                'wspd': 2.8,
                'pres': 1009.0
            },
            {
                'tavg': 27.2,
                'tmin': 22.8,
                'tmax': 31.8,
                'prcp': 0.0,
                'wdir': 175,
                'wspd': 2.6,
                'pres': 1008.8
            },
            {
                'tavg': 26.8,
                'tmin': 22.2,
                'tmax': 31.2,
                'prcp': 1.5,
                'wdir': 190,
                'wspd': 3.0,
                'pres': 1007.5
            },
            {
                'tavg': 26.0,
                'tmin': 21.5,
                'tmax': 30.5,
                'prcp': 2.5,
                'wdir': 195,
                'wspd': 3.2,
                'pres': 1007.0
            },
            {
                'tavg': 26.2,
                'tmin': 21.8,
                'tmax': 30.8,
                'prcp': 0.5,
                'wdir': 185,
                'wspd': 2.9,
                'pres': 1008.0
            },
            {
                'tavg': 26.5,
                'tmin': 22.0,
                'tmax': 31.0,
                'prcp': 0.0,
                'wdir': 180,
                'wspd': 2.7,
                'pres': 1008.2
            }
        ],
        'days_to_predict': 3
    } 