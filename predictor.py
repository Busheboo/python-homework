"""Получение предсказания обученной моделью."""
import joblib
import numpy as np
import pandas as pd

FEATURES = ['total_square', 'rooms', 'floor', 'lat', 'lon']


def load_model(path='model.joblib'):
    return joblib.load(path)


def predict_price(model, total_square, rooms, floor, lat, lon):
    """Возвращает предсказанную стоимость квартиры в рублях."""
    X = pd.DataFrame(
        [[total_square, rooms, floor, lat, lon]], columns=FEATURES
    )
    # модель обучена на логарифме цены, поэтому берём экспоненту
    return float(np.exp(model.predict(X)[0]))
