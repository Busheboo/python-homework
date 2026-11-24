"""Обучение модели прогнозирования стоимости квартиры.

Запуск:  python train_model.py realty_data.csv
Результат: файл model.joblib
"""
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error, r2_score)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
FEATURES = ['total_square', 'rooms', 'floor', 'lat', 'lon']


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=FEATURES + ['price'])
    # убираем явные выбросы и ошибки в данных
    df = df[
        (df['price'] > 5e5) & (df['price'] < 1e9)
        & df['total_square'].between(10, 500)
        & df['floor'].between(1, 60)
        & df['rooms'].between(1, 10)
    ]
    return df


def main(path):
    df = load_data(path)
    X = df[FEATURES]
    y = np.log(df['price'])      # учим логарифм цены: цены сильно скошены

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    pred = np.exp(model.predict(X_test))
    true = np.exp(y_test)
    print('Объектов в обучении:', len(X_train), ', в тесте:', len(X_test))
    print('R2:  ', round(r2_score(true, pred), 4))
    print('MAE: ', round(mean_absolute_error(true, pred)), 'руб.')
    print('MAPE:', round(mean_absolute_percentage_error(true, pred), 4))

    joblib.dump(model, 'model.joblib', compress=3)
    print('Модель сохранена в model.joblib')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'realty_data.csv')
