from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from predictor import load_model, predict_price

app = FastAPI(
    title='API прогнозирования стоимости квартиры',
    description='Москва и Московская область. Модель: градиентный бустинг '
                '(площадь, комнаты, этаж, координаты).',
    version='1.0.0',
)

model = load_model('model.joblib')

# Допустимые диапазоны взяты из обучающих данных
LAT_MIN, LAT_MAX = 55.4, 56.1
LON_MIN, LON_MAX = 37.1, 38.2


class Apartment(BaseModel):
    """Параметры квартиры для post-запроса."""
    total_square: float = Field(..., ge=10, le=500, description='Общая площадь, м²',
                                examples=[60])
    rooms: int = Field(..., ge=1, le=10, description='Количество комнат',
                       examples=[2])
    floor: int = Field(..., ge=1, le=60, description='Этаж', examples=[5])
    lat: float = Field(55.741, ge=LAT_MIN, le=LAT_MAX,
                       description='Широта (по умолчанию центр Москвы)')
    lon: float = Field(37.557, ge=LON_MIN, le=LON_MAX,
                       description='Долгота (по умолчанию центр Москвы)')


class PriceResponse(BaseModel):
    price: float = Field(..., description='Прогноз стоимости, руб.')


@app.get('/health', summary='Liveness-проба')
def health():
    """Проверка, что сервис запущен."""
    return {'status': 'ok'}


@app.get('/predict_get', response_model=PriceResponse,
         summary='Предсказание через GET')
def predict_get(
    total_square: float = Query(..., ge=10, le=500, description='Общая площадь, м²'),
    rooms: int = Query(..., ge=1, le=10, description='Количество комнат'),
    floor: int = Query(..., ge=1, le=60, description='Этаж'),
    lat: float = Query(55.741, ge=LAT_MIN, le=LAT_MAX, description='Широта'),
    lon: float = Query(37.557, ge=LON_MIN, le=LON_MAX, description='Долгота'),
):
    price = predict_price(model, total_square, rooms, floor, lat, lon)
    return PriceResponse(price=round(price))


@app.post('/predict_post', response_model=PriceResponse,
          summary='Предсказание через POST')
def predict_post(apartment: Apartment):
    price = predict_price(model, apartment.total_square, apartment.rooms,
                          apartment.floor, apartment.lat, apartment.lon)
    return PriceResponse(price=round(price))
