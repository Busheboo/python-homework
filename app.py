import streamlit as st

from predictor import load_model, predict_price

# Координаты центров городов (медиана по данным), чтобы пользователю
# не нужно было вводить координаты вручную.
CITIES = {
    'Москва': (55.741, 37.557),
    'Балашиха': (55.792, 37.969),
    'Видное': (55.544, 37.709),
    'Дзержинский': (55.632, 37.859),
    'Долгопрудный': (55.948, 37.500),
    'Ивантеевка': (55.972, 37.914),
    'Королёв': (55.921, 37.846),
    'Котельники': (55.661, 37.857),
    'Красногорск': (55.822, 37.320),
    'Лобня': (56.010, 37.464),
    'Лыткарино': (55.583, 37.899),
    'Люберцы': (55.695, 37.909),
    'Московский': (55.599, 37.367),
    'Мытищи': (55.921, 37.725),
    'Одинцово': (55.682, 37.288),
    'Подольск': (55.472, 37.564),
    'Пушкино': (56.007, 37.809),
    'Реутов': (55.756, 37.862),
    'Химки': (55.908, 37.423),
    'Щербинка': (55.504, 37.555),
    'Щёлково': (55.925, 37.985),
}


@st.cache_resource
def get_model():
    return load_model('model.joblib')


st.set_page_config(page_title='Оценка стоимости квартиры', page_icon='🏠')
st.title('🏠 Оценка стоимости квартиры')
st.write('Введите параметры квартиры и нажмите кнопку, чтобы получить прогноз цены '
         '(Москва и Московская область).')

model = get_model()

city = st.selectbox('Город', list(CITIES))
default_lat, default_lon = CITIES[city]

col1, col2 = st.columns(2)
with col1:
    total_square = st.number_input('Общая площадь, м²', min_value=10.0,
                                   max_value=500.0, value=60.0, step=1.0)
    rooms = st.slider('Количество комнат', min_value=1, max_value=10, value=2)
    floor = st.number_input('Этаж', min_value=1, max_value=60, value=5, step=1)
with col2:
    # key зависит от города: при смене города координаты сбрасываются на центр города
    lat = st.number_input('Широта', value=default_lat, format='%.4f',
                          key=f'lat_{city}')
    lon = st.number_input('Долгота', value=default_lon, format='%.4f',
                          key=f'lon_{city}')
st.caption('Координаты по умолчанию — центр выбранного города. '
           'Чтобы уточнить район, измените их.')

if st.button('Рассчитать стоимость'):
    price = predict_price(model, total_square, rooms, floor, lat, lon)
    st.success(f'Ориентировочная стоимость: {price:,.0f} ₽'.replace(',', ' '))
    st.caption(f'Средняя ошибка модели на тестовых данных около 11%: '
               f'вероятный диапазон {price * 0.89:,.0f} – {price * 1.11:,.0f} ₽'
               .replace(',', ' '))
