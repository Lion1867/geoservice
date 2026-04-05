'''import folium

# Рабочий URL с подписями
m = folium.Map(
    location=[48.0158, 37.8028],  # Донецк
    zoom_start=14,
    tiles='https://core-renderer-tiles.maps.yandex.net/tiles?l=map&v=21.12.01-0&x={x}&y={y}&z={z}&scale=1&lang=ru_RU',
    attr='Яндекс Карты'
)

m.save('yandex_working.html')
print("Карта сохранена")'''

import folium
import os
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

# Получаем API ключ из переменных окружения
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')

if not YANDEX_API_KEY:
    raise ValueError("API ключ не найден! Создайте файл .env с YANDEX_API_KEY=ваш_ключ")

# Координаты центра Мариуполя
CENTER_LAT, CENTER_LON = 47.0951, 37.5415

# Создаем карту с Яндекс.Картами
m = folium.Map(
    location=[CENTER_LAT, CENTER_LON],
    zoom_start=14,
    tiles=f'https://core-renderer-tiles.maps.yandex.net/tiles?l=map&v=21.12.01-0&x={{x}}&y={{y}}&z={{z}}&scale=1&lang=ru_RU&apikey={YANDEX_API_KEY}',
    attr='Яндекс Карты © 2026 | <a href="https://yandex.ru/legal/maps_api/" target="_blank">Условия использования</a>'
)

# Сохраняем карту
m.save('mariupol_map.html')
print("Карта Мариуполя с API сохранена")