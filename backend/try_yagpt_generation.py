import os
from dotenv import load_dotenv
import requests

# Загрузка переменных окружения из .env файла
load_dotenv()

# Конфигурация
API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
OAUTH_TOKEN = os.getenv("OAUTH_TOKEN")  # Загружаем токен из .env
CATALOG_ID = os.getenv("CATALOG_ID")  # Загружаем ID каталога из .env

# Проверка загрузки переменных
if not OAUTH_TOKEN or not CATALOG_ID:
    raise ValueError("Проверьте ваш .env файл: OAUTH_TOKEN или CATALOG_ID не найдены.")


# Функция для получения IAM-токена из OAuth-токена
def get_iam_token(oauth_token):
    iam_url = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "yandexPassportOauthToken": oauth_token
    }
    response = requests.post(iam_url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["iamToken"]
    else:
        print(f"Ошибка при получении IAM-токена: {response.status_code}")
        print(response.text)
        return None


# Функция для отправки запроса к YandexGPT
def generate_text(prompt):
    # Получаем IAM-токен
    iam_token = get_iam_token(OAUTH_TOKEN)
    if not iam_token:
        raise ValueError("Не удалось получить IAM-токен.")

    headers = {
        "Authorization": f"Bearer {iam_token}",
        "Content-Type": "application/json"
    }

    data = {
        # Используем последнюю модель YandexGPT 5 Pro
                #"modelUri": f"gpt://{CATALOG_ID}/aliceai-llm/latest", # 2 рубля за небольшой запрос
        #"modelUri": f"gpt://{CATALOG_ID}/yandexgpt", # 60 копеек за небольшой запрос
        "modelUri": f"gpt://{CATALOG_ID}/yandexgpt-lite",  # Указываем модель # 6 копеек за небольшой запрос
        "completionOptions": {
            "maxTokens": 1000,  # Максимальное количество токенов в ответе
            "temperature": 0.7,  # Температура для управления случайностью
            "stream": False  # Отключаем потоковую передачу
        },
        "messages": [
            {
                "role": "user",
                "text": prompt  # Ваш запрос к модели
            }
        ]
    }

    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        return result['result']['alternatives'][0]['message']['text']  # Извлекаем текст ответа
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.text)
        return None


# Пример использования
if __name__ == "__main__":
    user_prompt = "Как следует выполнять гидравлический расчет канализационных сетей?"
    response = generate_text(user_prompt)
    if response:
        print("Ответ модели:")
        print(response)