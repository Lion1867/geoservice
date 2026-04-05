"""
Yandex GPT и Yandex Embeddings клиенты
"""

import requests
import time
from typing import List, Dict

from config import (
    OAUTH_TOKEN, CATALOG_ID,
    YANDEX_LLM_URL, YANDEX_LLM_MODEL,
    YANDEX_EMB_URL, YANDEX_EMB_MODEL_DOC, YANDEX_EMB_MODEL_QUERY,
    LLM_TEMPERATURE, LLM_MAX_TOKENS
)


class YandexAuth:
    """Управление IAM токеном Яндекса."""
    
    def __init__(self, oauth_token: str):
        self.oauth_token = oauth_token
        self._iam_token = None
        self._iam_expires = 0
    
    def get_iam_token(self) -> str:
        """Получает IAM токен (кэширует на время жизни)."""
        # IAM токен живёт 12 часов, обновляем за час до истечения
        if self._iam_token and time.time() < self._iam_expires - 3600:
            return self._iam_token
        
        url = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
        data = {"yandexPassportOauthToken": self.oauth_token}
        
        resp = requests.post(url, json=data, timeout=30)
        
        if resp.status_code != 200:
            raise RuntimeError(f"Ошибка получения IAM токена: {resp.status_code} - {resp.text}")
        
        result = resp.json()
        self._iam_token = result["iamToken"]
        # IAM токен живёт 12 часов
        self._iam_expires = time.time() + 12 * 3600
        
        return self._iam_token


# Глобальный объект авторизации
_auth = None

def get_auth() -> YandexAuth:
    global _auth
    if _auth is None:
        if not OAUTH_TOKEN:
            raise ValueError("OAUTH_TOKEN не задан в .env")
        _auth = YandexAuth(OAUTH_TOKEN)
    return _auth


# =====================================================
# YANDEX EMBEDDINGS
# =====================================================
class YandexEmbedder:
    """
    Эмбеддер через Yandex Foundation Models API.
    
    Использует разные модели для документов и запросов:
    - text-search-doc: для индексации документов
    - text-search-query: для поисковых запросов
    """
    
    def __init__(self):
        self.auth = get_auth()
        self.catalog_id = CATALOG_ID
        self.dimension = 256  # Yandex text-search модели возвращают 256-мерные векторы
        print(f"🔗 Yandex Embedder | catalog: {self.catalog_id}")
    
    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.auth.get_iam_token()}",
            "Content-Type": "application/json"
        }
    
    def _embed_single(self, text: str, model_type: str = "doc") -> List[float]:
        """Получает эмбеддинг для одного текста."""
        model_name = YANDEX_EMB_MODEL_DOC if model_type == "doc" else YANDEX_EMB_MODEL_QUERY
        model_uri = f"emb://{self.catalog_id}/{model_name}/latest"
        
        data = {
            "modelUri": model_uri,
            "text": text
        }
        
        resp = requests.post(
            YANDEX_EMB_URL,
            headers=self._get_headers(),
            json=data,
            timeout=60
        )
        
        if resp.status_code != 200:
            raise RuntimeError(f"Yandex Embedding error: {resp.status_code} - {resp.text[:300]}")
        
        result = resp.json()
        return result["embedding"]
    
    def embed(self, texts: List[str], model_type: str = "doc") -> List[List[float]]:
        """
        Получает эмбеддинги для списка текстов.
        
        Args:
            texts: список текстов
            model_type: "doc" для документов, "query" для запросов
        """
        embeddings = []
        for text in texts:
            # Ограничение на длину текста (Yandex ~10000 токенов)
            truncated = text[:8000] if len(text) > 8000 else text
            emb = self._embed_single(truncated, model_type)
            embeddings.append(emb)
        return embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Эмбеддинги для документов (индексация)."""
        return self.embed(texts, model_type="doc")
    
    def embed_query(self, text: str) -> List[float]:
        """Эмбеддинг для поискового запроса."""
        return self._embed_single(text, model_type="query")
    
    def embed_single(self, text: str) -> List[float]:
        """Алиас для совместимости (использует query модель)."""
        return self.embed_query(text)
    
    def get_dimension(self) -> int:
        return self.dimension


# =====================================================
# YANDEX LLM
# =====================================================
class YandexLLM:
    """Клиент для Yandex GPT."""
    
    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        self.auth = get_auth()
        self.catalog_id = CATALOG_ID
        self.model = model or YANDEX_LLM_MODEL
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.max_tokens = max_tokens or LLM_MAX_TOKENS
        
        self.model_uri = f"gpt://{self.catalog_id}/{self.model}/latest"
        print(f"🤖 Yandex LLM: {self.model}")
    
    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.auth.get_iam_token()}",
            "Content-Type": "application/json"
        }
    
    def generate(self, messages: List[Dict]) -> str:
        """
        Генерация ответа.
        
        Args:
            messages: список сообщений [{"role": "user/system/assistant", "content": "..."}]
        """
        # Конвертируем формат сообщений (Yandex использует "text" вместо "content")
        yandex_messages = []
        for msg in messages:
            yandex_messages.append({
                "role": msg["role"],
                "text": msg.get("content") or msg.get("text", "")
            })
        
        data = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False
            },
            "messages": yandex_messages
        }
        
        try:
            resp = requests.post(
                YANDEX_LLM_URL,
                headers=self._get_headers(),
                json=data,
                timeout=120
            )
        except requests.exceptions.RequestException as e:
            return f"[Ошибка подключения к Yandex GPT] {e}"
        
        if resp.status_code != 200:
            return f"[Yandex GPT ошибка {resp.status_code}] {resp.text[:300]}"
        
        result = resp.json()
        return result["result"]["alternatives"][0]["message"]["text"]
    
    def ask_with_context(self, question: str, context_chunks: List[Dict]) -> str:
        """Формирует промпт с контекстом и генерирует ответ."""
        
        system_prompt = """Ты — интеллектуальный ассистент по документам.
Отвечай на вопросы, используя ТОЛЬКО предоставленный контекст.

Правила:
1. Опирайся только на факты из контекста.
2. Если ответа нет — скажи: «В документах ответ не найден».
3. Указывай источник информации.
4. Отвечай развёрнуто, но по делу."""

        # Собираем контекст
        parts = []
        for i, ch in enumerate(context_chunks, 1):
            source = ch.get("source", "неизвестно")
            score = ch.get("score", 0)
            parts.append(f"[Документ: {source}, релевантность: {score:.3f}]\n{ch['text']}")
        
        context_str = "\n\n---\n\n".join(parts)
        
        user_message = f"""Контекст из документов:

{context_str}

---

Вопрос: {question}

Ответ:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return self.generate(messages)


# =====================================================
# ФАБРИКИ ДЛЯ СОВМЕСТИМОСТИ
# =====================================================
def create_embedder(**kwargs) -> YandexEmbedder:
    """Создаёт Yandex Embedder."""
    return YandexEmbedder()


def create_llm(**kwargs) -> YandexLLM:
    """Создаёт Yandex LLM."""
    return YandexLLM()