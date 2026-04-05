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
    def __init__(self, oauth_token: str):
        self.oauth_token = oauth_token
        self._iam_token = None
        self._iam_expires = 0

    def get_iam_token(self) -> str:
        if self._iam_token and time.time() < self._iam_expires - 3600:
            return self._iam_token

        url = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
        data = {"yandexPassportOauthToken": self.oauth_token}

        resp = requests.post(url, json=data, timeout=30)

        if resp.status_code != 200:
            raise RuntimeError(f"IAM token error: {resp.status_code} - {resp.text}")

        result = resp.json()
        self._iam_token = result["iamToken"]
        self._iam_expires = time.time() + 12 * 3600

        return self._iam_token


_auth = None

def get_auth() -> YandexAuth:
    global _auth
    if _auth is None:
        if not OAUTH_TOKEN:
            raise ValueError("OAUTH_TOKEN not set in .env")
        _auth = YandexAuth(OAUTH_TOKEN)
    return _auth


class YandexEmbedder:
    def __init__(self):
        self.auth = get_auth()
        self.catalog_id = CATALOG_ID
        self.dimension = 256
        print(f"Yandex Embedder | catalog: {self.catalog_id}")

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.auth.get_iam_token()}",
            "Content-Type": "application/json"
        }

    def _embed_single(self, text: str, model_type: str = "doc") -> List[float]:
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
        embeddings = []
        for text in texts:
            truncated = text[:8000] if len(text) > 8000 else text
            emb = self._embed_single(truncated, model_type)
            embeddings.append(emb)
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts, model_type="doc")

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text, model_type="query")

    def embed_single(self, text: str) -> List[float]:
        return self.embed_query(text)

    def get_dimension(self) -> int:
        return self.dimension


class YandexLLM:
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
        print(f"Yandex LLM: {self.model}")

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.auth.get_iam_token()}",
            "Content-Type": "application/json"
        }

    def generate(self, messages: List[Dict]) -> str:
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
            return f"[Connection error to Yandex GPT] {e}"

        if resp.status_code != 200:
            return f"[Yandex GPT error {resp.status_code}] {resp.text[:300]}"

        result = resp.json()
        return result["result"]["alternatives"][0]["message"]["text"]


def create_embedder(**kwargs) -> YandexEmbedder:
    return YandexEmbedder()


def create_llm(**kwargs) -> YandexLLM:
    return YandexLLM()