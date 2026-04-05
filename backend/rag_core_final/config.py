import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OAUTH_TOKEN = os.getenv("OAUTH_TOKEN")
CATALOG_ID = os.getenv("CATALOG_ID")

YANDEX_LLM_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YANDEX_LLM_MODEL = "yandexgpt-lite"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2000

YANDEX_EMB_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
YANDEX_EMB_MODEL_DOC = "text-search-doc"
YANDEX_EMB_MODEL_QUERY = "text-search-query"

QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_data")
QDRANT_URL = None
COLLECTION_NAME = "newcollection_final"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K_PER_METHOD = 25
FINAL_TOP_K = 5
RRF_K = 60

BM25_INDEX_PATH = os.path.join(BASE_DIR, "bm25_index")