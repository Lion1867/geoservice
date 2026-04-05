"""
RAG FastAPI Server с Yandex GPT и Yandex Embeddings
"""

import os
import time
import json
import uuid
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import *
from document_processor import process_document, SUPPORTED_EXTENSIONS, get_file_type
from yandex_client import YandexEmbedder, YandexLLM
from vector_store import VectorStore


# =====================================================
# ГЛОБАЛЬНЫЕ КОМПОНЕНТЫ
# =====================================================
embedder = None
store = None
llm = None

LOGS_DIR = "./logs"
UPLOADS_DIR = "./uploads"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, store, llm

    print("\n" + "="*60)
    print("🚀 RAG SERVER — Yandex GPT + Yandex Embeddings")
    print("="*60 + "\n")

    embedder = YandexEmbedder()
    store = VectorStore(collection_name=COLLECTION_NAME, path=QDRANT_PATH, url=QDRANT_URL)
    llm = YandexLLM()

    print(f"\n✅ Готов | Векторов: {store.count()}")
    yield
    print("\n👋 Остановлен")


app = FastAPI(
    title="RAG API (Yandex)",
    description="RAG с Yandex GPT и Yandex Embeddings",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# МОДЕЛИ
# =====================================================
class ChatRequest(BaseModel):
    user_id: str
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    include_history: bool = True
    max_history_messages: int = 10


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    elapsed_seconds: float


class IndexResponse(BaseModel):
    status: str
    collection_name: str
    documents_processed: int
    total_chunks: int
    total_vectors: int
    elapsed_seconds: float
    files_info: List[dict]


# =====================================================
# ИСТОРИЯ
# =====================================================
def get_history_path(user_id: str) -> str:
    safe_id = "".join(c for c in user_id if c.isalnum() or c in "-_")
    return os.path.join(LOGS_DIR, f"{safe_id}.json")


def load_history(user_id: str) -> dict:
    path = get_history_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {"user_id": user_id, "created_at": datetime.now().isoformat(), "messages": []}


def save_history(user_id: str, history: dict):
    history["updated_at"] = datetime.now().isoformat()
    with open(get_history_path(user_id), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def add_to_history(user_id: str, role: str, content: str, sources: list = None):
    history = load_history(user_id)
    msg = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
    if sources:
        msg["sources"] = sources
    history["messages"].append(msg)
    save_history(user_id, history)


def get_history_for_llm(user_id: str, max_messages: int = 10) -> List[dict]:
    history = load_history(user_id)
    messages = history.get("messages", [])
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    return [{"role": m["role"], "content": m["content"]} for m in recent]


# =====================================================
# ИНДЕКСАЦИЯ
# =====================================================
def index_documents(file_paths: List[str], collection_name: str, use_api: bool = True) -> dict:
    global embedder, store
    from qdrant_client.models import PointStruct

    all_chunks = []
    files_info = []

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        file_type = get_file_type(file_path)

        try:
            chunks = process_document(file_path, CHUNK_SIZE, CHUNK_OVERLAP, use_api=use_api)
            for chunk in chunks:
                chunk["source"] = filename
            all_chunks.extend(chunks)
            files_info.append({"filename": filename, "type": file_type, "chunks": len(chunks), "status": "ok"})
        except Exception as e:
            files_info.append({"filename": filename, "type": file_type, "chunks": 0, "status": "error", "error": str(e)})

    if not all_chunks:
        return {"error": "Не удалось извлечь текст", "files_info": files_info}

    dim = embedder.get_dimension()
    if collection_name:
        store.collection_name = collection_name
    store.create_collection(dim, recreate=True)

    # Векторизация (используем embed_documents для документов)
    print(f"\n🔢 Векторизация {len(all_chunks)} чанков через Yandex...")
    all_embeddings = []
    batch_size = 10  # Yandex API медленнее, делаем меньшие батчи

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        
        t0 = time.time()
        embs = embedder.embed_documents(texts)
        dt = time.time() - t0
        
        all_embeddings.extend(embs)
        done = min(i + batch_size, len(all_chunks))
        print(f"   [{done}/{len(all_chunks)}] за {dt:.1f}с")

    # Сохранение
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={"text": ch["text"], "chunk_id": ch["chunk_id"], "source": ch["source"]}
        )
        for ch, emb in zip(all_chunks, all_embeddings)
    ]

    for i in range(0, len(points), 100):
        store.client.upsert(collection_name=store.collection_name, points=points[i:i+100])

    return {
        "total_chunks": len(all_chunks),
        "total_vectors": store.count(),
        "collection_name": store.collection_name,
        "files_info": files_info
    }


# =====================================================
# ПОИСК + ОТВЕТ
# =====================================================
def search_and_answer(
    question: str,
    user_id: str,
    top_k: int = 5,
    include_history: bool = True,
    max_history: int = 10
) -> dict:
    global embedder, store, llm

    # Используем embed_query для запроса (другая модель!)
    q_vec = embedder.embed_query(question)
    results = store.search(q_vec, top_k=top_k)

    if not results:
        return {"answer": "База пуста. Загрузите документы через /index", "sources": []}

    # Контекст
    context_parts = [f"[{r['source']}, score={r['score']:.3f}]\n{r['text']}" for r in results]
    context_str = "\n\n---\n\n".join(context_parts)

    # История
    history_messages = get_history_for_llm(user_id, max_history) if include_history else []

    system = """Ты — интеллектуальный ассистент по документам.
Отвечай ТОЛЬКО на основе предоставленного контекста.
Если ответа нет — скажи: «В документах ответ не найден».
Указывай источник. Отвечай на русском."""

    messages = [{"role": "system", "content": system}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": f"Контекст:\n\n{context_str}\n\n---\n\nВопрос: {question}\n\nОтвет:"})

    answer = llm.generate(messages)

    # История
    add_to_history(user_id, "user", question)
    add_to_history(user_id, "assistant", answer, [{"source": r["source"], "score": r["score"]} for r in results])

    return {
        "answer": answer,
        "sources": [{"source": r["source"], "score": round(r["score"], 4), "preview": r["text"][:200]} for r in results]
    }


# =====================================================
# ENDPOINTS
# =====================================================
@app.get("/")
async def root():
    return {
        "service": "RAG API (Yandex GPT + Yandex Embeddings)",
        "formats": ["PDF", "DOC", "DOCX"],
        "endpoints": ["POST /index", "POST /chat", "GET /status"]
    }


@app.get("/status")
async def status():
    return {
        "status": "ok",
        "vectors": store.count() if store else 0,
        "llm": "Yandex GPT",
        "embedder": "Yandex Embeddings (256 dim)"
    }


@app.post("/index", response_model=IndexResponse)
async def index_endpoint(
    files: List[UploadFile] = File(...),
    collection_name: Optional[str] = Form(default=None),
    use_api_parser: bool = Form(default=True)
):
    if len(files) < 1 or len(files) > 20:
        raise HTTPException(400, "От 1 до 20 файлов")

    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(400, f"Неподдерживаемый: {f.filename}")

    t0 = time.time()
    saved = []

    try:
        for f in files:
            path = os.path.join(UPLOADS_DIR, f"{uuid.uuid4().hex[:8]}_{f.filename}")
            with open(path, "wb") as out:
                out.write(await f.read())
            saved.append(path)

        result = index_documents(saved, collection_name or COLLECTION_NAME, use_api_parser)

        if "error" in result:
            raise HTTPException(400, result["error"])

        return IndexResponse(
            status="success",
            collection_name=result["collection_name"],
            documents_processed=len(files),
            total_chunks=result["total_chunks"],
            total_vectors=result["total_vectors"],
            elapsed_seconds=round(time.time() - t0, 2),
            files_info=result["files_info"]
        )
    finally:
        for p in saved:
            try: os.remove(p)
            except: pass


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if store.count() == 0:
        raise HTTPException(400, "База пуста")

    t0 = time.time()
    result = search_and_answer(req.question, req.user_id, req.top_k, req.include_history, req.max_history_messages)

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        elapsed_seconds=round(time.time() - t0, 2)
    )


@app.get("/history/{user_id}")
async def get_history_endpoint(user_id: str):
    h = load_history(user_id)
    return {"user_id": user_id, "messages": h["messages"]}


@app.delete("/history/{user_id}")
async def clear_history_endpoint(user_id: str):
    path = get_history_path(user_id)
    if os.path.exists(path):
        os.remove(path)
        return {"status": "deleted"}
    return {"status": "not_found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)