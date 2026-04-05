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

from config import (
    COLLECTION_NAME, QDRANT_PATH, QDRANT_URL,
    CHUNK_SIZE, CHUNK_OVERLAP,
    TOP_K_PER_METHOD, FINAL_TOP_K, BM25_INDEX_PATH,
)
from document_processor import process_document, SUPPORTED_EXTENSIONS, get_file_type
from yandex_client import YandexEmbedder, YandexLLM
from vector_store import VectorStore
from bm25_search import BM25Index
from query_expander import generate_query_variants
from retrieval_fusion import parallel_search, reciprocal_rank_fusion
from context_builder import build_context_with_neighbors, format_context_for_llm
from critic import generate_and_verify


embedder = None
store = None
llm = None
bm25_index = None
all_chunks_cache = []

LOGS_DIR = "./logs"
UPLOADS_DIR = "./uploads"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, store, llm, bm25_index, all_chunks_cache

    print("\n" + "=" * 60)
    print("RAG SERVER -- Yandex GPT + Yandex Embeddings + Hybrid Search")
    print("=" * 60 + "\n")

    embedder = YandexEmbedder()
    store = VectorStore(collection_name=COLLECTION_NAME, path=QDRANT_PATH, url=QDRANT_URL)
    llm = YandexLLM()
    bm25_index = BM25Index()

    if bm25_index.load(BM25_INDEX_PATH):
        all_chunks_cache = bm25_index.docs
    else:
        try:
            all_chunks_cache = store.get_all_chunks()
            if all_chunks_cache:
                bm25_index.build(all_chunks_cache)
        except:
            all_chunks_cache = []

    print(f"\nReady | Vectors: {store.count()} | BM25 docs: {bm25_index.n_docs}")
    yield
    print("\nStopped")


app = FastAPI(
    title="RAG API (Yandex) - Hybrid",
    description="RAG with Yandex GPT, Yandex Embeddings, BM25, RRF, Critic",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    include_history: bool = True
    max_history_messages: int = 10


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    was_corrected: bool
    elapsed_seconds: float


class IndexResponse(BaseModel):
    status: str
    collection_name: str
    documents_processed: int
    total_chunks: int
    total_vectors: int
    bm25_docs: int
    elapsed_seconds: float
    files_info: List[dict]


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


def index_documents(file_paths: List[str], collection_name: str, use_api: bool = True) -> dict:
    global embedder, store, bm25_index, all_chunks_cache
    from qdrant_client.models import PointStruct

    all_chunks = []
    files_info = []

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        file_type = get_file_type(file_path)

        try:
            chunks = process_document(file_path, CHUNK_SIZE, CHUNK_OVERLAP, use_api=use_api)
            all_chunks.extend(chunks)
            files_info.append({"filename": filename, "type": file_type, "chunks": len(chunks), "status": "ok"})
        except Exception as e:
            files_info.append({"filename": filename, "type": file_type, "chunks": 0, "status": "error", "error": str(e)})

    if not all_chunks:
        return {"error": "Failed to extract text", "files_info": files_info}

    dim = embedder.get_dimension()
    if collection_name:
        store.collection_name = collection_name
    store.create_collection(dim, recreate=True)

    print(f"\nVectorizing {len(all_chunks)} chunks via Yandex...")
    all_embeddings = []
    batch_size = 10

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]

        t0 = time.time()
        embs = embedder.embed_documents(texts)
        dt = time.time() - t0

        all_embeddings.extend(embs)
        done = min(i + batch_size, len(all_chunks))
        print(f"   [{done}/{len(all_chunks)}] in {dt:.1f}s")

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "text": ch["text"],
                "chunk_id": ch["chunk_id"],
                "source": ch.get("source", ""),
                "filename": ch.get("filename", ""),
                "page": ch.get("page", 1),
            }
        )
        for ch, emb in zip(all_chunks, all_embeddings)
    ]

    for i in range(0, len(points), 100):
        store.client.upsert(collection_name=store.collection_name, points=points[i:i + 100])

    bm25_index.build(all_chunks)
    bm25_index.save(BM25_INDEX_PATH)
    all_chunks_cache = all_chunks

    return {
        "total_chunks": len(all_chunks),
        "total_vectors": store.count(),
        "bm25_docs": bm25_index.n_docs,
        "collection_name": store.collection_name,
        "files_info": files_info
    }


def search_and_answer(
    question: str,
    user_id: str,
    top_k: int = 5,
    include_history: bool = True,
    max_history: int = 10
) -> dict:
    global embedder, store, llm, bm25_index, all_chunks_cache

    print(f"\n--- Query: {question[:80]}...")

    t0 = time.time()
    query_variants = generate_query_variants(llm, question)
    print(f"   Generated {len(query_variants)} query variants in {time.time() - t0:.1f}s")
    for v in query_variants:
        print(f"      [{v['type']}] {v['text'][:80]}...")

    t0 = time.time()
    all_ranked_lists = parallel_search(
        query_variants=query_variants,
        embedder=embedder,
        vector_store=store,
        bm25_index=bm25_index,
        top_k_per_method=TOP_K_PER_METHOD,
    )
    print(f"   Parallel search: {len(all_ranked_lists)} lists, {sum(len(r) for r in all_ranked_lists)} total candidates in {time.time() - t0:.1f}s")

    top_chunks = reciprocal_rank_fusion(all_ranked_lists, final_top_k=top_k)
    print(f"   RRF selected {len(top_chunks)} chunks")

    if not top_chunks:
        return {"answer": "Database is empty. Upload documents via /index", "sources": [], "was_corrected": False}

    sorted_cache = sorted(all_chunks_cache, key=lambda x: (x.get("source", ""), x.get("chunk_id", -1)))
    enriched = build_context_with_neighbors(top_chunks, sorted_cache)
    context_str = format_context_for_llm(enriched)

    history_messages = get_history_for_llm(user_id, max_history) if include_history else []

    t0 = time.time()
    result = generate_and_verify(
        llm=llm,
        question=question,
        context_str=context_str,
        history_messages=history_messages,
    )
    print(f"   Generation + verification in {time.time() - t0:.1f}s, corrected={result['was_corrected']}")

    add_to_history(user_id, "user", question)
    sources_for_history = [
        {"source": ch.get("source", ""), "page": ch.get("page", 1), "rrf_score": ch.get("rrf_score", 0)}
        for ch in top_chunks
    ]
    add_to_history(user_id, "assistant", result["answer"], sources_for_history)

    sources_response = [
        {
            "source": ch.get("source", ""),
            "page": ch.get("page", 1),
            "rrf_score": round(ch.get("rrf_score", 0), 4),
            "preview": ch.get("text", "")[:200],
        }
        for ch in top_chunks
    ]

    return {
        "answer": result["answer"],
        "sources": sources_response,
        "was_corrected": result["was_corrected"],
    }


@app.get("/")
async def root():
    return {
        "service": "RAG API (Yandex GPT + Yandex Embeddings + Hybrid Search)",
        "formats": ["PDF", "DOC", "DOCX"],
        "features": ["Multi-query expansion", "BM25 + Vector hybrid", "RRF fusion", "Critic verification"],
        "endpoints": ["POST /index", "POST /chat", "GET /status"]
    }


@app.get("/status")
async def status():
    return {
        "status": "ok",
        "vectors": store.count() if store else 0,
        "bm25_docs": bm25_index.n_docs if bm25_index else 0,
        "llm": "Yandex GPT",
        "embedder": "Yandex Embeddings (256 dim)",
        "features": ["multi-query", "hybrid-search", "rrf", "critic"]
    }


@app.post("/index", response_model=IndexResponse)
async def index_endpoint(
    files: List[UploadFile] = File(...),
    collection_name: Optional[str] = Form(default=None),
    use_api_parser: bool = Form(default=True)
):
    if len(files) < 1 or len(files) > 20:
        raise HTTPException(400, "From 1 to 20 files")

    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported: {f.filename}")

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
            bm25_docs=result["bm25_docs"],
            elapsed_seconds=round(time.time() - t0, 2),
            files_info=result["files_info"]
        )
    finally:
        for p in saved:
            try:
                os.remove(p)
            except:
                pass


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if store.count() == 0:
        raise HTTPException(400, "Database is empty")

    t0 = time.time()
    result = search_and_answer(req.question, req.user_id, req.top_k, req.include_history, req.max_history_messages)

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        was_corrected=result["was_corrected"],
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