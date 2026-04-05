import uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import qdrant_client

class VectorStore:
    def __init__(self, collection_name="string_1", path="./qdrant_data", url=None):
        self.collection_name = collection_name
        if url:
            self.client = QdrantClient(url=url, timeout=30)
        else:
            self.client = QdrantClient(path=path)
            print(f"Qdrant: локально -> {path}")

    def create_collection(self, dimension: int, recreate: bool = False):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            if recreate:
                self.client.delete_collection(self.collection_name)
            else:
                print(f"   Коллекция {self.collection_name} уже есть")
                return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        print(f"   Коллекция создана (dim={dimension})")

    def insert_chunks(self, chunks: List[Dict], embeddings: List[List[float]], source: str = ""):
        points = []
        for chunk, emb in zip(chunks, embeddings):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={"text": chunk["text"], "chunk_id": chunk["chunk_id"], "source": source},
            ))
        for i in range(0, len(points), 100):
            self.client.upsert(collection_name=self.collection_name, points=points[i:i+100])
        print(f"   Вставлено: {len(points)} точек")

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        # Новый API (qdrant-client >= 1.10)
        query_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            )
        hits = query_result.points

        results = []
        for hit in hits:
            results.append({
                "text": hit.payload["text"],
                "score": getattr(hit, "score", 0.0),
                "chunk_id": hit.payload.get("chunk_id", -1),
                "source": hit.payload.get("source", ""),
            })
        return results

    def count(self) -> int:
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def close(self):
        try:
            self.client.close()
        except:
            pass