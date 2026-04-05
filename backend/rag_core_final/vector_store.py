import uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class VectorStore:
    def __init__(self, collection_name="string_1", path="./qdrant_data", url=None):
        self.collection_name = collection_name
        if url:
            self.client = QdrantClient(url=url, timeout=30)
        else:
            self.client = QdrantClient(path=path)
            print(f"Qdrant: local -> {path}")

    def create_collection(self, dimension: int, recreate: bool = False):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            if recreate:
                self.client.delete_collection(self.collection_name)
            else:
                print(f"   Collection {self.collection_name} exists")
                return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        print(f"   Collection created (dim={dimension})")

    def insert_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        points = []
        for chunk, emb in zip(chunks, embeddings):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "text": chunk["text"],
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk.get("source", ""),
                    "filename": chunk.get("filename", ""),
                    "page": chunk.get("page", 1),
                },
            ))
        for i in range(0, len(points), 100):
            self.client.upsert(collection_name=self.collection_name, points=points[i:i+100])
        print(f"   Inserted: {len(points)} points")

    def search(self, query_vector: List[float], top_k: int = 25) -> List[Dict]:
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
                "filename": hit.payload.get("filename", ""),
                "page": hit.payload.get("page", 1),
            })
        return results

    def get_all_chunks(self) -> List[Dict]:
        all_chunks = []
        offset = None
        while True:
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = result
            for point in points:
                all_chunks.append({
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "chunk_id": point.payload.get("chunk_id", -1),
                    "source": point.payload.get("source", ""),
                    "filename": point.payload.get("filename", ""),
                    "page": point.payload.get("page", 1),
                })
            if next_offset is None:
                break
            offset = next_offset
        return all_chunks

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