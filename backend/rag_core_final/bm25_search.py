import os
import json
import math
import re
from typing import List, Dict
from collections import Counter

from config import BM25_INDEX_PATH


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\u0400-\u04FF]+", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2]
    return tokens


class BM25Index:
    def __init__(self):
        self.docs = []
        self.doc_freqs = Counter()
        self.doc_lens = []
        self.avgdl = 0.0
        self.n_docs = 0
        self.inverted_index = {}
        self.k1 = 1.5
        self.b = 0.75

    def build(self, chunks: List[Dict]):
        self.docs = chunks
        self.n_docs = len(chunks)
        self.doc_lens = []
        self.inverted_index = {}
        self.doc_freqs = Counter()

        all_token_lists = []
        for i, chunk in enumerate(chunks):
            tokens = tokenize(chunk["text"])
            all_token_lists.append(tokens)
            self.doc_lens.append(len(tokens))

            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append((i, count))

        self.avgdl = sum(self.doc_lens) / self.n_docs if self.n_docs > 0 else 1.0
        print(f"   BM25 index built: {self.n_docs} documents, {len(self.doc_freqs)} unique tokens")

    def search(self, query: str, top_k: int = 25) -> List[Dict]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = {}

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            df = self.doc_freqs[token]
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

            for doc_idx, tf in self.inverted_index[token]:
                dl = self.doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score = idf * numerator / denominator

                if doc_idx not in scores:
                    scores[doc_idx] = 0.0
                scores[doc_idx] += score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_idx, score in ranked:
            doc = self.docs[doc_idx]
            results.append({
                "text": doc["text"],
                "score": score,
                "chunk_id": doc.get("chunk_id", -1),
                "source": doc.get("source", ""),
                "filename": doc.get("filename", ""),
                "page": doc.get("page", 1),
            })

        return results

    def save(self, path: str = None):
        path = path or BM25_INDEX_PATH
        os.makedirs(path, exist_ok=True)

        data = {
            "docs": self.docs,
            "doc_freqs": dict(self.doc_freqs),
            "doc_lens": self.doc_lens,
            "avgdl": self.avgdl,
            "n_docs": self.n_docs,
            "inverted_index": {k: v for k, v in self.inverted_index.items()},
        }

        with open(os.path.join(path, "bm25_data.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"   BM25 index saved to {path}")

    def load(self, path: str = None) -> bool:
        path = path or BM25_INDEX_PATH
        filepath = os.path.join(path, "bm25_data.json")
        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.docs = data["docs"]
            self.doc_freqs = Counter(data["doc_freqs"])
            self.doc_lens = data["doc_lens"]
            self.avgdl = data["avgdl"]
            self.n_docs = data["n_docs"]
            self.inverted_index = {k: [tuple(pair) for pair in v] for k, v in data["inverted_index"].items()}
            print(f"   BM25 index loaded: {self.n_docs} documents")
            return True
        except Exception as e:
            print(f"   BM25 load error: {e}")
            return False