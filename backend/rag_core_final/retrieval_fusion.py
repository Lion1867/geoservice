from typing import List, Dict, Tuple
from collections import defaultdict

from config import RRF_K, TOP_K_PER_METHOD, FINAL_TOP_K


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    k: int = None,
    final_top_k: int = None,
) -> List[Dict]:
    k = k or RRF_K
    final_top_k = final_top_k or FINAL_TOP_K

    scores = defaultdict(float)
    chunk_data = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            key = (item.get("source", ""), item.get("chunk_id", -1))

            rrf_score = 1.0 / (k + rank + 1)
            scores[key] += rrf_score

            if key not in chunk_data:
                chunk_data[key] = item

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for key in sorted_keys[:final_top_k]:
        item = chunk_data[key].copy()
        item["rrf_score"] = scores[key]
        results.append(item)

    return results


def parallel_search(
    query_variants: List[Dict],
    embedder,
    vector_store,
    bm25_index,
    top_k_per_method: int = None,
) -> List[List[Dict]]:
    top_k_per_method = top_k_per_method or TOP_K_PER_METHOD
    all_ranked_lists = []

    for variant in query_variants:
        query_text = variant["text"]
        query_type = variant["type"]

        q_vec = embedder.embed_query(query_text)
        vector_results = vector_store.search(q_vec, top_k=top_k_per_method)
        all_ranked_lists.append(vector_results)

        bm25_results = bm25_index.search(query_text, top_k=top_k_per_method)
        all_ranked_lists.append(bm25_results)

    return all_ranked_lists