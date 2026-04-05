from typing import List, Dict


def build_context_with_neighbors(
    top_chunks: List[Dict],
    all_chunks_sorted: List[Dict],
) -> List[Dict]:
    chunk_index = {}
    for i, ch in enumerate(all_chunks_sorted):
        key = (ch.get("source", ""), ch.get("chunk_id", -1))
        chunk_index[key] = i

    enriched = []
    seen_keys = set()

    for chunk in top_chunks:
        key = (chunk.get("source", ""), chunk.get("chunk_id", -1))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        idx = chunk_index.get(key)
        if idx is None:
            enriched.append({
                "main_text": chunk["text"],
                "before_text": "",
                "after_text": "",
                "source": chunk.get("source", ""),
                "page": chunk.get("page", 1),
                "chunk_id": chunk.get("chunk_id", -1),
                "rrf_score": chunk.get("rrf_score", 0),
            })
            continue

        before_text = ""
        if idx > 0:
            prev = all_chunks_sorted[idx - 1]
            if prev.get("source", "") == chunk.get("source", ""):
                before_text = prev["text"]

        after_text = ""
        if idx < len(all_chunks_sorted) - 1:
            nxt = all_chunks_sorted[idx + 1]
            if nxt.get("source", "") == chunk.get("source", ""):
                after_text = nxt["text"]

        enriched.append({
            "main_text": chunk["text"],
            "before_text": before_text,
            "after_text": after_text,
            "source": chunk.get("source", ""),
            "page": chunk.get("page", 1),
            "chunk_id": chunk.get("chunk_id", -1),
            "rrf_score": chunk.get("rrf_score", 0),
        })

    return enriched


def format_context_for_llm(enriched_chunks: List[Dict]) -> str:
    parts = []
    for i, ch in enumerate(enriched_chunks, 1):
        source = ch["source"]
        page = ch["page"]
        marker = f"[Source {i}: {source}, p. {page}]"

        combined = ""
        if ch["before_text"]:
            combined += f"[preceding context] {ch['before_text']}\n\n"
        combined += ch["main_text"]
        if ch["after_text"]:
            combined += f"\n\n[following context] {ch['after_text']}"

        parts.append(f"{marker}\n{combined}")

    return "\n\n---\n\n".join(parts)