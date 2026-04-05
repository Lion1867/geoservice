from typing import List, Dict
from yandex_client import YandexLLM


def generate_query_variants(llm: YandexLLM, original_query: str) -> List[Dict]:
    variants = [
        {"type": "original", "text": original_query}
    ]

    synonym_prompt = [
        {"role": "system", "content": "You are a helpful assistant. Respond ONLY with the rephrased query, nothing else. Keep the language of the original query."},
        {"role": "user", "content": f"Rephrase the following query using synonyms, keep the same meaning:\n\n{original_query}"}
    ]
    synonym_text = llm.generate(synonym_prompt).strip()
    if synonym_text and not synonym_text.startswith("["):
        variants.append({"type": "synonym", "text": synonym_text})

    expert_prompt = [
        {"role": "system", "content": "You are a domain expert. Respond ONLY with the rephrased query using professional terminology, nothing else. Keep the language of the original query."},
        {"role": "user", "content": f"Rephrase this query as a domain expert would formulate it, using precise professional terms:\n\n{original_query}"}
    ]
    expert_text = llm.generate(expert_prompt).strip()
    if expert_text and not expert_text.startswith("["):
        variants.append({"type": "expert", "text": expert_text})

    hyde_prompt = [
        {"role": "system", "content": "You are a helpful assistant. Write a short paragraph (3-5 sentences) that would be an ideal answer to the given question. Respond ONLY with the answer text, nothing else. Keep the language of the original query."},
        {"role": "user", "content": f"Write an ideal short answer to this question:\n\n{original_query}"}
    ]
    hyde_text = llm.generate(hyde_prompt).strip()
    if hyde_text and not hyde_text.startswith("["):
        variants.append({"type": "hyde", "text": hyde_text})

    return variants