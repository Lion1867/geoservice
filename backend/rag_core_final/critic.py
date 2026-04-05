from typing import List, Dict
from yandex_client import YandexLLM


def audit_answer(
    llm: YandexLLM,
    question: str,
    draft_answer: str,
    context_str: str,
) -> Dict:
    audit_prompt = [
        {"role": "system", "content": (
            "You are a strict auditor. Your task is to verify a draft answer against source documents. "
            "Check each factual claim in the draft. Check if cited document names and page numbers actually exist in the provided sources. "
            "Respond in the following JSON-like format:\n"
            "VERDICT: OK or FAIL\n"
            "ISSUES: list of issues (if FAIL), each on a new line starting with '- '\n"
            "If everything is correct, write VERDICT: OK and ISSUES: none\n"
            "Respond in Russian."
        )},
        {"role": "user", "content": (
            f"Source documents:\n\n{context_str}\n\n---\n\n"
            f"Question: {question}\n\n"
            f"Draft answer:\n{draft_answer}\n\n"
            f"Verify the draft answer. Are all claims supported by the sources? Are cited references correct?"
        )}
    ]

    audit_result = llm.generate(audit_prompt)
    audit_upper = audit_result.upper()

    is_ok = "VERDICT: OK" in audit_upper or ("OK" in audit_upper and "FAIL" not in audit_upper)

    return {
        "is_ok": is_ok,
        "audit_text": audit_result,
    }


def fix_answer(
    llm: YandexLLM,
    question: str,
    draft_answer: str,
    audit_text: str,
    context_str: str,
) -> str:
    fix_prompt = [
        {"role": "system", "content": (
            "You are an intelligent document assistant. "
            "Your previous answer was reviewed by an auditor and issues were found. "
            "Rewrite the answer fixing all issues noted by the auditor. "
            "Use ONLY information from the provided source documents. "
            "Cite sources in format (Document name, p. N). "
            "Respond in Russian."
        )},
        {"role": "user", "content": (
            f"Source documents:\n\n{context_str}\n\n---\n\n"
            f"Question: {question}\n\n"
            f"Your previous answer:\n{draft_answer}\n\n"
            f"Auditor comments:\n{audit_text}\n\n"
            f"Rewrite the answer fixing all issues:"
        )}
    ]

    return llm.generate(fix_prompt)


def generate_and_verify(
    llm: YandexLLM,
    question: str,
    context_str: str,
    history_messages: List[Dict] = None,
) -> Dict:
    system = (
        "You are an intelligent document assistant. "
        "Answer ONLY based on the provided context. "
        "If the answer is not in the documents, say: 'The answer was not found in the documents.' "
        "Cite sources in format (Document name, p. N). "
        "Respond in Russian."
    )

    messages = [{"role": "system", "content": system}]
    if history_messages:
        messages.extend(history_messages)
    messages.append({
        "role": "user",
        "content": f"Context:\n\n{context_str}\n\n---\n\nQuestion: {question}\n\nAnswer:"
    })

    draft = llm.generate(messages)

    audit = audit_answer(llm, question, draft, context_str)

    if audit["is_ok"]:
        return {
            "answer": draft,
            "was_corrected": False,
            "audit_text": audit["audit_text"],
        }

    fixed = fix_answer(llm, question, draft, audit["audit_text"], context_str)

    return {
        "answer": fixed,
        "was_corrected": True,
        "audit_text": audit["audit_text"],
    }