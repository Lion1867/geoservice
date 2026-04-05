"""
metrics.py
Метрики для оценки RAG-системы: поиск + генерация.

Импорт:
    from metrics import (
        compute_semantic_similarity,
        evaluate_search_metrics,
        evaluate_generation_metrics,
        THRESHOLDS
    )
"""

import re
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer

# =====================================================
# КОНФИГУРАЦИЯ
# =====================================================

THRESHOLDS = {
    "hit_rate_at_5": 0.80,
    "mrr": 0.70,
    "sbert_score": 0.68,
    "rouge_l_f1": 0.30
}

SEMANTIC_RELEVANCE_THRESHOLD = 0.70

# Модель для русского языка (лучшая совместимость с Windows)
SBERT_MODEL_NAME = "cointegrated/rubert-tiny2"

_sbert_model = None
_rouge_scorer_instance = None


# =====================================================
# ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ
# =====================================================

def get_sbert_model(model_name: str = SBERT_MODEL_NAME) -> SentenceTransformer:
    global _sbert_model
    if _sbert_model is None:
        print(f"Загрузка модели: {model_name}")
        _sbert_model = SentenceTransformer(model_name)
    return _sbert_model


def get_rouge_scorer() -> rouge_scorer.RougeScorer:
    global _rouge_scorer_instance
    if _rouge_scorer_instance is None:
        _rouge_scorer_instance = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=False
        )
    return _rouge_scorer_instance


# =====================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================

def preprocess_ru(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u0400-\u04FF\-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def compute_semantic_similarity(text1: str, text2: str) -> float:
    model = get_sbert_model()
    emb1 = model.encode(text1, convert_to_tensor=True, show_progress_bar=False)
    emb2 = model.encode(text2, convert_to_tensor=True, show_progress_bar=False)
    return float(util.cos_sim(emb1, emb2).item())


# =====================================================
# МЕТРИКИ ПОИСКА (2 штуки)
# =====================================================

def evaluate_search_metrics(
    retrieved_sources: List[Dict],
    reference_text: str,
    top_k: int = 5,
    semantic_threshold: float = SEMANTIC_RELEVANCE_THRESHOLD
) -> Dict[str, float]:
    """
    Оценка качества поиска.
    
    Args:
        retrieved_sources: список источников из API (с полем 'preview')
        reference_text: эталонный текст чанка
        top_k: количество верхних результатов для оценки
        semantic_threshold: порог семантического сходства
    
    Returns:
        dict с метриками hit_rate_at_5, mrr, max_similarity
    """
    if not retrieved_sources:
        return {
            "hit_rate_at_5": 0,
            "mrr": 0.0,
            "max_similarity": 0.0,
            "similarities": []
        }
    
    similarities = []
    for source in retrieved_sources[:top_k]:
        chunk_text = source.get("preview", "") or source.get("text", "")
        if chunk_text:
            sim = compute_semantic_similarity(chunk_text, reference_text)
            similarities.append(sim)
        else:
            similarities.append(0.0)
    
    hit_rate = 1 if any(s >= semantic_threshold for s in similarities) else 0
    
    mrr = 0.0
    for rank, sim in enumerate(similarities, 1):
        if sim >= semantic_threshold:
            mrr = 1.0 / rank
            break
    
    return {
        "hit_rate_at_5": hit_rate,
        "mrr": round(mrr, 4),
        "max_similarity": round(max(similarities) if similarities else 0.0, 4),
        "similarities": [round(s, 4) for s in similarities]
    }


# =====================================================
# МЕТРИКИ ГЕНЕРАЦИИ (2 штуки)
# =====================================================

def evaluate_generation_metrics(
    generated_answer: str,
    expected_answer: str
) -> Dict[str, float]:
    if not generated_answer or not expected_answer:
        return {
            "sbert_score": 0.0,
            "rouge_l_f1": 0.0
        }
    
    sbert_score = compute_semantic_similarity(generated_answer, expected_answer)
    
    scorer = get_rouge_scorer()
    ref_clean = preprocess_ru(expected_answer)
    gen_clean = preprocess_ru(generated_answer)
    rouge_scores = scorer.score(ref_clean, gen_clean)
    rouge_l_f1 = round(rouge_scores['rougeL'].fmeasure, 4)
    
    return {
        "sbert_score": round(sbert_score, 4),
        "rouge_l_f1": rouge_l_f1
    }


# =====================================================
# КОМБИНИРОВАННАЯ ОЦЕНКА
# =====================================================

def evaluate_case(
    retrieved_sources: List[Dict],
    reference_text: str,
    generated_answer: str,
    expected_answer: str,
    top_k: int = 5
) -> Dict[str, Dict[str, float]]:
    return {
        "search": evaluate_search_metrics(retrieved_sources, reference_text, top_k),
        "generation": evaluate_generation_metrics(generated_answer, expected_answer)
    }


