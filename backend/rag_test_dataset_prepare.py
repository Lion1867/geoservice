"""
rag_test_dataset_prepare.py
Подготовка тестового датасета для оценки RAG-системы на домене СанПиН/СП.

Логика:
- Из каждого PDF-документа берётся ровно 20 случайных чанков
- Для каждого чанка генерируется вопрос и эталонный ответ через LLM
- Фокус: вопросы для градостроительного анализа и проектирования

Использование:
    python rag_test_dataset_prepare.py

На выходе:
    data/test_dataset_sanpin.json — датасет из ~280 тестовых кейсов (14×20)
"""

import os
import json
import re
import time
import random
import requests
from datetime import datetime
from typing import List, Dict, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("Установите PyMuPDF: pip install PyMuPDF")

# =====================================================
# КОНФИГУРАЦИЯ
# =====================================================

PDF_FOLDER = "data"
OUTPUT_FILE = "data/test_dataset_sanpin.json"
RESUME_FILE = "data/.dataset_prepare_state.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 100

CHUNKS_PER_DOCUMENT = 20  # Ключевое: ровно 20 чанков из каждого документа
API_DELAY = 2.0

random.seed(42)


# =====================================================
# LLM API КЛИЕНТ
# =====================================================

class LLMClient:
    """Клиент для работы с LLM API."""
    
    def __init__(self, oauth_token: str, catalog_id: str):
        self.oauth_token = oauth_token
        self.catalog_id = catalog_id
        self._iam_token = None
        self._iam_expires = 0
    
    def _get_iam_token(self) -> str:
        if self._iam_token and time.time() < self._iam_expires - 3600:
            return self._iam_token
        
        url = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
        data = {"yandexPassportOauthToken": self.oauth_token}
        
        resp = requests.post(url, json=data, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Ошибка получения IAM токена: {resp.status_code} - {resp.text}")
        
        result = resp.json()
        self._iam_token = result["iamToken"]
        self._iam_expires = time.time() + 12 * 3600
        
        return self._iam_token
    
    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._get_iam_token()}",
            "Content-Type": "application/json"
        }
    
    def generate_qa(self, chunk_text: str, doc_name: str) -> Optional[Dict]:
        """
        Генерирует вопрос и эталонный ответ для тестирования RAG.
        Фокус: вопросы для градостроительного анализа и проектирования.
        """
        prompt = f"""Ты — эксперт по нормативной базе в строительстве и градостроительстве (СанПиН, СП, ГОСТ).
Твоя задача — создавать тестовые вопросы для проверки RAG-системы, которая помогает архитекторам и инженерам.

ДОКУМЕНТ: {doc_name}

ФРАГМЕНТ ТЕКСТА:
---
{chunk_text[:700]}
---

ЗАДАЧА:
Создай ОДИН профессиональный вопрос, который мог бы задать проектировщик или градостроительный аналитик при работе с этим нормативом.

ТИПЫ РЕЛЕВАНТНЫХ ВОПРОСОВ (приоритет):
1. Требования к параметрам объекта: "Какие минимальные расстояния должны соблюдаться между...", "Какова допустимая инсоляция для..."
2. Ограничения и запреты: "При каких условиях запрещено размещение...", "Что не допускается при проектировании..."
3. Методики расчёта: "Как рассчитывается коэффициент...", "По какой формуле определяется..."
4. Критерии соответствия: "При каких значениях объект считается соответствующим нормативу...", "Какие параметры проверяются при экспертизе..."
5. Ссылки на другие нормативы: "На какой документ ссылается данный пункт при определении..."

НЕ РЕКОМЕНДУЕТСЯ:
- Вопросы на простое извлечение фактов: "Какая температура в городе Х?", "Какая дата указана?"
- Вопросы про метаданные документа: "Кто утвердил?", "Когда опубликован?"
- Общие вопросы без привязки к проектированию: "Что говорится о температуре?"

ТРЕБОВАНИЯ К ВОПРОСУ:
1. Вопрос должен быть практическим: отвечать на него нужно при принятии проектного решения
2. Ответ должен содержаться в приведённом фрагменте (не выдумывай)
3. Формулируй вопрос так, как его задал бы профессионал (терминология)
4. Избегай вопросов с очевидным ответом из одного слова

ЭТАЛОННЫЙ ОТВЕТ:
- Должен быть развёрнутым (2-4 предложения)
- Должен содержать ссылку на конкретное требование/значение из фрагмента
- Может включать краткое пояснение применения в проектировании

Верни ответ ТОЛЬКО в формате JSON без markdown:
{{
    "question": "текст профессионального вопроса",
    "expected_answer": "развёрнутый эталонный ответ",
    "topic": "тема (например: инсоляция, пожарная_безопасность, шум, расстояния)",
    "difficulty": "easy или medium или hard"
}}
"""
        
        data = {
            "modelUri": f"gpt://{self.catalog_id}/yandexgpt-lite/latest",
            "completionOptions": {
                "maxTokens": 600,
                "temperature": 0.3,
                "stream": False
            },
            "messages": [
                {"role": "system", "text": "Ты создаёшь тестовые вопросы для профессионалов в градостроительстве. Отвечай только JSON."},
                {"role": "user", "text": prompt}
            ]
        }
        
        try:
            resp = requests.post(
                "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                headers=self._get_headers(),
                json=data,
                timeout=90
            )
            
            if resp.status_code != 200:
                print(f"   [WARN] Ошибка API: {resp.status_code}")
                return None
            
            result = resp.json()
            answer_text = result["result"]["alternatives"][0]["message"]["text"]
            
            qa = self._parse_json_response(answer_text)
            if qa:
                if self._is_low_quality_question(qa["question"], qa["expected_answer"]):
                    print(f"      [SKIP] Низкое качество вопроса")
                    return None
                return qa
            
        except Exception as e:
            print(f"   [WARN] Исключение: {e}")
        
        return None
    
    def _is_low_quality_question(self, question: str, answer: str) -> bool:
        """Фильтр низкокачественных вопросов."""
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        low_quality_patterns = [
            r'какая дата',
            r'какой документ.*утвердил',
            r'кто утвердил',
            r'когда.*опубликован',
            r'температура в городе',
            r'температура в .* в ',
            r'среднегодовая температура',
            r'минимальная температура',
            r'какая организация',
            r'что представлено в фрагменте',
            r'какой регион.*город',
        ]
        
        for pattern in low_quality_patterns:
            if re.search(pattern, question_lower):
                return True
        
        if re.match(r'^[\d\.,\s°%-]+$', answer.strip()):
            return True
        
        if len(set(question_lower.split()) & set(answer_lower.split())) / max(len(question_lower.split()), 1) > 0.7:
            return True
        
        return False
    
    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Извлекает JSON из ответа модели."""
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            return json.loads(text)
        except:
            first_sent = text.split('.')[0].strip()[:100]
            return {
                "question": f"Какие требования предъявляются к: '{first_sent}'?",
                "expected_answer": text[:300],
                "topic": "general",
                "difficulty": "medium"
            }


# =====================================================
# ИЗВЛЕЧЕНИЕ ТЕКСТА ИЗ PDF
# =====================================================

def extract_text_from_pdf_with_pages(file_path: str) -> List[Dict]:
    """Извлекает текст из PDF с сохранением номеров страниц."""
    pages = []
    
    try:
        doc = fitz.open(file_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            text = re.sub(r'\s+', ' ', text).strip()
            
            if text and len(text) >= 50:
                pages.append({
                    "page_num": page_num + 1,
                    "text": text
                })
        
        doc.close()
        
    except Exception as e:
        print(f"   [ERROR] Ошибка чтения PDF: {e}")
    
    return pages


# =====================================================
# ЧАНКИНГ С МЕТАДАННЫМИ
# =====================================================

def split_text_into_chunks(
    text: str,
    source_doc: str,
    page_num: int,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    global_chunk_offset: int = 0
) -> tuple:
    """Разбивает текст на чанки с метаданными."""
    chunks = []
    start = 0
    chunk_id = global_chunk_offset
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        if end < len(text):
            search_from = start + int(chunk_size * 0.7)
            best_break = -1
            
            for sep in ['. ', '.\n', '! ', '!\n', '? ', '?\n', ';\n', '\n\n', '\n']:
                pos = text.rfind(sep, search_from, end)
                if pos != -1 and pos + len(sep) > best_break:
                    best_break = pos + len(sep)
            
            if best_break > search_from:
                end = best_break
        
        chunk_text = text[start:end].strip()
        
        if len(chunk_text) >= MIN_CHUNK_LENGTH:
            chunks.append({
                "text": chunk_text,
                "chunk_id": f"{source_doc}_page{page_num}_chunk{chunk_id}",
                "source_doc": source_doc,
                "page_num": page_num,
                "chunk_index_in_doc": chunk_id,
                "start_char": start,
                "end_char": end
            })
            chunk_id += 1
        
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end
        start = next_start
    
    return chunks, chunk_id


# =====================================================
# СОЗДАНИЕ ДАТАСЕТА
# =====================================================

def load_resume_state() -> Dict:
    if os.path.exists(RESUME_FILE):
        try:
            with open(RESUME_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"processed_docs": [], "test_cases": [], "completed": False}


def save_resume_state(state: Dict):
    os.makedirs(os.path.dirname(RESUME_FILE) or ".", exist_ok=True)
    with open(RESUME_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def prepare_test_dataset(
    pdf_folder: str = PDF_FOLDER,
    output_file: str = OUTPUT_FILE,
    chunks_per_doc: int = CHUNKS_PER_DOCUMENT,
    resume: bool = True
) -> Dict:
    """
    Создаёт тестовый датасет.
    
    Логика: из каждого документа берётся ровно chunks_per_doc случайных чанков.
    """
    
    print("\n" + "="*70)
    print("ПОДГОТОВКА ТЕСТОВОГО ДАТАСЕТА ДЛЯ RAG")
    print(f"Режим: {chunks_per_doc} случайных чанков из каждого документа")
    print("="*70 + "\n")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    oauth_token = os.getenv("OAUTH_TOKEN")
    catalog_id = os.getenv("CATALOG_ID")
    
    if not oauth_token or not catalog_id:
        raise ValueError("Не найдены OAUTH_TOKEN или CATALOG_ID в .env файле")
    
    llm_client = LLMClient(oauth_token, catalog_id)
    print("LLM клиент инициализирован")
    
    pdf_files = sorted([f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')])
    
    if not pdf_files:
        raise FileNotFoundError(f"В папке {pdf_folder} не найдено PDF-файлов")
    
    print(f"Найдено PDF-файлов: {len(pdf_files)}")
    for f in pdf_files:
        print(f"   - {f}")
    print()
    
    state = load_resume_state() if resume else {"processed_docs": [], "test_cases": [], "completed": False}
    
    if resume and state["test_cases"]:
        print(f"Найдено сохранённое состояние: {len(state['test_cases'])} вопросов уже создано\n")
    
    all_test_cases = state["test_cases"]
    processed_docs = set(state["processed_docs"])
    
    for doc_idx, filename in enumerate(pdf_files, 1):
        if filename in processed_docs:
            print(f"[{doc_idx}/{len(pdf_files)}] Пропущен (уже обработан): {filename}")
            continue
        
        print(f"\n[{doc_idx}/{len(pdf_files)}] Обработка: {filename}")
        
        filepath = os.path.join(pdf_folder, filename)
        doc_name = filename.replace('.pdf', '').replace('.PDF', '')
        
        pages = extract_text_from_pdf_with_pages(filepath)
        print(f"   Страниц с текстом: {len(pages)}")
        
        if not pages:
            print(f"   [WARN] Нет текста в документе")
            processed_docs.add(filename)
            save_resume_state({
                "processed_docs": list(processed_docs),
                "test_cases": all_test_cases,
                "completed": False
            })
            continue
        
        # Чанкинг всех страниц
        doc_chunks = []
        global_chunk_offset = 0
        
        for page in pages:
            chunks, global_chunk_offset = split_text_into_chunks(
                text=page["text"],
                source_doc=doc_name,
                page_num=page["page_num"],
                global_chunk_offset=global_chunk_offset
            )
            doc_chunks.extend(chunks)
        
        print(f"   Создано чанков: {len(doc_chunks)}")
        
        if not doc_chunks:
            print(f"   [WARN] Нет чанков в документе")
            processed_docs.add(filename)
            save_resume_state({
                "processed_docs": list(processed_docs),
                "test_cases": all_test_cases,
                "completed": False
            })
            continue
        
        # === КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: ровно chunks_per_doc случайных чанков ===
        if len(doc_chunks) >= chunks_per_doc:
            selected_chunks = random.sample(doc_chunks, chunks_per_doc)
        else:
            print(f"   [WARN] В документе только {len(doc_chunks)} чанков, берём все")
            selected_chunks = doc_chunks
        
        print(f"   Отобрано случайных чанков: {len(selected_chunks)}")
        
        # Генерация вопросов через LLM
        for chunk_idx, chunk in enumerate(selected_chunks, 1):
            print(f"   [{chunk_idx}/{len(selected_chunks)}] Генерация QA...")
            
            qa = llm_client.generate_qa(chunk["text"], doc_name)
            
            if qa:
                test_case = {
                    "id": f"test_{len(all_test_cases) + 1:03d}",
                    "question": qa["question"],
                    "expected_answer": qa["expected_answer"],
                    "reference_chunk_id": chunk["chunk_id"],
                    "reference_text": chunk["text"],
                    "reference_source": chunk["source_doc"],
                    "reference_page": chunk["page_num"],
                    "topic": qa.get("topic", "general"),
                    "difficulty": qa.get("difficulty", "medium"),
                    "generated_at": datetime.now().isoformat(),
                    "generation_model": "llm-api"
                }
                
                all_test_cases.append(test_case)
                print(f"      Вопрос: {qa['question'][:70]}...")
            else:
                print(f"      [SKIP] Пропущен (низкое качество или ошибка)")
            
            time.sleep(API_DELAY)
            
            save_resume_state({
                "processed_docs": list(processed_docs),
                "test_cases": all_test_cases,
                "completed": False
            })
        
        processed_docs.add(filename)
    
    # Финальное сохранение
    dataset = {
        "metadata": {
            "domain": "sanpin_regulations",
            "created_at": datetime.now().isoformat(),
            "num_test_cases": len(all_test_cases),
            "source_files": pdf_files,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunks_per_document": chunks_per_doc,
            "generation_model": "llm-api",
            "question_focus": "urban_planning_analysis",
            "evaluation_metrics": {
                "search": ["hit_rate_at_5", "mrr"],
                "generation": ["sbert_score", "rouge_l_f1"]
            }
        },
        "test_cases": all_test_cases
    }
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    if os.path.exists(RESUME_FILE):
        os.remove(RESUME_FILE)
    
    # Вывод статистики
    print("\n" + "="*70)
    print("ДАТАСЕТ СОЗДАН")
    print("="*70)
    print(f"Обработано документов: {len(processed_docs)}")
    print(f"Чанков на документ: {chunks_per_doc}")
    print(f"Ожидаемо вопросов: {len(processed_docs) * chunks_per_doc}")
    print(f"Создано вопросов: {len(all_test_cases)}")
    print(f"Сохранено в: {output_file}")
    
    topics = {}
    difficulties = {}
    for case in all_test_cases:
        topic = case.get("topic", "unknown")
        diff = case.get("difficulty", "unknown")
        topics[topic] = topics.get(topic, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    print(f"\nТемы: {topics}")
    print(f"Сложность: {difficulties}")
    
    if all_test_cases:
        print(f"\nПример тестового кейса:")
        example = all_test_cases[0]
        print(f"   ID: {example['id']}")
        print(f"   Вопрос: {example['question'][:80]}...")
        print(f"   Чанк: {example['reference_chunk_id']}")
        print(f"   Страница: {example['reference_page']}")
    
    return dataset


if __name__ == "__main__":
    try:
        dataset = prepare_test_dataset(
            pdf_folder=PDF_FOLDER,
            output_file=OUTPUT_FILE,
            chunks_per_doc=CHUNKS_PER_DOCUMENT,
            resume=True
        )
        print("\nГотово! Теперь запустите:")
        print("   1. python rag_server.py  (в отдельном терминале)")
        print("   2. python rag_evaluator.py  (для тестирования)")
    except Exception as e:
        print(f"\n[ERROR] Ошибка: {e}")
        raise