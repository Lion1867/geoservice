import os
import re
import numpy as np
from collections import Counter
from pypdf import PdfReader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy import stats

# Загрузка ресурсов NLTK (нужно выполнить один раз)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') # Часто требуется для новых версий

def extract_text_from_pdfs(pdf_folder, output_txt_file):
    """
    Извлекает текст из всех PDF в папке и сохраняет в один TXT файл.
    """
    full_text = ""
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    
    print(f"Найдено PDF-файлов: {len(pdf_files)}")
    
    for filename in pdf_files:
        filepath = os.path.join(pdf_folder, filename)
        print(f"Обработка: {filename}...")
        try:
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # Базовая очистка: удаление лишних пробелов, переносов
            text = re.sub(r'\s+', ' ', text).strip()
            full_text += text + "\n\n" # Разделитель между документами
            
        except Exception as e:
            print(f"Ошибка при чтении {filename}: {e}")

    with open(output_txt_file, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"\nВсе тексты объединены в '{output_txt_file}'.")
    return full_text

def analyze_text(text):
    """
    Проводит статистический анализ текста.
    """
    print("\n--- ЗАПУСК СТАТИСТИЧЕСКОГО АНАЛИЗА ---")
    
    # 1. Токенизация (разбиение на слова и предложения)
    sentences = sent_tokenize(text, language='russian')
    words = word_tokenize(text, language='russian')
    
    # Фильтрация: оставляем только слова (без пунктуации и чисел)
    words_alpha = [w.lower() for w in words if w.isalpha()]
    
    # 2. Общие метрики
    total_chars = len(text)
    total_words = len(words_alpha)
    total_sentences = len(sentences)
    
    if total_words == 0:
        print("Текст пуст или не содержит слов.")
        return

    # 3. Анализ длин слов
    word_lengths = [len(w) for w in words_alpha]
    avg_word_len = np.mean(word_lengths)
    median_word_len = np.median(word_lengths)
    mode_word_len = stats.mode(word_lengths, keepdims=True)[0][0]
    
    # 4. Анализ длин предложений (в словах)
    # Считаем количество слов в каждом предложении
    sent_lengths = [len([w for w in word_tokenize(s) if w.isalpha()]) for s in sentences]
    avg_sent_len = np.mean(sent_lengths)
    median_sent_len = np.median(sent_lengths)
    try:
        mode_sent_len = stats.mode(sent_lengths, keepdims=True)[0][0]
    except:
        mode_sent_len = 0

    # 5. Частотный анализ (без стоп-слов)
    stop_words = set(stopwords.words('russian'))
    # Добавим специфичные для документов слова, которые не несут смысла
    custom_stops = {'г', 'ст', 'п', 'ч', 'n', 'рф', 'года', 'также', 'либо'} 
    stop_words.update(custom_stops)
    
    filtered_words = [w for w in words_alpha if w not in stop_words and len(w) > 2]
    word_counts = Counter(filtered_words)
    top_20 = word_counts.most_common(20)
    
    # 6. Лексическое разнообразие (TTR - Type-Token Ratio)
    unique_words = set(words_alpha)
    ttr = len(unique_words) / total_words

    # --- ВЫВОД РЕЗУЛЬТАТОВ ---
    print(f"\n=== ОПИСАТЕЛЬНАЯ СТАТИСТИКА ПО ДАТАСЕТУ ===")
    print(f"Общий объем текста: {total_chars:,} символов")
    print(f"Количество предложений: {total_sentences:,}")
    print(f"Количество слов (токенов): {total_words:,}")
    print(f"Количество уникальных слов: {len(unique_words):,}")
    print(f"Лексическое разнообразие (TTR): {ttr:.4f} (чем ближе к 1, тем богаче язык)")
    
    print(f"\n--- Статистика по словам ---")
    print(f"Средняя длина слова: {avg_word_len:.2f} символов")
    print(f"Медианная длина слова: {median_word_len} символов")
    print(f"Мода (самая частая длина): {mode_word_len} символов")
    
    print(f"\n--- Статистика по предложениям ---")
    print(f"Средняя длина предложения: {avg_sent_len:.2f} слов")
    print(f"Медианная длина предложения: {median_sent_len} слов")
    print(f"Мода (самая частая длина): {mode_sent_len} слов")
    
    print(f"\n--- ТОП-20 самых частых слов (тематика) ---")
    for word, count in top_20:
        print(f"{word}: {count}")

# --- НАСТРОЙКИ ---
PDF_DIR = "data"  # Папка, куда вы положили PDF файлы (СНиПы, ГОСТы и т.д.)
OUTPUT_FILE = "dataset_full.txt"

# Создадим папку для примера, если её нет
if not os.path.exists(PDF_DIR):
    os.makedirs(PDF_DIR)
    print(f"Папка '{PDF_DIR}' создана. Положите в неё PDF-файлы и запустите скрипт снова.")
else:
    # Запуск
    full_text = extract_text_from_pdfs(PDF_DIR, OUTPUT_FILE)
    if full_text:
        analyze_text(full_text)
        
        # Вывод первых примеров данных (как требовалось)
        print("\n\n=== ПЕРВЫЕ 500 СИМВОЛОВ СОБРАННЫХ ДАННЫХ (ПРИМЕР) ===")
        print(full_text[:500])