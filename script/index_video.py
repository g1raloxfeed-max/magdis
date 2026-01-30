import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from elasticsearch import Elasticsearch, helpers
import time
import sys
import os

# =================== КОНФИГУРАЦИЯ ===================
VIDEO_FILE_PATH = "путь_к_вашему_видеофайлу.mp4"  # <-- ЗАМЕНИТЕ НА СВОЙ ПУТЬ
ELASTICSEARCH_HOST = "http://localhost:9200"
INDEX_NAME = "video_frames"
FRAME_EXTRACTION_INTERVAL = 1  # Брать кадр каждую секунду
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # Базовая модель, хороший баланс скорости/качества
# ===================================================

# 1. ИНИЦИАЛИЗАЦИЯ КЛИЕНТОВ
print("🔄 Инициализация клиентов...")

# Инициализация Elasticsearch
es = Elasticsearch(ELASTICSEARCH_HOST)
if not es.ping():
    print("❌ Не удалось подключиться к Elasticsearch!")
    sys.exit(1)

# Инициализация CLIP модели (загрузка на GPU, если доступен)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Используется устройство: {device}")

model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
model.eval()  # Переводим модель в режим инференса
print(f"✅ Загружена модель: {CLIP_MODEL_NAME}")

# 2. СОЗДАНИЕ ИНДЕКСА (ЕСЛИ НЕ СУЩЕСТВУЕТ)
index_mapping = {
    "mappings": {
        "properties": {
            "video_name": {"type": "keyword"},
            "frame_number": {"type": "integer"},
            "timestamp_sec": {"type": "float"},
            "frame_vector": {
                "type": "dense_vector",
                "dims": 512,  # Размерность вектора для clip-vit-base-patch32
                "index": True,
                "similarity": "cosine"
            },
            "file_path": {"type": "keyword"}
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    }
}

if not es.indices.exists(index=INDEX_NAME):
    es.indices.create(index=INDEX_NAME, body=index_mapping)
    print(f"✅ Создан индекс: {INDEX_NAME}")
else:
    print(f"ℹ️  Индекс '{INDEX_NAME}' уже существует")

# 3. ОБРАБОТКА ВИДЕО И ИНДЕКСАЦИЯ
print(f"\n🎥 Начинаю обработку видео: {VIDEO_FILE_PATH}")

# Проверка существования файла
if not os.path.exists(VIDEO_FILE_PATH):
    print(f"❌ Файл не найден: {VIDEO_FILE_PATH}")
    sys.exit(1)

# Открываем видео
cap = cv2.VideoCapture(VIDEO_FILE_PATH)
if not cap.isOpened():
    print("❌ Не удалось открыть видеофайл!")
    sys.exit(1)

# Получаем параметры видео
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps
video_name = os.path.basename(VIDEO_FILE_PATH)

print(f"📊 Информация о видео:")
print(f"   • FPS: {fps:.2f}")
print(f"   • Всего кадров: {total_frames}")
print(f"   • Длительность: {duration:.2f} сек")
print(f"   • Кадров для обработки: ~{int(duration / FRAME_EXTRACTION_INTERVAL)}")

# Функция для подготовки батчей данных
def generate_documents():
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Извлекаем кадры с заданным интервалом
        current_time_sec = frame_count / fps
        if frame_count % int(fps * FRAME_EXTRACTION_INTERVAL) == 0:
            
            # Конвертируем кадр из BGR (OpenCV) в RGB (для CLIP)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Подготовка изображения для модели CLIP
            inputs = processor(images=frame_rgb, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Векторизация кадра
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            
            # Конвертируем в numpy массив и нормализуем вектор
            vector = image_features.cpu().numpy().flatten().astype('float32')
            
            # Создаем документ для Elasticsearch
            doc = {
                "_index": INDEX_NAME,
                "_source": {
                    "video_name": video_name,
                    "frame_number": frame_count,
                    "timestamp_sec": round(current_time_sec, 2),
                    "frame_vector": vector.tolist(),  # Конвертируем в список для JSON
                    "file_path": VIDEO_FILE_PATH
                }
            }
            
            processed_count += 1
            if processed_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"   Обработано кадров: {processed_count} ({(processed_count/elapsed):.1f} кадр/сек)")
            
            yield doc
        
        frame_count += 1
    
    cap.release()
    print(f"\n✅ Обработка завершена!")
    print(f"   • Всего просмотрено кадров: {frame_count}")
    print(f"   • Векторизовано кадров: {processed_count}")
    print(f"   • Общее время: {time.time() - start_time:.1f} сек")

# 4. ИНДЕКСАЦИЯ В ELASTICSEARCH С ИСПОЛЬЗОВАНИЕМ BULK API
print("\n📤 Отправка данных в Elasticsearch...")
try:
    # Используем bulk-индексацию для эффективной загрузки
    success, failed = helpers.bulk(
        es,
        generate_documents(),
        chunk_size=50,  # Размер пачки для отправки
        request_timeout=30,
        max_retries=3
    )
    
    print(f"✅ Индексация завершена!")
    print(f"   • Успешно: {success} документов")
    print(f"   • Ошибок: {len(failed) if failed else 0}")
    
    # Выводим статистику индекса
    if success > 0:
        es.indices.refresh(index=INDEX_NAME)
        count = es.count(index=INDEX_NAME)['count']
        print(f"   • Всего в индексе: {count} документов")
        
except Exception as e:
    print(f"❌ Ошибка при индексации: {e}")
    cap.release()

# 5. ФИНАЛЬНАЯ СТАТИСТИКА
print("\n" + "="*50)
print("ИНДЕКСАЦИЯ ЗАВЕРШЕНА")
print("="*50)
print(f"Индекс: {INDEX_NAME}")
print(f"Видеофайл: {video_name}")
print(f"Сервер Elasticsearch: {ELASTICSEARCH_HOST}")