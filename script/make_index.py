from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

index_name = "video_frames"

index_mapping = {
    "mappings": {
        "properties": {
            "video_name": {"type": "keyword"},
            "timestamp_sec": {"type": "float"},  # Таймкод кадра в секундах
            "frame_vector": {  # Векторное представление кадра от CLIP
                "type": "dense_vector",
                "dims": 512,  # Размерность вектора модели CLIP (например, 512 для clip-vit-base-patch32)
                "index": True,
                "similarity": "cosine"  # Используем косинусное расстояние для поиска
            },
            "scene_description": {"type": "text"}  # Опционально: текстовое описание сцены
        }
    }
}

# Создаем индекс, если он еще не существует
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=index_mapping)
    print(f"✅ Индекс '{index_name}' создан.")
else:
    print(f"ℹ️  Индекс '{index_name}' уже существует.")