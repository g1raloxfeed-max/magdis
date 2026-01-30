from elasticsearch import Elasticsearch
es = Elasticsearch("http://localhost:9200")  # Используем HTTP
print(es.info())