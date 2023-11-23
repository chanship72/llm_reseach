import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
import json

client = chromadb.PersistentClient(path="./data")

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
collection = client.get_collection(name="", embedding_function=embeddings)
print(collection.count())

# print(collection.get(
#     ids=['1e1a0a19-7858-11ee-8a63-30d042ef4ace']
# ))

indexDB = collection.get()

print(indexDB['metadatas'])

# results = collection.query(query_texts=["This is a query document"],n_results=2)
# print(results)
