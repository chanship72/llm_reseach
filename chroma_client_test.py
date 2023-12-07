import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
import json

client = chromadb.PersistentClient(path="./TSLA_vector")

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

embeddings = embedding.embed_documents(
    ["This is a query document"]
)

collection = client.get_collection(name="TSLA", embedding_function=embedding)
print(collection.count())

print(collection.query(
    query_embeddings=embeddings,
    n_results=2,
))
# print(collection.get(
#     ids=['7', '10'], 
#     where={}
# ))

# indexDB = collection.get()

# print(indexDB['metadatas'])
# print(collection.peek())
# results = collection.query(query_texts=["This is a query document"], n_results=2)
# print(results)
