import uuid
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()

collection = client.get_or_create_collection("financial_docs")


def store_chunks(chunks):
    embeddings = model.encode(chunks).tolist()

    ids = [str(uuid.uuid4()) for _ in chunks]

    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids
    )

    return len(chunks)


def search_chunks(query):
    if collection.count() == 0:
        return []

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(2, collection.count())
    )

    return results["documents"][0]
