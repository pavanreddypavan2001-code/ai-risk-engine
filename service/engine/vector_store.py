import uuid
import os
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
EMBED_MODEL = "text-embedding-004"

chroma = chromadb.Client()
collection = chroma.get_or_create_collection("financial_docs")


def _embed(texts: list[str]) -> list[list[float]]:
    result = _client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return [e.values for e in result.embeddings]


def embed_chunks(chunks):
    return _embed(chunks)


def add_to_store(chunks, embeddings):
    ids = [str(uuid.uuid4()) for _ in chunks]
    collection.add(embeddings=embeddings, documents=chunks, ids=ids)


def store_chunks(chunks):
    embeddings = _embed(chunks)
    add_to_store(chunks, embeddings)
    return len(chunks)


def search_chunks(query):
    if collection.count() == 0:
        return []

    query_embedding = _embed([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(2, collection.count()),
    )
    return results["documents"][0]
