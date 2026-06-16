import os
import numpy as np
from google import genai
from dotenv import load_dotenv

load_dotenv()

_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
EMBED_MODEL = "text-embedding-004"

_docs: list[str] = []
_embeddings: list[list[float]] = []


def _embed(texts: list[str]) -> list[list[float]]:
    result = _client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return [e.values for e in result.embeddings]


def embed_chunks(chunks):
    return _embed(chunks)


def add_to_store(chunks, embeddings):
    _docs.extend(chunks)
    _embeddings.extend(embeddings)


def store_chunks(chunks):
    embeddings = _embed(chunks)
    add_to_store(chunks, embeddings)
    return len(chunks)


def search_chunks(query, n=2):
    if not _docs:
        return []

    q = np.array(_embed([query])[0])
    matrix = np.array(_embeddings)
    scores = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-9)

    top_n = min(n, len(_docs))
    indices = np.argsort(scores)[::-1][:top_n]
    return [_docs[i] for i in indices]
