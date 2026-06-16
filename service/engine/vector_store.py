import os
import time
import numpy as np
from google import genai
from dotenv import load_dotenv

load_dotenv()

_client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={"api_version": "v1"},
)
EMBED_MODEL = "text-embedding-004"

_docs: list[str] = []
_embeddings: list[list[float]] = []


def _embed_one(text: str) -> list[float]:
    result = _client.models.embed_content(model=EMBED_MODEL, contents=text)
    emb = result.embeddings[0] if result.embeddings else result.embedding
    return emb.values


def _embed(texts: list[str]) -> list[list[float]]:
    out = []
    for i, text in enumerate(texts):
        out.append(_embed_one(text))
        # pause every 20 calls to stay within free-tier rate limit (100 req/min)
        if (i + 1) % 20 == 0 and (i + 1) < len(texts):
            time.sleep(12)
    return out


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

    q = np.array(_embed_one(query))
    matrix = np.array(_embeddings)
    scores = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-9)

    top_n = min(n, len(_docs))
    indices = np.argsort(scores)[::-1][:top_n]
    return [_docs[i] for i in indices]
