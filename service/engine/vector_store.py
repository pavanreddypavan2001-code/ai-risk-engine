from rank_bm25 import BM25Okapi

_docs: list[str] = []
_bm25: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def embed_chunks(chunks):
    return [[] for _ in chunks]


def add_to_store(chunks, _=None):
    global _bm25, _docs
    _docs.extend(chunks)
    _bm25 = BM25Okapi([_tokenize(d) for d in _docs])


def store_chunks(chunks):
    add_to_store(chunks)
    return len(chunks)


def search_chunks(query, n=3):
    if not _docs or _bm25 is None:
        return []
    scores = _bm25.get_scores(_tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:min(n, len(_docs))]
    return [_docs[i] for i in top]
