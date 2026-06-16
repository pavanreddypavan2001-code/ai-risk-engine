import json
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from service.engine.text_extractor import extract_text
from service.engine.chunker import chunk_text
from service.engine.vector_store import embed_chunks, add_to_store, search_chunks
from service.engine.response_generator import generate_response

router = APIRouter(prefix="/pipeline")


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@router.post("/upload")
async def pipeline_upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()

    async def stream():
        # Stage 1: Extract
        yield sse({"stage": "extract", "status": "running", "message": "Reading PDF pages..."})
        text = await asyncio.to_thread(extract_text, content)
        if not text.strip():
            yield sse({"stage": "extract", "status": "error", "message": "Could not extract text from PDF."})
            return
        yield sse({
            "stage": "extract", "status": "done",
            "message": f"Extracted {len(text):,} characters",
            "preview": text[:200],
        })

        # Stage 2: Chunk
        yield sse({"stage": "chunk", "status": "running", "message": "Splitting into 3000-char slices..."})
        chunks = await asyncio.to_thread(chunk_text, text)
        yield sse({
            "stage": "chunk", "status": "done",
            "message": f"Created {len(chunks)} chunks",
            "count": len(chunks),
        })

        # Stage 3: Embed
        yield sse({"stage": "embed", "status": "running", "message": f"Embedding {len(chunks)} chunks via Google API..."})
        try:
            embeddings = await asyncio.to_thread(embed_chunks, chunks)
        except Exception as e:
            yield sse({"stage": "embed", "status": "error", "message": f"Embedding failed: {e}"})
            return
        yield sse({
            "stage": "embed", "status": "done",
            "message": f"{len(embeddings)} vectors ready",
        })

        # Stage 4: Store
        yield sse({"stage": "store", "status": "running", "message": "Storing vectors in memory..."})
        try:
            await asyncio.to_thread(add_to_store, chunks, embeddings)
        except Exception as e:
            yield sse({"stage": "store", "status": "error", "message": f"Store failed: {e}"})
            return
        yield sse({
            "stage": "store", "status": "done",
            "message": f"{len(chunks)} vectors stored",
            "chunks_stored": len(chunks),
        })

        yield sse({"stage": "complete", "status": "done"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class AskPayload(BaseModel):
    query: str


@router.post("/ask")
async def pipeline_ask(payload: AskPayload):
    async def stream():
        # Stage 1: Search
        yield sse({"stage": "search", "status": "running", "message": "Embedding query + searching vectors..."})
        chunks = await asyncio.to_thread(search_chunks, payload.query)
        yield sse({
            "stage": "search", "status": "done",
            "message": f"Found {len(chunks)} relevant chunks",
            "chunks": chunks,
        })

        # Stage 2: Generate
        yield sse({"stage": "generate", "status": "running", "message": "Generating structured risk assessment..."})
        try:
            result = await asyncio.to_thread(generate_response, payload.query, chunks)
        except Exception as e:
            yield sse({"stage": "generate", "status": "error", "message": str(e)})
            return

        verdict = result.get("verdict", "Neutral")
        score   = result.get("score", result.get("risk_score", 50))
        yield sse({
            "stage": "generate", "status": "done",
            "message": "Assessment complete",
            "executive_summary": result.get("executive_summary"),
            "key_points": result.get("key_points", result.get("key_risks", [])),
            "recommendation": result.get("recommendation"),
            "verdict": verdict,
            "score": score,
            "confidence": result.get("confidence"),
        })

        yield sse({"stage": "complete", "status": "done"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
