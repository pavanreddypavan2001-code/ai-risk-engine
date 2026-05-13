from fastapi import APIRouter, UploadFile, File, HTTPException

from service.engine.text_extractor import extract_text
from service.engine.chunker import chunk_text
from service.engine.vector_store import store_chunks

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()

    text = extract_text(content)
    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    chunks = chunk_text(text)
    store_chunks(chunks)

    return {
        "message": "uploaded successfully",
        "chunks_stored": len(chunks)
    }