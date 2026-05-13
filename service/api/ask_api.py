from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from service.engine.vector_store import search_chunks
from service.engine.response_generator import generate_response

router = APIRouter()

class Question(BaseModel):
    query: str


@router.post("/ask")
def ask_question(payload: Question):
    query = payload.query

    # Step 1: retrieve relevant chunks
    retrieved_chunks = search_chunks(query)

    # Step 2: generate response + risk score
    try:
        result = generate_response(query, retrieved_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "question": query,
        "answer": result.get("answer"),
        "risk_score": result.get("risk_score"),
        "risk_level": result.get("risk_level"),
        "sources": retrieved_chunks,
    }