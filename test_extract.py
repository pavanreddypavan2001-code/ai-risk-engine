from service.engine.text_extractor import extract_text_from_pdf
from service.engine.chunker import chunk_text
from service.engine.vector_store import store_chunks

pdf_path = "Tesla_Financial_Report.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text)

stored = store_chunks(chunks)

print("Chunks Stored:", stored)