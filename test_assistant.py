from service.engine.text_extractor import extract_text_from_pdf
from service.engine.chunker import chunk_text
from service.engine.vector_store import store_chunks, search_chunks
from service.engine.response_generator import generate_response

pdf_path = "Tesla_Financial_Report.pdf"

# Build knowledge base
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)
store_chunks(chunks)

# Ask question
query = "Summarize Tesla's financial performance"

retrieved = search_chunks(query)

# Generate answer
response = generate_response(query, retrieved)

print(response)