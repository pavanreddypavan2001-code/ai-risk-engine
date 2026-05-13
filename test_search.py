from service.engine.text_extractor import extract_text_from_pdf
from service.engine.chunker import chunk_text
from service.engine.vector_store import store_chunks, search_chunks

pdf_path = "Tesla_Financial_Report.pdf"

# Step 1: Extract
text = extract_text_from_pdf(pdf_path)

# Step 2: Chunk
chunks = chunk_text(text)

# Step 3: Store
store_chunks(chunks)

# Step 4: Search
query = "What is Tesla's financial performance?"

results = search_chunks(query)

print("\nRelevant Results:\n")

for result in results:
    print(result)
    print("\n" + "=" * 80 + "\n")