import json
import os
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = "llama3.2:1b"

SYSTEM_PROMPT = (
    "You are a senior credit risk analyst at a financial institution. "
    "You interpret raw financial data and produce concise, business-ready assessments. "
    "Always respond with valid JSON only — no markdown, no explanation outside the JSON."
)

PROMPT_TEMPLATE = """Using the financial document excerpts below, answer the analyst's question.

Rules:
- Translate raw numbers into business meaning (e.g. "debt-to-equity of 1.8x indicates moderate leverage").
- Write 2-4 sentences in a professional tone — no bullet lists.
- Assign a credit risk score from 0 (no risk) to 100 (maximum risk).
- Map the score to a risk level: Low (0-39), Moderate (40-64), High (65-84), Critical (85-100).

Financial Context:
{context}

Analyst Question: {query}

Respond with this JSON structure:
{{
  "answer": "<business-style interpretation>",
  "risk_score": <integer 0-100>,
  "risk_level": "<Low | Moderate | High | Critical>"
}}"""


def generate_response(query, chunks):
    if not chunks:
        return {
            "answer": "No relevant financial information found in the uploaded documents.",
            "risk_score": None,
            "risk_level": None,
        }

    context = "\n\n".join(chunks)
    prompt = f"{SYSTEM_PROMPT}\n\n{PROMPT_TEMPLATE.format(context=context, query=query)}"

    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    })
    response.raise_for_status()

    return json.loads(response.json()["response"])
