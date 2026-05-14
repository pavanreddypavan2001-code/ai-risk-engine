import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = "llama3.2:1b"

SYSTEM_PROMPT = (
    "You are a senior credit risk analyst at a financial institution. "
    "You produce structured, business-ready risk assessments from raw financial data. "
    "Always respond with valid JSON only — no markdown, no explanation outside the JSON."
)

PROMPT_TEMPLATE = """You are a credit risk analyst. Read the financial text below and answer the question.

Financial Text:
{context}

Question: {query}

Reply ONLY with this JSON (no extra text):
{{
  "executive_summary": "2-3 sentences summarising the financial situation and risk.",
  "key_risks": ["risk factor 1", "risk factor 2", "risk factor 3"],
  "recommendation": "1-2 sentences on what an analyst should do next.",
  "risk_score": 50,
  "risk_level": "Moderate",
  "confidence": "Medium"
}}

Replace the placeholder values with your actual analysis. risk_score must be an integer 0-100. risk_level must be Low, Moderate, High, or Critical."""


def _empty_response():
    return {
        "executive_summary": "Insufficient financial data was found in the uploaded documents to answer this question.",
        "key_risks": [],
        "recommendation": "Upload a document containing the relevant financial statements and re-run the analysis.",
        "risk_score": None,
        "risk_level": None,
        "confidence": "Low",
    }


def _generate_groq(prompt):
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        },
    )
    response.raise_for_status()
    return json.loads(response.json()["choices"][0]["message"]["content"])


def _generate_gemini(prompt):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
        ),
        contents=prompt,
    )
    return json.loads(response.text)


def _generate_ollama(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
        "stream": False,
        "format": "json",
    })
    response.raise_for_status()
    return json.loads(response.json()["response"])


def active_model():
    if GROQ_API_KEY:
        return f"Groq · {GROQ_MODEL}"
    if GOOGLE_API_KEY:
        return f"Gemini · {GEMINI_MODEL} (Ollama fallback)"
    return f"Ollama · {OLLAMA_MODEL}"


def generate_response(query, chunks):
    if not chunks:
        return _empty_response()

    context = "\n\n".join(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    if GROQ_API_KEY:
        return _generate_groq(prompt)
    if GOOGLE_API_KEY:
        try:
            return _generate_gemini(prompt)
        except Exception:
            pass  # quota exhausted or unavailable — fall through to Ollama
    return _generate_ollama(prompt)
