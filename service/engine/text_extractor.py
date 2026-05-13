from pypdf import PdfReader
from io import BytesIO


def extract_text(file_content: bytes) -> str:
    pdf = PdfReader(BytesIO(file_content))
    text = ""
    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def extract_text_from_pdf(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return extract_text(f.read())