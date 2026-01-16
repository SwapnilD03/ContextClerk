import openai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from pdf2image import convert_from_path
import pytesseract
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(
  base_url="http://localhost:11434/v1",
  api_key="ollama",
  timeout=300.0
)

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768

splitter = SentenceSplitter(chunk_size=1000,chunk_overlap=200)

def get_splitter(num_pages:int) -> SentenceSplitter:
    """
    Choose chunk size dynamically based on the length of the document

    """
    if num_pages <= 20:
        return SentenceSplitter(chunk_size=800, chunk_overlap=200)
    elif num_pages <= 100:
        return SentenceSplitter(chunk_size=1200, chunk_overlap=200)
    else:
        return SentenceSplitter(chunk_size=1600, chunk_overlap=300)
    

def ocr_pdf(path: str) -> list[str]:
    """
    Perform OCR on scanned PDFs

    """
    images = convert_from_path(path)
    texts = []
    for img in images:
        text = pytesseract.image_to_string(img)
        if text.strip():
            texts.append(text)
    return texts


def load_and_chunk_pdf(path:str):
    """
    Load a PDF and return dynamically sized text chunks.
    OCR fallback is used if no extractable text is found.
    """

    docs = PDFReader().load_data(file=path)

    texts = [d.text for d in docs if getattr(d,"text",None)]

    if not texts:
        texts = ocr_pdf(path)
        
    chunks = []

    for page_idx, doc in enumerate(docs):
        if not getattr(doc, "text", None):
            continue

        page_chunks = splitter.split_text(doc.text)
        for c in page_chunks:
            chunks.append({
                "text": c,
                "page": page_idx + 1  # 1-based indexing
            })

    return chunks




def embed_texts(texts:list[str]) -> list[list[float]]:

    response = client.embeddings.create(
        model = EMBED_MODEL,
        input=texts,
    )

    return [item.embedding for item in response.data]