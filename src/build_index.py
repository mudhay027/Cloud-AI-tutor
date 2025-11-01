import os
import re
from PyPDF2 import PdfReader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------- CONFIG ----------
PDF_FOLDER = "../staff_pdfs"        # Folder containing all staff PDFs
INDEX_FILE = "../config/pdf_index.faiss"   # Output FAISS index
METADATA_FILE = "../config/metadata.txt"   # Metadata mapping for chunks
CHUNK_SIZE = 800                 # ~ number of characters per chunk
CHUNK_OVERLAP = 200              # overlap between chunks
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# ----------------------------

def extract_text_from_pdf(pdf_path):
    """Extract text from a text-based PDF."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        try:
            text += page.extract_text() + "\n"
        except Exception:
            continue
    return text

def clean_text(text):
    """Clean and normalize text."""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\x00', '')
    return text.strip()

def chunk_text(text, chunk_size=800, overlap=200):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def build_index():
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = []
    metadata = []

    print(f"📄 Reading PDFs from '{PDF_FOLDER}' ...")
    for filename in tqdm(os.listdir(PDF_FOLDER)):
        if not filename.lower().endswith(".pdf"):
            continue

        path = os.path.join(PDF_FOLDER, filename)
        text = extract_text_from_pdf(path)
        text = clean_text(text)

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        print(f"→ {filename}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            # Skip if the chunk is not a valid string or empty
            if not isinstance(chunk, str) or not chunk.strip():
                print(f"⚠️ Skipping empty or invalid chunk {i} in {filename}")
                continue

            try:
                emb = model.encode([chunk])[0]
                embeddings.append(emb)
                metadata.append(f"{filename}|chunk_{i}")
            except Exception as e:
                print(f"⚠️ Skipped chunk {i} from {filename}: {e}")
                continue


    # Convert to numpy array
    embeddings = np.vstack(embeddings).astype("float32")

    # Create FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)

    # Save metadata
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(m + "\n")

    print(f"✅ Index built and saved: {INDEX_FILE}")
    print(f"✅ Metadata saved: {METADATA_FILE}")
    print(f"Total chunks indexed: {len(metadata)}")

if __name__ == "__main__":
    build_index()
