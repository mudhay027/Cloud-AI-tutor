import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader
import re
import torch
import os
import time
import google.generativeai as genai
import subprocess
import hashlib
import threading
from dotenv import load_dotenv

# ---------- CONFIG ----------
INDEX_FILE = "../config/pdf_index.faiss"
METADATA_FILE = "../config/metadata.txt"
PDF_FOLDER = "../staff_pdfs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 6
MAX_CONTEXT_CHARS = 3500
# ----------------------------

# ---------- Gemini Setup ----------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# ---------- Helper: PDF Hash for Auto Rebuild ----------
def get_pdf_hashes():
    pdfs = [os.path.join(PDF_FOLDER, f) for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    hash_data = ""
    for pdf in sorted(pdfs):
        mtime = os.path.getmtime(pdf)
        hash_data += f"{pdf}-{mtime}"
    return hashlib.md5(hash_data.encode()).hexdigest()

def needs_rebuild():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        print("⚠️ Index or metadata missing — will rebuild.")
        return True
    hash_file = "pdf_hash.txt"
    current_hash = get_pdf_hashes()
    if not os.path.exists(hash_file):
        print("⚠️ No previous hash found — will rebuild index.")
        return True
    with open(hash_file, "r") as f:
        old_hash = f.read().strip()
    if current_hash != old_hash:
        print("🆕 PDF files changed — rebuilding FAISS index...")
        return True
    return False

def update_hash():
    with open("pdf_hash.txt", "w") as f:
        f.write(get_pdf_hashes())

# ---------- Auto Rebuild ----------
if needs_rebuild():
    print("🔄 Running build_index.py to update FAISS index...\n")
    subprocess.run(["python", "build_index.py"], check=True)
    update_hash()
    print("✅ Index updated successfully!\n")
else:
    print("✅ FAISS index is up to date.\n")

# ---------- Load Embeddings ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔄 Loading embeddings on {device}...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("✅ Embedding model loaded.\n")

index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = [line.strip() for line in f.readlines()]

# ---------- PDF List ----------
def refresh_pdf_list():
    return sorted([f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")])

pdf_files = refresh_pdf_list()
active_pdf = 0  # 0 = All PDFs

def show_pdf_list():
    print("\n📂 Available PDFs:")
    print("  0. 🔍 Search across ALL PDFs (Default)\n")
    for i, pdf in enumerate(pdf_files, start=1):
        mark = "✅" if active_pdf == i else ""
        print(f"  {i}. {pdf} {mark}")

# ---------- Retrieve Chunks ----------
def retrieve_relevant_chunks(query, top_k=TOP_K):
    qv = embedder.encode([query]).astype("float32")
    distances, indices = index.search(qv, top_k)
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            pdf_name, chunk_id = metadata[idx].split("|")
            if active_pdf == 0 or pdf_files[active_pdf - 1] == pdf_name:
                results.append((pdf_name, chunk_id))
    return results

# ---------- Extract Chunk Text ----------
def get_chunk_text(pdf_name, chunk_index, chunk_size=800, overlap=200):
    path = f"{PDF_FOLDER}/{pdf_name}"
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    text = re.sub(r'\s+', ' ', text)
    start = chunk_index * (chunk_size - overlap)
    end = start + chunk_size
    return text[start:end]

# ---------- Background Watcher ----------
def watch_for_pdf_changes(interval=15):
    global pdf_files
    last_hash = get_pdf_hashes()
    while True:
        time.sleep(interval)
        try:
            current_hash = get_pdf_hashes()
            if current_hash != last_hash:
                print("\n🆕 Detected PDF change! Rebuilding FAISS index automatically...\n")
                subprocess.run(["python", "build_index.py"], check=True)
                update_hash()

                global index, metadata
                index = faiss.read_index(INDEX_FILE)
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    metadata = [line.strip() for line in f.readlines()]

                pdf_files = refresh_pdf_list()
                print("✅ Index rebuilt, reloaded, and PDF list updated!\n")
                last_hash = current_hash
        except Exception as e:
            print(f"⚠️ Error while watching PDFs: {e}")

# ---------- Generate Answer ----------
def generate_answer(question, mode="normal", last_answer=None):
    retrieved = retrieve_relevant_chunks(question)
    if not retrieved:
        print("⚠️ No relevant content found in the selected PDF(s).")
        return ""

    # Collect context + source PDFs
    context = ""
    source_pdfs = set()
    for pdf_name, chunk_id in retrieved:
        chunk_no = int(chunk_id.split("_")[1])
        context += get_chunk_text(pdf_name, chunk_no) + " "
        source_pdfs.add(pdf_name)
    context = context[:MAX_CONTEXT_CHARS]
    source_display = ", ".join(sorted(source_pdfs))

    if mode == "modify" and last_answer:
        prompt = f"""
You are an AI tutor. You must follow the user’s editing request
but strictly use the original staff-provided answer and PDF context.

Context (staff notes from {source_display}):
{context}

Original Answer:
{last_answer}

User Request: {question}

Now generate a revised version that follows the user's request.
Do NOT add any information outside the staff notes.
        """
    else:
        prompt = f"""
You are a college AI tutor. Use ONLY the information from the context below
(which comes from staff-provided notes). Do NOT use any external knowledge.

Write a clear, simple-English answer for the question. Adjust length if user mentions "2 marks", "13 marks", etc.

Context (from {source_display}):
{context}

Question: {question}

Answer:
        """

    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    final_answer = response.text.strip()

    print(f"\n📘 Answer from: {source_display}\n")
    print(final_answer)
    print("\n" + "-" * 80 + "\n")
    return final_answer

# ---------- CLI ----------
if __name__ == "__main__":
    print("✅ AI Tutor Ready (Hybrid Mode)\n")
    print("📂 Watching for PDF updates in real time...\n")

    watcher_thread = threading.Thread(target=watch_for_pdf_changes, daemon=True)
    watcher_thread.start()

    show_pdf_list()
    print("Commands: 'switch <num>', 'list', 'current', 'exit'\n")

    last_answer = None

    while True:
        active_name = pdf_files[active_pdf-1] if active_pdf else "ALL PDFs"
        q = input(f"\n[{active_name}] Ask or command: ").strip()
        if q.lower() == "exit":
            break
        if q.lower() == "list":
            show_pdf_list()
            continue
        if q.lower() == "current":
            print(f"📘 Currently active: {active_name}")
            continue
        if q.lower().startswith("switch"):
            try:
                num = int(q.split()[1])
                if 0 <= num <= len(pdf_files):
                    active_pdf = num
                    print("\n✅ Switched to:",
                          "ALL PDFs" if active_pdf == 0 else pdf_files[active_pdf - 1])
                    show_pdf_list()
                else:
                    print("⚠️ Invalid PDF number.")
            except:
                print("⚠️ Usage: switch <num> (example: switch 1)")
            continue
        if any(word in q.lower() for word in ["short", "expand", "simplify", "example", "rewrite", "again"]):
            if last_answer:
                last_answer = generate_answer(q, mode="modify", last_answer=last_answer)
            else:
                print("⚠️ No previous answer found to modify.")
        else:
            last_answer = generate_answer(q, mode="normal")
