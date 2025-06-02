from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import faiss
import numpy as np
import json
import os
import io
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account
from PyPDF2 import PdfReader
import re


load_dotenv()

app = FastAPI()

# ✅ CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://qu-demo-clipboardai.vercel.app"],  # change to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Transcript GCS setup
TRANSCRIPT_BUCKET = "transcript_puzzle"
TRANSCRIPT_JSON_PATH = "transcripts/transcript_chunks.json"

# ✅ PDF GCS setup
PDF_BUCKET = "puzzle_io"  # Change this
PDF_FOLDER = "pdf/"     # Change this (e.g., "pdf_chunks/")

# 🔐 GCP credentials
def get_credentials():
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path or not os.path.exists(key_path):
        raise RuntimeError("Valid GOOGLE_APPLICATION_CREDENTIALS path is required")
    return service_account.Credentials.from_service_account_file(key_path)

# 📄 Load transcript chunks
def load_transcript_chunks():
    creds = get_credentials()
    client = storage.Client(credentials=creds)
    bucket = client.bucket(TRANSCRIPT_BUCKET)
    blob = bucket.blob(TRANSCRIPT_JSON_PATH)
    content = blob.download_as_text()
    return json.loads(content)

# 📄 Load PDF chunks from another bucket/folder
def load_pdf_chunks():
    creds = get_credentials()
    client = storage.Client(credentials=creds)
    bucket = client.bucket(PDF_BUCKET)
    blobs = client.list_blobs(PDF_BUCKET, prefix=PDF_FOLDER)

    chunks = []
    skipped = 0

    for blob in blobs:
        if not blob.name.lower().endswith(".pdf"):
            continue
        try:
            content = blob.download_as_bytes()
            reader = PdfReader(io.BytesIO(content))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    chunks.append({
                        "source": f"{os.path.basename(blob.name)} (page {i+1})",
                        "text": text.strip()
                    })
        except Exception as e:
            skipped += 1
            print(f"⚠️ Skipped {blob.name}: {e}")
    
    print(f"✅ Loaded {len(chunks)} PDF chunks. Skipped {skipped} PDFs.")
    return chunks

# 🧠 Build FAISS index
def build_index(chunks):
    texts = [chunk["text"] for chunk in chunks]
    response = openai.embeddings.create(input=texts, model="text-embedding-3-small")
    embeddings = response.data
    vectors = np.array([e.embedding for e in embeddings]).astype("float32")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, chunks

# 🔁 Load all chunks at startup
@app.on_event("startup")
def startup_event():
    global all_chunks, faiss_index
    transcript_chunks = load_transcript_chunks()
    pdf_chunks = load_pdf_chunks()
    all_chunks = transcript_chunks + pdf_chunks
    faiss_index, _ = build_index(all_chunks)
    print(f"✅ Loaded {len(all_chunks)} chunks (from transcripts and PDFs)")

# 📥 Request model
class Question(BaseModel):
    question: str

# 💬 Question answering endpoint
@app.post("/ask")
def ask_question(payload: Question):
    question = payload.question
    q_embedding = openai.embeddings.create(
        input=[question], model="text-embedding-3-small"
    ).data[0].embedding

    D, I = faiss_index.search(np.array([q_embedding], dtype="float32"), k=5)
    top_chunks = [all_chunks[i] for i in I[0]]

    context = "\n\n".join([f"{chunk['source']}: {chunk['text']}" for chunk in top_chunks])
    
    system_prompt = (
        "You are a helpful assistant. Use the context below to answer the question. "
        "If information comes from a video or PDF, cite the source and page or timestamp."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    raw_answer = completion.choices[0].message.content
    clean_answer = ' '.join(raw_answer.split())  # Remove extra newlines, etc.

    return {
            "answer": clean_answer,  # Frontend shows this
            "sources": [chunk["source"] for chunk in top_chunks]  # Frontend uses these to create links
    }