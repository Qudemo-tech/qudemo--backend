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

# ✅ CORS for frontend - adjust origins as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://qu-demo-clipboardai.vercel.app"],  # Change to your frontend URL(s) "https://qu-demo-clipboardai.vercel.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Google Cloud Storage bucket info
TRANSCRIPT_BUCKET = "transcript_puzzle"
TRANSCRIPT_JSON_PATH = "transcripts/transcript_chunks.json"

PDF_BUCKET = "puzzle_io"
PDF_FOLDER = "pdf/"

# Global mapping of downloaded video filenames to their YouTube URLs
VIDEO_URL_MAP = {
    "downloaded_video_0.mp4": "https://youtu.be/ZAGxqOT2l2U?si=uSwgsYfcqKMxAWGc",
    "downloaded_video_1.mp4": "https://youtu.be/ZAGxqOT2l2U?si=DJ0JsvvIBIz19cJ1",
    "downloaded_video_2.mp4": "https://youtu.be/_zRaJOF-trE?si=7ob6ZbLED2butzfa",
    "downloaded_video_3.mp4": "https://youtu.be/opV4Tmgepno?si=-9aHDmOvNeQbLVDY",
    "downloaded_video_4.mp4": "https://youtu.be/q2Rb2ZR5eyw?si=r3NYHGhDWE63puzm",
    "downloaded_video_5.mp4": "https://youtu.be/A_IVog6Vs3I?si=xUovSgHHxrj8jPLc",
    "downloaded_video_6.mp4": "https://youtu.be/Em8ixilyoEo?si=UVWgS9SOccmpytRP",
    "downloaded_video_7.mp4": "https://youtu.be/sIun13utbI4?si=89bQAHXd_KQ0opzE",
    "downloaded_video_8.mp4": "https://youtu.be/-6aSKEs94cs?si=ne1vxH5NC6VG0Cuu",
    "downloaded_video_9.mp4": "https://youtu.be/Dd2FxrAQQtI?si=WIr9qZwJkShqNNem",
    "downloaded_video_10.mp4": "https://youtu.be/7XivT1Ts2jU?si=UBhpiCKH9d4lSgRF",
    "downloaded_video_11.mp4": "https://youtu.be/Tt8ucqPwfzM?si=CJqwRIkxFZhI8oGn",
    "downloaded_video_12.mp4": "https://youtu.be/tbupLhuf-yo?si=DdI4JM1mu3N5e1wU"
}

def get_credentials():
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path or not os.path.exists(key_path):
        raise RuntimeError("Valid GOOGLE_APPLICATION_CREDENTIALS path is required")
    return service_account.Credentials.from_service_account_file(key_path)

def load_transcript_chunks():
    creds = get_credentials()
    client = storage.Client(credentials=creds)
    bucket = client.bucket(TRANSCRIPT_BUCKET)
    blob = bucket.blob(TRANSCRIPT_JSON_PATH)
    content = blob.download_as_text()
    chunks = json.loads(content)
    enriched_chunks = []
    for chunk in chunks:
        source = chunk.get("source", "")
        m = re.match(r"(.+\.mp4) \[(\d{2}):(\d{2}):(\d{2}),", source)
        if m:
            filename = m.group(1)
            h, mm, s = int(m.group(2)), int(m.group(3)), int(m.group(4))
            seconds = h * 3600 + mm * 60 + s
            yt_url = VIDEO_URL_MAP.get(filename)
            if yt_url:
                # Properly append timestamp query parameter (&t=...) if URL already has '?'
                if '?' in yt_url:
                    enriched_source = f"{yt_url}&t={seconds}"
                else:
                    enriched_source = f"{yt_url}?t={seconds}"
                enriched_chunks.append({"source": enriched_source, "text": chunk["text"]})
                continue
        enriched_chunks.append(chunk)
    return enriched_chunks

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

def build_index(chunks):
    print("⏳ Building FAISS index for chunks...")
    texts = [chunk["text"] for chunk in chunks]
    try:
        response = openai.embeddings.create(input=texts, model="text-embedding-3-small")
        embeddings = response.data
        vectors = np.array([e.embedding for e in embeddings]).astype("float32")
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        print("✅ FAISS index built successfully.")
        return index, chunks
    except Exception as e:
        print(f"❌ Error building FAISS index: {e}")
        raise

@app.on_event("startup")
def startup_event():
    global all_chunks, faiss_index
    print("🚀 Starting up backend...")

    # Load real data and build index
    transcript_chunks = load_transcript_chunks()
    pdf_chunks = load_pdf_chunks()
    all_chunks = transcript_chunks + pdf_chunks
    faiss_index, _ = build_index(all_chunks)
    print(f"✅ Loaded {len(all_chunks)} chunks (transcripts + PDFs)")

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(payload: Question):
    print(f"🔍 Received question: {payload.question}")

    try:
        q_embedding = openai.embeddings.create(
            input=[payload.question], model="text-embedding-3-small"
        ).data[0].embedding
        print("✅ Created question embedding.")
    except Exception as e:
        print(f"❌ Error creating question embedding: {e}")
        return {"error": "Failed to create question embedding."}

    try:
        D, I = faiss_index.search(np.array([q_embedding], dtype="float32"), k=5)
        top_chunks = [all_chunks[i] for i in I[0]]
        print(f"✅ Retrieved top {len(top_chunks)} chunks from FAISS.")
    except Exception as e:
        print(f"❌ FAISS search failed: {e}")
        return {"error": "Failed to search for relevant chunks."}

    context = "\n\n".join([f"{chunk['source']}: {chunk['text']}" for chunk in top_chunks])

    system_prompt = (
        "You are a concise and knowledgeable assistant. Based on the context below, answer the question with a clear and accurate summary. "
        "Prioritize quality over length. Only include the most relevant details. "
        "If the answer uses information from a video or PDF, cite the source (with page or timestamp)."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {payload.question}"

    try:
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        raw_answer = completion.choices[0].message.content
        print("✅ Received response from OpenAI chat completion.")
    except Exception as e:
        print(f"❌ OpenAI chat completion failed: {e}")
        return {"error": "Failed to generate answer."}

    clean_answer = ' '.join(raw_answer.split())

    # Find first video URL + timestamp (if any) from chunks
    first_video_source = None
    for chunk in top_chunks:
        if chunk["source"].startswith("http"):
            first_video_source = chunk["source"]
            break

    return {
        "answer": clean_answer,
        "sources": [chunk["source"] for chunk in top_chunks],
        "video_url": first_video_source  # Frontend can use this for video player
    }
