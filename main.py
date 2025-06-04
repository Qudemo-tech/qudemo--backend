from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import faiss
import numpy as np
import json
import os
import io
import re
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account
from PyPDF2 import PdfReader

load_dotenv()

app = FastAPI()

# ✅ CORS for frontend - adjust origins as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://qu-demo-clipboardai.vercel.app"],  # Change to your frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Google Cloud Storage bucket info
TRANSCRIPT_BUCKET = "transcript_puzzle"
TRANSCRIPT_JSON_PATH = "transcripts/transcript_chunks.json"
FAISS_INDEX_PATH_GCS = "faiss_indexes/faiss_index.bin"

PDF_BUCKET = "puzzle_io"
PDF_FOLDER = "pdf/"

VIDEO_URL_MAP = {
    "downloaded_video_0.mp4": "https://youtu.be/sIun13utbI4?si=BGnNPK1xGPWl9Xi4",
    "downloaded_video_1.mp4": "https://youtu.be/-6aSKEs94cs?si=FpAcrTwq5hnXgs_a",
    "downloaded_video_2.mp4": "https://youtu.be/Dd2FxrAQQtI?si=hlwwcd4yyzw7-07i",
    "downloaded_video_3.mp4": "https://youtu.be/7XivT1Ts2jU?si=mHsQa3WlmbVBzKpZ",
    "downloaded_video_4.mp4": "https://youtu.be/Tt8ucqPwfzM?si=hjuNOi876k0RHWhz",
    "downloaded_video_5.mp4": "https://youtu.be/tbupLhuf-yo?si=8KMdlQp1NM_joLeM",
    "downloaded_video_6.mp4": "https://youtu.be/Em8ixilyoEo?si=H6MmWhvvxTyabl8j",
    "downloaded_video_7.mp4": "https://youtu.be/A_IVog6Vs3I?si=rxXFiOj0qzBK_Rvb",
    "downloaded_video_8.mp4": "https://youtu.be/ZAGxqOT2l2U?si=jruJj1BTcdN4TAzQ",
    "downloaded_video_9.mp4": "https://youtu.be/_zRaJOF-trE?si=7ru8qoTluYiTxMbu",
    "downloaded_video_10.mp4": "https://youtu.be/o1ReLrUYPfY?si=mnQla0BTkXDFk9NO",
    "downloaded_video_11.mp4": "https://youtu.be/wR_JMXuUOpk?si=hW6Q2HR-JWwhEmPS",
    "downloaded_video_12.mp4": "https://youtu.be/WA4N_3Fdk2A?si=brMfbsaNNzVVZ10F",
    "downloaded_video_13.mp4": "https://youtu.be/q9cbCws782M?si=SpuGLsk9zGf9iEwV",
    "downloaded_video_14.mp4": "https://youtu.be/opV4Tmgepno?si=OkLfKS1975MGhLub",
    "downloaded_video_15.mp4": "https://youtu.be/q2Rb2ZR5eyw?si=HbX0iiVkjnQD_uOR",
    "downloaded_video_16.mp4": "https://youtu.be/TzFIfvtL5mk?si=QrT3KmhqYunltaRl",
    "downloaded_video_17.mp4": "https://youtu.be/vvmsA_EvPJA?si=UzykkVOeJjVQKtUH",
    "downloaded_video_18.mp4": "https://youtu.be/1EELDkH9tC8?si=c6mfu5fPs6C2J5RG",
    "downloaded_video_19.mp4": "https://youtu.be/1EELDkH9tC8?si=c6mfu5fPs6C2J5RG",  # duplicate
    "downloaded_video_20.mp4": "https://youtu.be/tF0uoicP9Q0?si=wqNQu9FFbmxlhGyg",
    "downloaded_video_21.mp4": "https://youtu.be/_TfLvzLrCXA?si=DFFUz7SIWLYz5u0H",
    "downloaded_video_22.mp4": "https://youtu.be/-b8az8mAE6k?si=jv5IIuRc5CIJ4cVF"
}


def get_credentials():
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path or not os.path.exists(key_path):
        raise RuntimeError("Valid GOOGLE_APPLICATION_CREDENTIALS path is required")
    return service_account.Credentials.from_service_account_file(key_path)

def download_faiss_index(local_path="faiss_index.bin"):
    creds = get_credentials()
    client = storage.Client(credentials=creds)
    bucket = client.bucket(TRANSCRIPT_BUCKET)
    blob = bucket.blob(FAISS_INDEX_PATH_GCS)
    if not blob.exists():
        raise RuntimeError(f"FAISS index file {FAISS_INDEX_PATH_GCS} not found in bucket {TRANSCRIPT_BUCKET}")
    blob.download_to_filename(local_path)
    print(f"✅ Downloaded FAISS index from GCS to {local_path}")

def load_faiss_index(local_path="faiss_index.bin"):
    if not os.path.exists(local_path):
        download_faiss_index(local_path)
    index = faiss.read_index(local_path)
    print(f"✅ Loaded FAISS index from {local_path}")
    return index

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

@app.on_event("startup")
def startup_event():
    global all_chunks, faiss_index
    print("🚀 Starting up backend...")

    # Load transcript + PDF chunks
    transcript_chunks = load_transcript_chunks()
    pdf_chunks = load_pdf_chunks()
    all_chunks = transcript_chunks + pdf_chunks
    print(f"✅ Loaded {len(all_chunks)} chunks (transcripts + PDFs)")

    # Load FAISS index from file downloaded from GCS
    faiss_index = load_faiss_index()
    print("✅ FAISS index loaded successfully on startup.")

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

    # 🔁 Rerank top chunks using GPT to find most relevant
    try:
        rerank_prompt = f"Question: {payload.question}\n\n"
        rerank_prompt += "Here are the chunks:\n"
        for i, chunk in enumerate(top_chunks):
            rerank_prompt += f"{i+1}. {chunk['text'][:500]}\n"  # trim long chunks

        rerank_prompt += "\nWhich chunk is the most relevant to the question above? Just give the number."

        rerank_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": rerank_prompt}
            ]
        )
        best_index = int(re.findall(r"\d+", rerank_response.choices[0].message.content)[0]) - 1
        best_chunk = top_chunks[best_index]
        print(f"✅ Reranked top chunk: #{best_index+1}")
    except Exception as e:
        print(f"⚠️ Rerank failed: {e}")
        best_chunk = top_chunks[0]  # fallback

    # ⬇️ Prepare context for answer
    context = "\n\n".join([f"{chunk['source']}: {chunk['text']}" for chunk in top_chunks])

    system_prompt = (
        "You are a concise, knowledgeable, and helpful assistant. Your task is to provide clear and accurate answers based on the context provided. "
        "Always prioritize quality and relevance over length—focus only on the most important details. "
        "If the information is derived from a video or PDF, include a citation with a timestamp (for videos) or page number (for PDFs). "
        "Present your answers in a well-organized and easy-to-understand format. Use paragraphs or bullet points if necessary to improve readability and structure."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {payload.question}"

    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
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

    # 🎯 Use reranked best_chunk for autoplay video
    first_video_source = best_chunk["source"] if best_chunk["source"].startswith("http") else None

    return {
        "answer": clean_answer,
        "sources": [chunk["source"] for chunk in top_chunks],
        "video_url": first_video_source
    }
