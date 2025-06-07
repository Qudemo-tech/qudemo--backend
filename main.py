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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://qu-demo-clipboardai.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

TRANSCRIPT_BUCKET = "transcript_puzzle"
TRANSCRIPT_JSON_PATH = "transcripts/transcript_chunks.json"
FAISS_INDEX_PATH_LOCAL = "faiss_index.bin"
FAISS_INDEX_PATH_GCS = "faiss_indexes/faiss_index.bin"

PDF_BUCKET = "puzzle_io_v2"
PDF_FOLDER = "pdf/"

VIDEO_URL_MAP = {
    "downloaded_video_0.mp4": "https://youtu.be/ZAGxqOT2l2U?si=OUD15LLjytENXQ_u",
    "downloaded_video_1.mp4": "https://youtu.be/_zRaJOF-trE?si=NWHuV4Cv9HmRHosB",
    "downloaded_video_2.mp4": "https://youtu.be/o1ReLrUYPfY?si=PyDBG11M1RbCdmPH",
    "downloaded_video_3.mp4": "https://youtu.be/wR_JMXuUOpk?si=HwDBDbP6AtfHQ8fc",
    "downloaded_video_4.mp4": "https://youtu.be/WA4N_3Fdk2A?si=E8qijFxRUmHHwAWM",
    "downloaded_video_5.mp4": "https://youtu.be/q9cbCws782M?si=ZTk19UzoILlITMNu",
    "downloaded_video_6.mp4": "https://youtu.be/opV4Tmgepno?si=enQRSeWop4rB1jbm",
    "downloaded_video_7.mp4": "https://youtu.be/q2Rb2ZR5eyw?si=Y2qjXKJtxiEeomwn",
    "downloaded_video_8.mp4": "https://youtu.be/TzFIfvtL5mk?si=63t_9WWox_Wl7fRX",
}


def get_credentials():
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path or not os.path.exists(key_path):
        raise RuntimeError("Valid GOOGLE_APPLICATION_CREDENTIALS path is required")
    return service_account.Credentials.from_service_account_file(key_path)


def download_faiss_index(local_path=FAISS_INDEX_PATH_LOCAL):
    creds = get_credentials()
    client = storage.Client(credentials=creds)
    bucket = client.bucket(TRANSCRIPT_BUCKET)
    blob = bucket.blob(FAISS_INDEX_PATH_GCS)
    if not blob.exists():
        raise RuntimeError(f"FAISS index {FAISS_INDEX_PATH_GCS} not found in bucket {TRANSCRIPT_BUCKET}")
    blob.download_to_filename(local_path)
    print(f"✅ Downloaded FAISS index to {local_path}")


def upload_faiss_index(local_path=FAISS_INDEX_PATH_LOCAL):
    creds = get_credentials()
    client = storage.Client(credentials=creds)
    bucket = client.bucket(TRANSCRIPT_BUCKET)
    blob = bucket.blob(FAISS_INDEX_PATH_GCS)
    blob.upload_from_filename(local_path)
    print(f"✅ Uploaded FAISS index to GCS: {FAISS_INDEX_PATH_GCS}")


def load_faiss_index(local_path=FAISS_INDEX_PATH_LOCAL):
    if not os.path.exists(local_path):
        download_faiss_index(local_path)
    return faiss.read_index(local_path)


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
            filename, h, m_, s = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            seconds = h * 3600 + m_ * 60 + s
            yt_url = VIDEO_URL_MAP.get(filename)
            if yt_url:
                enriched_source = f"{yt_url}&t={seconds}" if "?" in yt_url else f"{yt_url}?t={seconds}"
                enriched_chunks.append({
                    "source": enriched_source,
                    "text": chunk["text"],
                    "type": "video",
                    "context": chunk.get("context", "")
                })
                continue
        enriched_chunks.append({**chunk, "type": "video" if source.endswith(".mp4") else "pdf", "context": chunk.get("context", "")})
    return enriched_chunks


def load_pdf_chunks():
    creds = get_credentials()
    client = storage.Client(credentials=creds)
    blobs = client.list_blobs(PDF_BUCKET, prefix=PDF_FOLDER)
    chunks = []
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
                        "text": text.strip(),
                        "type": "pdf",
                        "context": f"Content from page {i+1} of {os.path.basename(blob.name)}"
                    })
        except Exception as e:
            print(f"⚠️ Skipped {blob.name}: {e}")
    return chunks


@app.on_event("startup")
def startup_event():
    global all_chunks, faiss_index
    print("🚀 Starting up backend...")
    transcript_chunks = load_transcript_chunks()
    pdf_chunks = load_pdf_chunks()
    all_chunks = transcript_chunks + pdf_chunks
    print(f"✅ Loaded {len(all_chunks)} chunks.")
    faiss_index = load_faiss_index()
    print("✅ FAISS index loaded.")


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask_question(payload: Question):
    print(f"🔍 Received question: {payload.question}")
    
    try:
        print("🔍 Creating question embedding...")
        q_embedding = openai.embeddings.create(
            input=[payload.question],
            model="text-embedding-3-small",
            timeout=15
        ).data[0].embedding
        print("✅ Got question embedding.")
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return {"error": "Failed to create question embedding."}

    try:
        print("🔍 Searching FAISS index...")
        D, I = faiss_index.search(np.array([q_embedding], dtype="float32"), k=6)
        top_chunks = [all_chunks[i] for i in I[0]]
        print(f"✅ Found top {len(top_chunks)} chunks.")
    except Exception as e:
        return {"error": f"FAISS search failed: {e}"}

    try:
        print("🔍 Reranking chunks...")
        rerank_prompt = f"Question: {payload.question}\n\nHere are the chunks:\n"
        for i, chunk in enumerate(top_chunks):
            snippet = chunk["text"][:500].strip().replace("\n", " ")
            rerank_prompt += f"{i+1}. [{chunk['type']}] {chunk.get('context','')}\n{snippet}\n\n"
        rerank_prompt += "Which chunk is most relevant to the question above? Just give the number."

        rerank_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": rerank_prompt}],
            timeout=20
        )
        best_index = int(re.findall(r"\d+", rerank_response.choices[0].message.content)[0]) - 1
        best_chunk = top_chunks[best_index]
        print(f"✅ Best chunk: {best_index + 1}")
    except Exception as e:
        print(f"⚠️ Reranking failed: {e}")
        best_chunk = top_chunks[0]

    try:
        print("🔍 Generating final answer...")
        context = "\n\n".join([
            f"{chunk['source']}: {chunk['text'][:500]}" for chunk in top_chunks
        ])

        system_prompt = (
            "You are a concise, knowledgeable, and helpful assistant specialized in answering questions about Puzzle.io. "
            "Use short answers (max 700 characters), cite PDFs or videos with timestamps when helpful, and use bullet points or short paragraphs."
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {payload.question}"

        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            timeout=20
        )
        raw_answer = completion.choices[0].message.content
        print("✅ Got final answer.")
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return {"error": "Failed to generate answer."}

    def format_answer(text):
        text = re.sub(r'\s*[-•]\s+', r'\n• ', text)
        text = re.sub(r'\s*\d+\.\s+', lambda m: f"\n{m.group(0)}", text)
        return re.sub(r'\n+', '\n', text).strip()

    clean_answer = format_answer(raw_answer)

    video_url = next(
        (chunk["source"] for chunk in top_chunks if chunk.get("type") == "video" and chunk.get("source", "").startswith("http")),
        None
    )

    return {
        "answer": clean_answer,
        "sources": [chunk["source"] for chunk in top_chunks],
        "video_url": video_url
    }


# @app.post("/rebuild_index")
# def rebuild_index():
#     global all_chunks, faiss_index

#     print("🔁 Rebuilding FAISS index...")

#     all_chunks = load_transcript_chunks() + load_pdf_chunks()
#     texts = [chunk["text"] for chunk in all_chunks]

#     try:
#         embeddings = []
#         for i in range(0, len(texts), 10):
#             response = openai.embeddings.create(
#                 input=texts[i:i+10],
#                 model="text-embedding-3-small",
#                 timeout=30
#             )
#             batch_embeddings = [e.embedding for e in response.data]
#             embeddings.extend(batch_embeddings)
#         embeddings_np = np.array(embeddings).astype("float32")
#         dim = len(embeddings_np[0])
#         index = faiss.IndexFlatL2(dim)
#         index.add(embeddings_np)
#         faiss.write_index(index, FAISS_INDEX_PATH_LOCAL)
#         upload_faiss_index()
#         faiss_index = index
#         return {"message": f"✅ Rebuilt FAISS index with {len(all_chunks)} chunks."}
#     except Exception as e:
#         return {"error": f"Failed to rebuild FAISS index: {e}"}
