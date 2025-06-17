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
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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
FAQ_CSV_PATH = "csv/faq.csv"

VIDEO_URL_MAP = {
    "downloaded_video_0.mp4": "https://youtu.be/ZAGxqOT2l2U?si=pu-C_I-1DrHbQB2f",
    "downloaded_video_1.mp4": "https://youtu.be/_zRaJOF-trE?si=CEi69DKyrZtTHiBe",
    "downloaded_video_2.mp4": "https://youtu.be/o1ReLrUYPfY?si=bML0AHQxzHSxNTzK",
    "downloaded_video_3.mp4": "https://youtu.be/wR_JMXuUOpk?si=fSCK5Gb6w6PV7vPS",
    "downloaded_video_4.mp4": "https://youtu.be/WA4N_3Fdk2A?si=4qfH5STgCJLtFiXZ",
    "downloaded_video_5.mp4": "https://youtu.be/q9cbCws782M?si=LpZvM-1-NObRXQQq",
    "downloaded_video_6.mp4": "https://youtu.be/opV4Tmgepno?si=7YrE8KpTSKfQMzzZ",
    "downloaded_video_7.mp4": "https://youtu.be/q2Rb2ZR5eyw?si=4_li0YTvbFjHdXc9",
    "downloaded_video_8.mp4": "https://youtu.be/TzFIfvtL5mk?si=Kt8ffJyswHRK0STS",
}

class Question(BaseModel):
    question: str

FAQ_EMBEDDINGS = []
FAQ_DATA = []

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

def upload_faiss_index(local_path=FAISS_INDEX_PATH_LOCAL):
    creds = get_credentials()
    client = storage.Client(credentials=creds)
    bucket = client.bucket(TRANSCRIPT_BUCKET)
    blob = bucket.blob(FAISS_INDEX_PATH_GCS)
    blob.upload_from_filename(local_path)

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
    return enriched_chunks

def load_faqs():
    global FAQ_EMBEDDINGS, FAQ_DATA

    creds = get_credentials()
    client = storage.Client(credentials=creds)
    bucket = client.bucket(TRANSCRIPT_BUCKET)
    blob = bucket.blob(FAQ_CSV_PATH)
    content = blob.download_as_text()

    df = pd.read_csv(io.StringIO(content))
    df.columns = df.columns.str.strip().str.lower()  # Normalize headers ✅

    FAQ_DATA = df.to_dict(orient="records")

    # Embed FAQ questions
    questions = [entry["question"] for entry in FAQ_DATA]
    embeddings = []
    for i in range(0, len(questions), 10):
        resp = openai.embeddings.create(
            input=questions[i:i+10],
            model="text-embedding-3-small",
            timeout=20
        )
        batch_embeddings = [e.embedding for e in resp.data]
        embeddings.extend(batch_embeddings)

    FAQ_EMBEDDINGS = np.array(embeddings).astype("float32")
    print(f"✅ Loaded {len(FAQ_DATA)} FAQ entries.")


@app.on_event("startup")
def startup_event():
    global all_chunks, faiss_index
    all_chunks = load_transcript_chunks()
    faiss_index = load_faiss_index()
    load_faqs()

@app.post("/ask")
def ask_question(payload: Question):
    try:
        q_embedding = openai.embeddings.create(
            input=[payload.question],
            model="text-embedding-3-small",
            timeout=15
        ).data[0].embedding
    except Exception as e:
        return {"error": "Failed to create question embedding."}

    try:
        similarities = cosine_similarity(
            np.array([q_embedding]), FAQ_EMBEDDINGS
        )[0]

        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]

        if best_score > 0.85:
            return {
                "answer": FAQ_DATA[best_idx]["answer"],
                "sources": ["faq"],
                "video_url": None
            }
    except Exception as e:
        pass

    try:
        D, I = faiss_index.search(np.array([q_embedding], dtype="float32"), k=6)
        top_chunks = [all_chunks[i] for i in I[0]]
    except Exception as e:
        return {"error": f"FAISS search failed: {e}"}

    try:
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
    except Exception as e:
        best_chunk = top_chunks[0]

    try:
        context = "\n\n".join([
            f"{chunk['source']}: {chunk['text'][:500]}" for chunk in top_chunks
        ])

        system_prompt = (
            "You are a product expert bot with comprehensive knowledge of Puzzle.io, based on video transcripts and FAQs. "
            "Always prioritize official Puzzle.io sources when answering. "
            "If an answer is not available from those, use ChatGPT’s general knowledge — but clearly state it's not based on Puzzle.io content. "
            "Respond with clarity, confidence, and conciseness, using no more than 700 characters. "
            "Prefer bullet points or short paragraphs. "
            "Do not include inline citations. "
            "Avoid hallucinating; if unsure, say so honestly."
        )


        user_prompt = f"Context:\n{context}\n\nQuestion: {payload.question}"

        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=20
        )
        raw_answer = completion.choices[0].message.content
    except Exception as e:
        return {"error": "Failed to generate answer."}

    def strip_sources(text):
        return re.sub(r'\[source\]\([^)]+\)', '', text).strip()

    def format_answer(text):
        text = re.sub(r'\s*[-•]\s+', r'\n• ', text)
        text = re.sub(r'\s*\d+\.\s+', lambda m: f"\n{m.group(0)}", text)
        return re.sub(r'\n+', '\n', text).strip()

    raw_answer = strip_sources(raw_answer)
    clean_answer = format_answer(raw_answer)

    sources = [chunk["source"] for chunk in top_chunks]

    def extract_time(url):
        match = re.search(r"[?&]t=(\d+)", url)
        return int(match.group(1)) if match else float("inf")

    video_url = min(sources, key=extract_time, default=None)

    return {
        "answer": clean_answer,
        "sources": sources,
        "video_url": video_url
    }
