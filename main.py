from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import faiss
import numpy as np
import json
import os
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account

load_dotenv()

app = FastAPI()

# ✅ CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://qu-demo-clipboardai.vercel.app"],  # Replace/add with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# GCS Config
BUCKET_NAME = "transcript_puzzle"
TRANSCRIPT_JSON_PATH = "transcripts/transcript_chunks.json"

def get_storage_client():
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path or not os.path.exists(credentials_path):
        raise RuntimeError("Valid GOOGLE_APPLICATION_CREDENTIALS path is required")
    
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    return storage.Client(credentials=credentials)

def load_transcript_chunks():
    storage_client = get_storage_client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(TRANSCRIPT_JSON_PATH)
    content = blob.download_as_text()
    return json.loads(content)

def build_index(chunks):
    texts = [chunk["text"] for chunk in chunks]
    response = openai.embeddings.create(input=texts, model="text-embedding-3-small")
    embeddings = response.data
    vectors = np.array([e.embedding for e in embeddings]).astype("float32")
    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index, chunks

@app.on_event("startup")
def startup_event():
    global transcript_chunks, faiss_index
    transcript_chunks = load_transcript_chunks()
    faiss_index, _ = build_index(transcript_chunks)
    print(f"✅ Loaded {len(transcript_chunks)} transcript chunks into index.")

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(payload: Question):
    question = payload.question
    q_embedding = openai.embeddings.create(input=[question], model="text-embedding-3-small").data[0].embedding
    D, I = faiss_index.search(np.array([q_embedding], dtype="float32"), k=5)
    relevant_chunks = [transcript_chunks[i] for i in I[0]]

    context = "\n\n".join([f"{chunk['source']}: {chunk['text']}" for chunk in relevant_chunks])
    
    system_prompt = (
        "You are a helpful assistant. Use the context below to answer the question. "
        "If from a video, cite the timestamp."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return {
        "answer": completion.choices[0].message.content,
        "sources": [chunk["source"] for chunk in relevant_chunks]
    }
