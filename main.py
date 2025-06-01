from fastapi import FastAPI, Query
from typing import List
from pydantic import BaseModel
import openai
import faiss
import numpy as np
import json
from google.cloud import storage
import os
from dotenv import load_dotenv

load_dotenv()  # load variables from .env file into environment



app = FastAPI()

# 🔑 Set your OpenAI key securely
openai.api_key = os.getenv("OPENAI_API_KEY")

# 📂 GCS Config
BUCKET_NAME = "transcript_puzzle"
TRANSCRIPT_JSON_PATH = "transcripts/transcript_chunks.json"

# Load from GCS
def load_transcript_chunks():
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(TRANSCRIPT_JSON_PATH)
    content = blob.download_as_text()
    return json.loads(content)

# Build FAISS index
def build_index(chunks):
    texts = [chunk["text"] for chunk in chunks]
    response = openai.embeddings.create(input=texts, model="text-embedding-3-small")
    embeddings = response.data
    vectors = np.array([e.embedding for e in embeddings]).astype("float32")
    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index, chunks

# Load & Index on startup
@app.on_event("startup")
def startup_event():
    global transcript_chunks, faiss_index
    transcript_chunks = load_transcript_chunks()
    faiss_index, _ = build_index(transcript_chunks)
    print(f"✅ Loaded {len(transcript_chunks)} transcript chunks into index.")

# User question schema
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

    user_prompt = f"""Context:\n{context}\n\nQuestion: {question}"""

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
