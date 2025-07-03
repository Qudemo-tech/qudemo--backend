from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import answer_question

router = APIRouter()

class AskRequest(BaseModel):
    question: str

@router.post("/ask/puzzle")
def ask_puzzle(payload: AskRequest):
    return answer_question("puzzle", payload.question)

@router.post("/ask/mixpanel")
def ask_mixpanel(payload: AskRequest):
    return answer_question("mixpanel", payload.question)

# For direct FastAPI usage
app = FastAPI()
app.include_router(router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://qu-demo-clipboardai.vercel.app",
        "https://qudemo-waiting-list-git-v2-clipboardai.vercel.app",
        "https://www.qudemo.com",
        "https://qudemo.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 
