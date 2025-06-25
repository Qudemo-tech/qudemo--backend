from fastapi import APIRouter, FastAPI
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