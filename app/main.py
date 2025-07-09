from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from app.chatbot_logic import get_chatbot_response
from app.dash_app import create_dash_app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

@app.post("/ask-chatbot")
async def ask_chatbot(request: ChatRequest):
    """
    This endpoint receives a question from the frontend, gets a response
    from the Groq LLM via our chatbot logic, and returns the answer.
    """
    try:
        user_question = request.question
        answer = get_chatbot_response(user_question)
        return {"answer": answer}
    except Exception as e:
        print(f"Error in /ask-chatbot endpoint: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the chatbot request.")

@app.get("/")
def root():
    return HTMLResponse(content="<h3>✅ FastAPI backend with mounted Dash app and Chatbot is running.</h3>")

dash_app = create_dash_app()
app.mount("/dashboard", WSGIMiddleware(dash_app.server))
