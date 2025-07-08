# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

from dash_app import create_dash_app
from chatbot_logic import setup_rag_pipeline, generate_answer

class ChatQuery(BaseModel):
    question: str

# Dictionary to hold our application's state (the RAG chain)
app_state = {}

# Use the new lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup
    print("--- Server starting up ---")
    app_state["rag_chain"] = setup_rag_pipeline()
    yield
    # On shutdown
    print("--- Server shutting down ---")
    app_state.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, restrict to your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return HTMLResponse(content="<h3>✅ FastAPI backend is running.</h3>")

@app.post("/ask-chatbot")
async def ask_chatbot(query: ChatQuery):
    # Use the globally available RAG chain from app_state
    answer = await generate_answer(app_state.get("rag_chain"), query.question)
    return {"answer": answer}

# Mount the Dash app
dash_app = create_dash_app()
app.mount("/dashboard", WSGIMiddleware(dash_app.server))
